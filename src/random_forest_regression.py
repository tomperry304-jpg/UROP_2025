# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 22:06:11 2025

@author: tompe
"""

# -*- coding: utf-8 -*-
"""
random_forrest_regression_16.0.py  (viewer-friendly artifacts version)

- Trains the model for 'Ecological Class'
- Keeps wb_id aligned throughout
- Exports:
    artifacts/predictions_all.csv     (wb_id + Predicted Class + Confidence + Actual Class)
    artifacts/metrics.json            (accuracy, f1_macro on test split)
    plots/confusion_matrix.png        (test split confusion matrix)
- Preserves your original plots & PDP logic

Run:
    python random_forrest_regression_16.0.py \
      --excel_path WFD_SW_Classification_Status_and_Objectives_Cycle2_v4.xlsx \
      --parquet_path wims_wfd_merged.parquet \
      --target "Ecological Class"
"""

import argparse
import os
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import csr_matrix, hstack

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.base import is_classifier

# Repro
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Encoding config
MAX_CATEGORIES_FOR_ONEHOT = 20

# Optional 2D PDP pairs (use names that exist after encoding if you want these to render)
PDP_2D_PAIRS = [
    ("Ammonia(N)", "Orthophospht"),
    ("pH", "Temp Water"),
    ("Nitrate-N", "N Oxidised"),
    ("Temp Water", "Ammonia(N)"),
    ("Temp Water", "Nitrate-N"),
    ("Orthophospht", "Nitrate-N"),
    ("Ammonia(N)", "N Oxidised"),
    ("Dissolved Oxygen", "Ammonia(N)"),
    ("Dissolved Oxygen", "Orthophospht"),
    ("pH", "Ammonia(N)"),
    ("pH", "N Oxidised"),
    ("Suspended Solids", "Orthophospht"),
    ("Conductivity", "Nitrate-N"),
]

# --------------------------- IO helpers ---------------------------

def load_excel_data(excel_path: str) -> pd.DataFrame:
    """Load all sheets, skip second header row, drop >=50% missing columns."""
    all_sheets = []
    xls = pd.ExcelFile(excel_path, engine="openpyxl")
    for sheet in xls.sheet_names:
        df_sheet = pd.read_excel(excel_path, sheet_name=sheet, skiprows=1, engine="openpyxl")
        all_sheets.append(df_sheet)
    if not all_sheets:
        raise ValueError("No sheets found in Excel.")
    df_excel = pd.concat(all_sheets, ignore_index=True)
    thresh = len(df_excel) * 0.5
    df_excel = df_excel.loc[:, df_excel.isnull().sum() < thresh]
    return df_excel


def load_parquet_data(parquet_path: str) -> pd.DataFrame:
    """Load Parquet; keep columns with >= 1,000,000 non-null (as per your original)."""
    df = pd.read_parquet(parquet_path)
    non_null_counts = df.notnull().sum()
    cols_to_keep = non_null_counts[non_null_counts >= 1_000_000].index
    return df.loc[:, cols_to_keep]


def aggregate_parquet(df_parquet: pd.DataFrame) -> pd.DataFrame:
    """(Kept from your original for richer features if needed)"""
    numeric_cols = df_parquet.select_dtypes(include=np.number).columns.tolist()

    agg_df = df_parquet.groupby("wb_id")[numeric_cols].agg(
        ['mean', 'median', 'std', 'min', 'max',
         lambda x: x.quantile(0.25),
         lambda x: x.quantile(0.75)]
    )
    agg_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in agg_df.columns]
    agg_df = agg_df.rename(columns={
        **{col: col.replace('<lambda_0>', 'Q1').replace('<lambda_1>', 'Q3') for col in agg_df.columns}
    })

    for num_col in numeric_cols:
        q1_col = f"{num_col}_<lambda_0>"
        q3_col = f"{num_col}_<lambda_1>"
        if q1_col in agg_df.columns and q3_col in agg_df.columns:
            agg_df[f"{num_col}_IQR"] = agg_df[q3_col] - agg_df[q1_col]

    df_parquet['dynamic_threshold'] = df_parquet.groupby('variable')['result'].transform(
        lambda x: x.mean() + 2*x.std()
    )
    df_parquet['above_threshold'] = (df_parquet['result'] > df_parquet['dynamic_threshold']).astype(int)
    agg_extra = df_parquet.groupby('wb_id')['above_threshold'].agg(['sum', 'mean']).reset_index()
    agg_extra.rename(columns={'sum': 'count_above_thr', 'mean': 'prop_above_thr'}, inplace=True)

    if 'date' in df_parquet.columns:
        df_parquet['ordinal_time'] = pd.to_datetime(df_parquet['date']).map(pd.Timestamp.toordinal)
        slope_df = df_parquet.groupby('wb_id').apply(
            lambda g: np.polyfit(g['ordinal_time'], g['result'], 1)[0] if len(g) > 1 else 0
        ).reset_index(name='trend_slope')
    else:
        slope_df = pd.DataFrame({'wb_id': df_parquet['wb_id'].unique(), 'trend_slope': 0})

    agg_df = agg_df.reset_index()
    agg_df = agg_df.merge(agg_extra, on='wb_id', how='left')
    agg_df = agg_df.merge(slope_df, on='wb_id', how='left')
    return agg_df


def determine_target_type(series: pd.Series, cat_threshold: int = 15) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "categorical" if series.nunique() <= cat_threshold else "continuous"
    else:
        return "categorical"

# --------------------------- Plot helpers ---------------------------

def get_top_features(importances, feature_names, top_n=15):
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    top_features = [f for f, imp in feat_imp[:top_n]]
    return top_features, feat_imp[:top_n]


def plot_feature_importance(importances, feature_names, output_path, target_name=None):
    feats, imps = zip(*sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True))
    top_feats = list(feats[:15])
    top_imps = list(imps[:15])
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=top_imps, y=top_feats, palette='coolwarm')
    title = "Top 15 Feature Importances"
    if target_name:
        title += f" for predicting {target_name}"
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.001, p.get_y() + p.get_height()/2, f"{width:.2f}", va='center')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_feature_distributions(df, features, target_name, target_type, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df_plot = df.copy()
    df_plot = df_plot.loc[:, ~df_plot.columns.duplicated()]
    if target_type == "continuous":
        n_unique = df_plot[target_name].nunique()
        n_bins = min(5, n_unique)
        try:
            df_plot['target_group'] = pd.qcut(df_plot[target_name], q=n_bins, labels=False, duplicates='drop')
        except Exception:
            df_plot['target_group'] = pd.cut(df_plot[target_name], bins=n_bins, labels=False)
        hue_col = 'target_group'
    else:
        hue_col = target_name

    for feat in features:
        plt.figure(figsize=(8, 6))
        if pd.api.types.is_numeric_dtype(df_plot[feat]):
            sns.boxplot(x=hue_col, y=feat, data=df_plot)
            title = f"Boxplot of {feat} by {'Target' if target_type=='continuous' else target_name}"
            plt.title(title)
            plt.xlabel('Target Bin' if target_type=='continuous' else target_name)
            plt.ylabel(feat)
        else:
            sns.countplot(x=feat, hue=hue_col, data=df_plot)
            title = f"Countplot of {feat} by {'Target' if target_type=='continuous' else target_name}"
            plt.title(title)
            plt.xlabel(feat)
            plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        safe_feat = feat.replace(" ", "_")
        plot_type = "boxplot" if pd.api.types.is_numeric_dtype(df_plot[feat]) else "countplot"
        out_file = os.path.join(output_dir, f"{safe_feat}_{plot_type}.png")
        plt.savefig(out_file)
        plt.close()


def evaluate_and_plot_feature_performance(X, y, feature_names, output_path="plots/feature_score_scatter.png"):
    os.makedirs("plots", exist_ok=True)

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names if feature_names is not None else [f"feature_{i}" for i in range(X.shape[1])])

    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def normalize_labels(series):
        return series.astype(str).str.strip().str.lower()

    y_norm = normalize_labels(y)

    for feat in feature_names:
        acc_scores, f1_scores = [], []
        for train_idx, test_idx in skf.split(X[[feat]], y_norm):
            X_train, X_test = X.iloc[train_idx][[feat]], X.iloc[test_idx][[feat]]
            y_train, y_test = y_norm.iloc[train_idx], y_norm.iloc[test_idx]
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc_scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='macro'))
        results.append({"Feature": feat, "Accuracy": np.mean(acc_scores), "F1 Score": np.mean(f1_scores)})

    acc_scores, f1_scores = [], []
    for train_idx, test_idx in skf.split(X[feature_names], y_norm):
        X_train, X_test = X.iloc[train_idx][feature_names], X.iloc[test_idx][feature_names]
        y_train, y_test = y_norm.iloc[train_idx], y_norm.iloc[test_idx]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='macro'))
    results.append({"Feature": "All Features", "Accuracy": np.mean(acc_scores), "F1 Score": np.mean(f1_scores)})

    baseline_label = "2"  # if "Moderate" encoded as 2 in some flows; kept from original
    y_baseline = pd.Series([baseline_label] * len(y_norm))
    baseline_acc = accuracy_score(y_norm, y_baseline)
    baseline_f1 = f1_score(y_norm, y_baseline, average='macro')
    results.append({"Feature": "Predict Moderate", "Accuracy": baseline_acc, "F1 Score": baseline_f1})

    df_results = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.scatter(df_results["Accuracy"], df_results["F1 Score"], edgecolors='black')
    baseline_row = df_results[df_results["Feature"] == "Predict Moderate"].iloc[0]
    plt.scatter(baseline_row["Accuracy"], baseline_row["F1 Score"], s=80)
    for i, row in df_results.iterrows():
        plt.annotate(row["Feature"], (row["Accuracy"] + 0.001, row["F1 Score"] + 0.001), fontsize=8)
    plt.xlabel("Accuracy")
    plt.ylabel("F1 Score")
    plt.title("Accuracy vs F1 Score (Including Predict-Moderate Baseline)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# --------------------------- Modelling core ---------------------------

def train_models(X_train, y_train, target_type):
    clf = None
    reg = None
    encoder_or_bins = None

    if target_type == "categorical":
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_train.values.ravel())

        param_grid = {
            'n_estimators': [300, 500],
            'max_depth': [None, 20, 50],
            'min_samples_leaf': [1, 2, 5],
            'class_weight': ['balanced']
        }

        grid = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE),
            param_grid,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1
        )
        grid.fit(X_train, y_encoded)
        clf = grid.best_estimator_

        reg = RandomForestRegressor(
            n_estimators=500,
            max_depth=grid.best_params_['max_depth'],
            min_samples_leaf=grid.best_params_['min_samples_leaf'],
            random_state=RANDOM_STATE
        )
        reg.fit(X_train, y_encoded)
        encoder_or_bins = le

    else:
        n_bins = min(10, y_train.nunique())
        if n_bins >= 2:
            y_array = y_train.values.ravel()
            y_binned, bins = pd.qcut(y_array, q=n_bins, labels=False, retbins=True, duplicates='drop')
            clf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE)
            clf.fit(X_train, y_binned)
            encoder_or_bins = bins
        reg = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE)
        reg.fit(X_train, y_train.values.ravel())

    return clf, reg, encoder_or_bins


def main(args):
    # ------------------------ Load data ------------------------
    df_excel = load_excel_data(args.excel_path)
    df_excel = df_excel.drop(columns=["Overall Water Body Class"], errors="ignore")
    print(f"Excel data: {df_excel.shape}")
    df_parquet = load_parquet_data(args.parquet_path)
    print(f"Parquet data: {df_parquet.shape}")

    # Build a wide table (one col per variable mean) for frequent variables
    var_counts = df_parquet['variable'].value_counts()
    frequent_vars = var_counts[var_counts > 1e6].index.tolist()
    print(f"Found {len(frequent_vars)} frequent variables with > 1M records")
    df_filtered = df_parquet[df_parquet['variable'].isin(frequent_vars)]
    pivot_df = df_filtered.pivot_table(
        index='wb_id', columns='variable', values='result', aggfunc='mean'
    ).reset_index()

    target_col = args.target
    # Merge target from Excel
    df_excel_norm = df_excel.rename(columns={'Water Body ID': 'wb_id'})
    if target_col not in df_excel_norm.columns:
        raise ValueError(f"Target '{target_col}' not in Excel.")
    # Merge ecological status with features, drop missing targets, reset index
    merged = (
        pd.merge(
            df_excel_norm[['wb_id', target_col]],
            pivot_df,
            on='wb_id',
            how='inner'
            )
            .dropna(subset=[target_col])
            .reset_index(drop=True)          # ✅ important
        )

    # Filter out "Not assessed"
    mask_valid = merged[target_col].astype(str).str.lower().str.strip() != 'not assessed'
    merged = merged.loc[mask_valid].reset_index(drop=True)

    # Keep wb_id for later map + artifact export
    wb_ids_all = merged['wb_id'].astype(str).reset_index(drop=True)

    # Build modelling tables
    y = merged[target_col].astype(str).str.strip()
    X = merged.drop(columns=[target_col, 'wb_id'])

    # Optional sanity check
    assert len(wb_ids_all) == len(merged) == len(y) == len(X), \
        "Length mismatch between wb_id / X / y after filtering"

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("wb_ids_all length:", len(wb_ids_all))

    target_type = determine_target_type(y)
    print(f"Target '{target_col}' detected as {target_type}")

    # ------------------------ Split ------------------------
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, wb_ids_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # ------------------------ Encode ------------------------
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    # decide one-hot vs ordinal
    small_cat_cols = [c for c in cat_cols if X_train[c].nunique() <= MAX_CATEGORIES_FOR_ONEHOT]
    high_cat_cols = [c for c in cat_cols if X_train[c].nunique() > MAX_CATEGORIES_FOR_ONEHOT]

    # one-hot
    if small_cat_cols:
        ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')
        X_train_onehot = ohe.fit_transform(X_train[small_cat_cols].astype(str))
        X_test_onehot = ohe.transform(X_test[small_cat_cols].astype(str))
        onehot_feature_names = ohe.get_feature_names_out(small_cat_cols)
    else:
        ohe = None
        X_train_onehot = None
        X_test_onehot = None
        onehot_feature_names = []

    # ordinal
    if high_cat_cols:
        ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train_ord = ord_enc.fit_transform(X_train[high_cat_cols].astype(str))
        X_test_ord = ord_enc.transform(X_test[high_cat_cols].astype(str))
    else:
        ord_enc = None
        X_train_ord = None
        X_test_ord = None

    # numeric
    numeric_cols = [c for c in X_train.columns if c not in cat_cols]
    parts_train = []
    parts_test = []
    if numeric_cols:
        parts_train.append(csr_matrix(X_train[numeric_cols].values))
        parts_test.append(csr_matrix(X_test[numeric_cols].values))
    if high_cat_cols:
        parts_train.append(csr_matrix(X_train_ord))
        parts_test.append(csr_matrix(X_test_ord))
    if small_cat_cols:
        parts_train.append(X_train_onehot)
        parts_test.append(X_test_onehot)

    X_train_enc = hstack(parts_train).tocsr()
    X_test_enc = hstack(parts_test).tocsr()
    feature_names_enc = numeric_cols + high_cat_cols + list(onehot_feature_names)

    # ------------------------ Train initial models ------------------------
    clf, reg, encoder_or_bins = train_models(X_train_enc, y_train, target_type)

    # get importances
    if target_type == "categorical" and clf is not None:
        importances = clf.feature_importances_
    else:
        importances = reg.feature_importances_
    top_feats, top_importances = get_top_features(importances, feature_names_enc, top_n=15)
    print("Top 15 features:")
    for f, imp in top_importances:
        print(f"  {f}: {imp:.4f}")

    # ------------------------ Correlation / CV / Plots ------------------------
    # Create DataFrame X_full for plotting top features where possible
    X_full_plot = X.copy()
    # add dummy columns for one-hot top-feats (if any)
    for feat in top_feats:
        if feat not in X_full_plot.columns:
            for col in cat_cols:
                if feat.startswith(str(col) + "_"):
                    cat_val = feat[len(col)+1:]
                    X_full_plot[feat] = (X[col].astype(str) == cat_val).astype(int)
                    break

    os.makedirs("plots", exist_ok=True)
    corr_df = X_full_plot[top_feats].dropna()
    if not corr_df.empty:
        corr_matrix = corr_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Pearson Correlation Matrix of Top 15 Features")
        plt.tight_layout()
        plt.savefig("plots/correlation_matrix.png")
        plt.close()

    # CV on top features using raw (not encoded) columns where possible
    # (kept from original approach)
    clf_cv = RandomForestClassifier(random_state=RANDOM_STATE)
    try:
        scores = cross_val_score(clf_cv, X_full_plot[top_feats].fillna(0), y.values.ravel(),
                                 cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                                 scoring='accuracy')
        plt.figure(figsize=(6, 6))
        sns.boxplot(y=scores, color='lightgreen')
        plt.title("5-fold Cross-validation Accuracy")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig("plots/cv_accuracy_boxplot.png")
        plt.close()
    except Exception as e:
        print(f"CV boxplot skipped: {e}")

    # Feature importance plot
    plot_feature_importance(importances, feature_names_enc, "plots/feature_importance.png", target_col)

    # ------------------------ Select top features & VT ------------------------
    feat_index_map = {name: idx for idx, name in enumerate(feature_names_enc)}
    top_indices = [feat_index_map[f] for f in top_feats if f in feat_index_map]

    # Slice encoded matrices to top features
    X_train_sel_enc = X_train_enc[:, top_indices]
    X_test_sel_enc = X_test_enc[:, top_indices]

    # Convert to dense DataFrames for plotting/compat if needed
    X_train_sel_df = pd.DataFrame(X_train_sel_enc.toarray(), columns=top_feats)
    X_test_sel_df = pd.DataFrame(X_test_sel_enc.toarray(), columns=top_feats)

    # VarianceThreshold
    vt = VarianceThreshold(threshold=1e-5)
    X_train_sel = vt.fit_transform(X_train_sel_df)
    X_test_sel = vt.transform(X_test_sel_df)

    # ------------------------ Retrain on top features ------------------------
    clf2, reg2, encoder_or_bins2 = train_models(X_train_sel, y_train, target_type)

    # ------------------------ Test predictions ------------------------
    if target_type == "categorical":
        # RF classifier on VT-top-features
        clf_pred_test = clf2.predict(X_test_sel)

        # Optional: blend with reg if you wish (kept simple here)
        final_pred = clf_pred_test

        # Save a simple test predictions CSV (optional)
        # Align shapes for the CSV
        test_out = pd.DataFrame({"wb_id": id_test.values,
                                 "True_Target": y_test.values,
                                 "Pred_Classifier": final_pred})
        test_out.to_csv("model_predictions.csv", index=False)

        # Metrics + confusion matrix image for the viewer app
        try:
            acc = accuracy_score(y_test, final_pred)
            f1 = f1_score(y_test, final_pred, average='macro')
            os.makedirs("artifacts", exist_ok=True)
            with open("artifacts/metrics.json", "w") as f:
                json.dump({"accuracy": float(acc), "f1_macro": float(f1)}, f, indent=2)

            cm = confusion_matrix(y_test, final_pred, labels=clf2.classes_)
            fig_cm, ax = plt.subplots(figsize=(5.5, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf2.classes_)
            disp.plot(ax=ax, colorbar=False)
            plt.title("Confusion Matrix (test split)")
            plt.tight_layout()
            plt.savefig("plots/confusion_matrix.png")
            plt.close(fig_cm)
            print(f"Saved metrics.json (acc={acc:.3f}, f1={f1:.3f}) and confusion_matrix.png")
        except Exception as e:
            print(f"Warning: could not save metrics/confusion matrix: {e}")

    else:
        # continuous branch (not used for Ecological Class)
        reg_pred = reg2.predict(X_test_sel)
        final_pred = reg_pred  # simplified

    # ------------------------ Feature distributions & eval scatter ------------------------
    # For distributions we need a dataset with target + top features
    df_plot = X_full_plot[top_feats].copy()
    df_plot[target_col] = y.values
    df_plot = df_plot.dropna()
    if target_type == "categorical":
        df_plot[target_col] = df_plot[target_col].astype(str)
        for feat in top_feats:
            if feat not in X.columns:
                df_plot[feat] = df_plot[feat].astype('category')
    plot_feature_distributions(df_plot, top_feats, target_col, target_type, output_dir="feature_plots")

    # Evaluate single-feature vs all (scatter)
    try:
        evaluate_and_plot_feature_performance(X_test_sel, pd.Series(LabelEncoder().fit_transform(y_test)),
                                              top_feats, output_path="plots/feature_score_scatter.png")
    except Exception as e:
        print(f"Scatter plot skipped: {e}")

    # ------------------------ PREDICTIONS FOR ALL wb_id ------------------------
    # Build encoded matrix for ALL rows -> select same top feature columns -> apply same VT
    def encode_like_train(df_in: pd.DataFrame):
        parts = []
        # numeric
        if numeric_cols:
            parts.append(csr_matrix(df_in[numeric_cols].values))
        # ordinal
        if high_cat_cols:
            parts.append(csr_matrix(ord_enc.transform(df_in[high_cat_cols].astype(str))))
        # one-hot
        if small_cat_cols:
            parts.append(ohe.transform(df_in[small_cat_cols].astype(str)))
        if not parts:
            return csr_matrix((len(df_in), 0))
        return hstack(parts).tocsr()

    X_enc_all = encode_like_train(X)
    X_all_sel_enc = X_enc_all[:, top_indices]
    X_all_sel = vt.transform(pd.DataFrame(X_all_sel_enc.toarray(), columns=top_feats))

    # Predict for ALL rows with the top-features model
    if target_type == "categorical":
        pred_all = clf2.predict(X_all_sel)
        try:
            proba_all = clf2.predict_proba(X_all_sel).max(axis=1)
        except Exception:
            proba_all = np.full(len(pred_all), np.nan)

        pred_all_df = pd.DataFrame({
            "wb_id": wb_ids_all,
            "Predicted Class": pred_all,
            "Prediction Confidence": proba_all
        })

        # attach actual if available (same length)
        if len(pred_all_df) == len(y):
            pred_all_df["Actual Class"] = y.values

        os.makedirs("artifacts", exist_ok=True)
        pred_all_df.to_csv("artifacts/predictions_all.csv", index=False)
        print("Saved artifacts/predictions_all.csv")
    else:
        pred_all_df = pd.DataFrame({
            "wb_id": wb_ids_all,
            "Predicted Value": reg2.predict(X_all_sel)
        })
        os.makedirs("artifacts", exist_ok=True)
        pred_all_df.to_csv("artifacts/predictions_all.csv", index=False)
        print("Saved artifacts/predictions_all.csv (continuous)")

    # ------------------------ Return for PDPs etc. ------------------------
    return final_pred, (clf2 if target_type == "categorical" else reg2), X_train_sel, top_feats

    # ===== Save predictions for ALL wb_id =====
    try:
        # Predict for all rows
        pred_all = clf2.predict(X_all_sel)  # X_all_sel must be same encoded form as training data

        try:
            proba_all = clf2.predict_proba(X_all_sel).max(axis=1)
        except Exception:
            proba_all = np.full(len(pred_all), np.nan)

        pred_all_df = pd.DataFrame({
            "wb_id": wb_ids_all,
            "Predicted Class": pred_all,
            "Prediction Confidence": proba_all
       })

        # Optional: attach actual class if available
        if len(y) == len(pred_all_df):
            pred_all_df["Actual Class"] = y.values

        os.makedirs("artifacts", exist_ok=True)
        pred_all_df.to_csv("artifacts/predictions_all.csv", index=False)
        print("✅ Saved full predictions to artifacts/predictions_all.csv")
    except Exception as e:
        print(f"⚠ Could not save predictions_all.csv: {e}")
    
    
    
def explain_model(model, X, feature_names=None, target_name="target", top_n=15):
    """Generate PDPs and text trends (kept from your original)."""
    if feature_names is None:
        try:
            feature_names = list(X.columns)
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:top_n]
    else:
        top_idx = np.arange(min(top_n, X.shape[1]))
    top_features = [feature_names[i] for i in top_idx]

    from sklearn.base import is_classifier as _is_classifier
    is_class = _is_classifier(model)

    explanations = []
    for idx, fname in zip(top_idx, top_features):
        try:
            pdp_res = partial_dependence(model, X, [idx], feature_names=feature_names, kind='both')
        except TypeError:
            pdp_res = partial_dependence(model, X, [idx], feature_names=feature_names)
        grid = pdp_res["values"][0] if "values" in pdp_res else pdp_res["grid_values"][0]
        avg_data = pdp_res["average"]
        avg_preds = np.ravel(avg_data[0]) if getattr(avg_data, "ndim", 1) > 1 else np.ravel(avg_data)
        diff = np.diff(avg_preds)
        if np.all(diff >= 0):
            trend = f"as {fname} increases, predicted {target_name} tends to increase"
        elif np.all(diff <= 0):
            trend = f"as {fname} increases, predicted {target_name} tends to decrease"
        else:
            if np.allclose(avg_preds, avg_preds[0], atol=1e-3):
                trend = f"{fname} has little effect on {target_name}"
            else:
                trend = f"{fname} has a non-monotonic effect on {target_name}"
        explanations.append(f"Feature '{fname}': {trend}.")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grid, avg_preds, label="Average prediction")
        indiv_curves = pdp_res.get("individual", None)
        if indiv_curves is not None:
            if getattr(indiv_curves, "ndim", 1) > 2:
                indiv_curves = indiv_curves[0]
            std = np.std(indiv_curves, axis=0)
            ax.fill_between(grid, avg_preds - std, avg_preds + std, alpha=0.3, label="± 1 std")
        ax.set_xlabel(fname)
        ax.set_ylabel(target_name)
        ax.set_title(f"PDP of {fname}")
        if indiv_curves is not None:
            ax.legend()
        plt.tight_layout()
        out_file = f"plots/pdp_{fname.replace(' ','_')}.png"
        Path("plots").mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, bbox_inches="tight")
        plt.close()

    print("\n".join(explanations))

    if PDP_2D_PAIRS:
        for f1, f2 in PDP_2D_PAIRS:
            if f1 in feature_names and f2 in feature_names:
                idx1 = feature_names.index(f1)
                idx2 = feature_names.index(f2)
            else:
                continue
            X_pdp = X.copy()
            if len(X_pdp) > 2000:
                X_pdp = X_pdp.sample(n=2000, random_state=RANDOM_STATE)
            try:
                display = PartialDependenceDisplay.from_estimator(
                    model, X_pdp, [(idx1, idx2)],
                    feature_names=feature_names, grid_resolution=25
                )
                plt.tight_layout()
                out_file = f"plots/pdp_2d_{f1}_{f2}.png"
                plt.savefig(out_file, bbox_inches="tight")
                plt.close()
                print(f"2D PDP saved as {out_file}")
            except Exception as e:
                print(f"Skipped 2D PDP for {f1} & {f2}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data processing and modeling")
    parser.add_argument("--excel_path", type=str, default="WFD_SW_Classification_Status_and_Objectives_Cycle2_v4.xlsx")
    parser.add_argument("--parquet_path", type=str, default="wims_wfd_merged.parquet")
    parser.add_argument("--target", type=str, default="Ecological Class")
    args = parser.parse_args()

    final_pred, model, X_train_used, feature_names_used = main(args)
    explain_model(model, X_train_used, feature_names=feature_names_used, target_name=args.target, top_n=15)

    print("\nDone. Key outputs for the Streamlit viewer:")
    print("  - artifacts/predictions_all.csv")
    print("  - artifacts/metrics.json")
    print("  - plots/confusion_matrix.png")
    print("  - plots/feature_importance.png, plots/correlation_matrix.png, plots/cv_accuracy_boxplot.png, plots/pdp_*.png")