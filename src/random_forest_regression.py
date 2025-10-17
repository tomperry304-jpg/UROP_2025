# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 01:38:39 2025

@author: tompe
"""

# -*- coding: utf-8 -*-
"""
random_forrest_regression_18.0.py  —  robust, artifact-exporting version

What’s new vs. 17.0:
- NaN-safe: SimpleImputer(strategy="median") on numeric columns, reused for ALL paths
- Consistent encoding: same OneHot/Ordinal + VarianceThreshold + Top-Feature selection everywhere
- Saves predictions for ALL water bodies: artifacts/predictions_all.csv
- Saves test metrics: artifacts/metrics.json  + plots/confusion_matrix.png
- Saves the whole inference stack (model + encoders + imputer + VT + feature lists + class list)

Run:
    python random_forrest_regression_18.0.py \
        --excel_path WFD_SW_Classification_Status_and_Objectives_Cycle2_v4.xlsx \
        --parquet_path wims_wfd_merged.parquet \
        --target "Ecological Class"

Requires: pandas, numpy, matplotlib, seaborn, scikit-learn, pyarrow or fastparquet, openpyxl, joblib, scipy
"""

import argparse
import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import csr_matrix, hstack

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.impute import SimpleImputer
from sklearn.base import is_classifier
import joblib

# ---------------------------- Config ----------------------------
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

MAX_CATEGORIES_FOR_ONEHOT = 20  # <=20 → one-hot; >20 → ordinal

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

ARTIFACTS_DIR = Path("artifacts")
PLOTS_DIR = Path("plots")
FEATURE_PLOTS_DIR = Path("feature_plots")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)
FEATURE_PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# ---------------------------- IO helpers ----------------------------
def load_excel_data(excel_path: str) -> pd.DataFrame:
    """Load all sheets, skip repeated header row, drop >=50% missing columns."""
    all_sheets = []
    xls = pd.ExcelFile(excel_path, engine="openpyxl")
    for sheet in xls.sheet_names:
        df_sheet = pd.read_excel(excel_path, sheet_name=sheet, skiprows=1, engine="openpyxl")
        all_sheets.append(df_sheet)
    if not all_sheets:
        raise ValueError("No sheets found in Excel.")
    df_excel = pd.concat(all_sheets, ignore_index=True)
    # drop very sparse cols
    thresh = len(df_excel) * 0.5
    df_excel = df_excel.loc[:, df_excel.isnull().sum() < thresh]
    df_excel.columns = df_excel.columns.str.strip()
    return df_excel


def load_parquet_data(parquet_path: str) -> pd.DataFrame:
    """Load Parquet; keep columns with >= 1,000,000 non-null (matches your original)."""
    df = pd.read_parquet(parquet_path)
    non_null_counts = df.notnull().sum()
    cols_to_keep = non_null_counts[non_null_counts >= 1_000_000].index
    return df.loc[:, cols_to_keep]


def determine_target_type(series: pd.Series, cat_threshold: int = 15) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "categorical" if series.nunique() <= cat_threshold else "continuous"
    else:
        return "categorical"

# ---------------------------- Plot helpers ----------------------------
def get_top_features(importances, feature_names, top_n=15):
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    top_features = [f for f, imp in feat_imp[:top_n]]
    return top_features, feat_imp[:top_n]


def plot_feature_importance(importances, feature_names, output_path, target_name=None):
    feats, imps = zip(*sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True))
    feats = list(feats)[:15]
    imps = list(imps)[:15]
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=imps, y=feats, palette="coolwarm")
    title = "Top 15 Feature Importances"
    if target_name:
        title += f" for predicting {target_name}"
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.001, p.get_y() + p.get_height()/2, f"{width:.2f}", va="center")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_feature_distributions(df, features, target_name, target_type, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df_plot = df.copy()
    df_plot = df_plot.loc[:, ~df_plot.columns.duplicated()]
    hue_col = target_name if target_type == "categorical" else "target_group"
    if target_type == "continuous":
        n_unique = df_plot[target_name].nunique()
        n_bins = min(5, n_unique)
        try:
            df_plot["target_group"] = pd.qcut(df_plot[target_name], q=n_bins, labels=False, duplicates="drop")
        except Exception:
            df_plot["target_group"] = pd.cut(df_plot[target_name], bins=n_bins, labels=False)

    for feat in features:
        plt.figure(figsize=(8, 6))
        if pd.api.types.is_numeric_dtype(df_plot[feat]):
            sns.boxplot(x=hue_col, y=feat, data=df_plot)
            title = f"Boxplot of {feat} by {target_name if target_type=='categorical' else 'Target Bins'}"
            plt.title(title)
            plt.xlabel(target_name if target_type == "categorical" else "Target bins")
            plt.ylabel(feat)
        else:
            sns.countplot(x=feat, hue=hue_col, data=df_plot)
            plt.title(f"Countplot of {feat} by {target_name}")
            plt.xlabel(feat)
            plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
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

    y_norm = y.astype(str).str.strip().str.lower()

    for feat in feature_names:
        acc_scores, f1_scores = [], []
        for train_idx, test_idx in skf.split(X[[feat]], y_norm):
            X_train, X_test = X.iloc[train_idx][[feat]], X.iloc[test_idx][[feat]]
            y_train, y_test = y_norm.iloc[train_idx], y_norm.iloc[test_idx]
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc_scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average="macro"))
        results.append({"Feature": feat, "Accuracy": float(np.mean(acc_scores)), "F1 Score": float(np.mean(f1_scores))})

    acc_scores, f1_scores = [], []
    for train_idx, test_idx in skf.split(X[feature_names], y_norm):
        X_train, X_test = X.iloc[train_idx][feature_names], X.iloc[test_idx][feature_names]
        y_train, y_test = y_norm.iloc[train_idx], y_norm.iloc[test_idx]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average="macro"))
    results.append({"Feature": "All Features", "Accuracy": float(np.mean(acc_scores)), "F1 Score": float(np.mean(f1_scores))})

    baseline_label = "2"  # kept from your original baseline logic
    y_baseline = pd.Series([baseline_label] * len(y_norm))
    baseline_acc = accuracy_score(y_norm, y_baseline)
    baseline_f1 = f1_score(y_norm, y_baseline, average="macro")
    results.append({"Feature": "Predict Moderate", "Accuracy": float(baseline_acc), "F1 Score": float(baseline_f1)})

    df_results = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.scatter(df_results["Accuracy"], df_results["F1 Score"], edgecolors="black")
    baseline_row = df_results[df_results["Feature"] == "Predict Moderate"].iloc[0]
    plt.scatter(baseline_row["Accuracy"], baseline_row["F1 Score"], s=80)
    for _, row in df_results.iterrows():
        plt.annotate(row["Feature"], (row["Accuracy"] + 0.001, row["F1 Score"] + 0.001), fontsize=8)
    plt.xlabel("Accuracy")
    plt.ylabel("F1 Score")
    plt.title("Accuracy vs F1 Score (Including Predict-Moderate Baseline)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ---------------------------- Model training ----------------------------
def train_models(X_train, y_train, target_type):
    clf = None
    reg = None
    encoder_or_bins = None

    if target_type == "categorical":
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_train.values.ravel())

        param_grid = {
            "n_estimators": [300, 500],
            "max_depth": [None, 20, 50],
            "min_samples_leaf": [1, 2, 5],
            "class_weight": ["balanced"],
        }

        grid = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE),
            param_grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1,
        )
        # X_train arrives NaN-free (we imputed earlier)
        grid.fit(X_train, y_encoded)
        clf = grid.best_estimator_

        reg = RandomForestRegressor(
            n_estimators=500,
            max_depth=grid.best_params_["max_depth"],
            min_samples_leaf=grid.best_params_["min_samples_leaf"],
            random_state=RANDOM_STATE,
        )
        reg.fit(X_train, y_encoded)
        encoder_or_bins = le

    else:
        n_bins = min(10, y_train.nunique())
        if n_bins >= 2:
            y_array = y_train.values.ravel()
            y_binned, bins = pd.qcut(y_array, q=n_bins, labels=False, retbins=True, duplicates="drop")
            clf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE)
            clf.fit(X_train, y_binned)
            encoder_or_bins = bins
        reg = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE)
        reg.fit(X_train, y_train.values.ravel())

    return clf, reg, encoder_or_bins

# ---------------------------- Main ----------------------------
def main(args):
    # 1) Load data
    df_excel = load_excel_data(args.excel_path)
    df_excel = df_excel.drop(columns=["Overall Water Body Class"], errors="ignore")
    print(f"Excel data: {df_excel.shape}")

    df_parquet = load_parquet_data(args.parquet_path)
    print(f"Parquet data: {df_parquet.shape}")

    # 2) Build a wide (mean by wb_id x variable) for frequent variables
    var_counts = df_parquet["variable"].value_counts()
    frequent_vars = var_counts[var_counts > 1_000_000].index.tolist()
    print(f"Found {len(frequent_vars)} frequent variables with > 1M records")
    df_filtered = df_parquet[df_parquet["variable"].isin(frequent_vars)]
    pivot_df = df_filtered.pivot_table(
        index="wb_id", columns="variable", values="result", aggfunc="mean"
    ).reset_index()

    # 3) Merge target from Excel + clean
    target_col = args.target
    df_excel_norm = df_excel.rename(columns={"Water Body ID": "wb_id"})
    if target_col not in df_excel_norm.columns:
        raise ValueError(f"Target '{target_col}' not in Excel.")

    merged = (
        pd.merge(df_excel_norm[["wb_id", target_col]], pivot_df, on="wb_id", how="inner")
        .dropna(subset=[target_col])
        .reset_index(drop=True)
    )

    mask_valid = merged[target_col].astype(str).str.lower().str.strip() != "not assessed"
    merged = merged.loc[mask_valid].reset_index(drop=True)

    # Keep aligned wb_id
    wb_ids_all = merged["wb_id"].astype(str).reset_index(drop=True)

    # Modelling table
    y = merged[target_col].astype(str).str.strip()
    X = merged.drop(columns=[target_col, "wb_id"])

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("wb_ids_all length:", len(wb_ids_all))

    target_type = determine_target_type(y)
    print(f"Target '{target_col}' detected as {target_type}")

    # 4) Split
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, wb_ids_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 5) Encoding plan
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    small_cat_cols = [c for c in cat_cols if X_train[c].nunique() <= MAX_CATEGORIES_FOR_ONEHOT]
    high_cat_cols = [c for c in cat_cols if X_train[c].nunique() > MAX_CATEGORIES_FOR_ONEHOT]
    numeric_cols = [c for c in X_train.columns if c not in cat_cols]

    # One-hot (fit on train)
    if small_cat_cols:
        ohe = OneHotEncoder(sparse=True, handle_unknown="ignore")
        X_train_onehot = ohe.fit_transform(X_train[small_cat_cols].astype(str))
        X_test_onehot = ohe.transform(X_test[small_cat_cols].astype(str))
        onehot_feature_names = ohe.get_feature_names_out(small_cat_cols)
    else:
        ohe, X_train_onehot, X_test_onehot, onehot_feature_names = None, None, None, []

    # Ordinal (fit on train)
    if high_cat_cols:
        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_ord = ord_enc.fit_transform(X_train[high_cat_cols].astype(str))
        X_test_ord = ord_enc.transform(X_test[high_cat_cols].astype(str))
    else:
        ord_enc, X_train_ord, X_test_ord = None, None, None

    # Numeric imputer (fit on train) → median
    imputer_num = None
    if numeric_cols:
        imputer_num = SimpleImputer(strategy="median")
        X_train_num = pd.DataFrame(
            imputer_num.fit_transform(X_train[numeric_cols]),
            columns=numeric_cols,
            index=X_train.index,
        )
        X_test_num = pd.DataFrame(
            imputer_num.transform(X_test[numeric_cols]),
            columns=numeric_cols,
            index=X_test.index,
        )
    else:
        X_train_num = pd.DataFrame(index=X_train.index)
        X_test_num = pd.DataFrame(index=X_test.index)

    # Build encoded matrices
    parts_train, parts_test = [], []
    if not X_train_num.empty:
        parts_train.append(csr_matrix(X_train_num.values))
        parts_test.append(csr_matrix(X_test_num.values))
    if high_cat_cols:
        parts_train.append(csr_matrix(X_train_ord))
        parts_test.append(csr_matrix(X_test_ord))
    if small_cat_cols:
        parts_train.append(X_train_onehot)
        parts_test.append(X_test_onehot)

    X_train_enc = hstack(parts_train).tocsr()
    X_test_enc = hstack(parts_test).tocsr()
    feature_names_enc = numeric_cols + high_cat_cols + list(onehot_feature_names)

    # 6) Train initial models (GridSearch for classifier)
    clf, reg, encoder_or_bins = train_models(X_train_enc, y_train, target_type)

    # Importances & top features
    if target_type == "categorical" and clf is not None:
        importances = clf.feature_importances_
    else:
        importances = reg.feature_importances_
    top_feats, top_importances = get_top_features(importances, feature_names_enc, top_n=15)
    print("Top 15 features:")
    for f, imp in top_importances:
        print(f"  {f}: {imp:.4f}")

    # 7) Correlation matrix over top features (on raw-ish table)
    X_full_plot = X.copy()
    for feat in top_feats:
        if feat not in X_full_plot.columns:
            for col in cat_cols:
                if feat.startswith(str(col) + "_"):
                    cat_val = feat[len(col) + 1 :]
                    X_full_plot[feat] = (X[col].astype(str) == cat_val).astype(int)
                    break

    corr_df = X_full_plot[top_feats].dropna()
    if not corr_df.empty:
        corr_matrix = corr_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Pearson Correlation Matrix of Top 15 Features")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "correlation_matrix.png")
        plt.close()

    # CV accuracy boxplot on top features (raw-constructed)
    clf_cv = RandomForestClassifier(random_state=RANDOM_STATE)
    try:
        scores = cross_val_score(
            clf_cv,
            X_full_plot[top_feats].fillna(0),
            y.values.ravel(),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring="accuracy",
        )
        plt.figure(figsize=(6, 6))
        sns.boxplot(y=scores, color="lightgreen")
        plt.title("5-fold Cross-validation Accuracy")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "cv_accuracy_boxplot.png")
        plt.close()
    except Exception as e:
        print(f"CV boxplot skipped: {e}")

    # Feature importance plot
    plot_feature_importance(importances, feature_names_enc, PLOTS_DIR / "feature_importance.png", target_col)

    # 8) Select top features + VarianceThreshold
    feat_index_map = {name: idx for idx, name in enumerate(feature_names_enc)}
    top_indices = [feat_index_map[f] for f in top_feats if f in feat_index_map]

    X_train_sel_enc = X_train_enc[:, top_indices]
    X_test_sel_enc = X_test_enc[:, top_indices]

    # dense for VT
    X_train_sel_df = pd.DataFrame(X_train_sel_enc.toarray(), columns=top_feats)
    X_test_sel_df = pd.DataFrame(X_test_sel_enc.toarray(), columns=top_feats)

    vt = VarianceThreshold(threshold=1e-5)
    X_train_sel = vt.fit_transform(X_train_sel_df)
    X_test_sel = vt.transform(X_test_sel_df)

    # 9) Retrain on VT-top-features
    clf2, reg2, encoder_or_bins2 = train_models(X_train_sel, y_train, target_type)

    # 10) Test predictions + metrics + confusion matrix
    if target_type == "categorical":
        final_pred = clf2.predict(X_test_sel)

        try:
            acc = accuracy_score(y_test, final_pred)
            f1 = f1_score(y_test, final_pred, average="macro")
            with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
                json.dump({"accuracy": float(acc), "f1_macro": float(f1)}, f, indent=2)

            cm = confusion_matrix(y_test, final_pred, labels=clf2.classes_)
            fig_cm, ax = plt.subplots(figsize=(5.5, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf2.classes_)
            disp.plot(ax=ax, colorbar=False)
            plt.title("Confusion Matrix (test split)")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "confusion_matrix.png")
            plt.close(fig_cm)
            print(f"Saved metrics.json (acc={acc:.3f}, f1={f1:.3f}) and plots/confusion_matrix.png")
        except Exception as e:
            print(f"Warning: could not compute/save metrics/confusion: {e}")
    else:
        final_pred = reg2.predict(X_test_sel)

    # 11) Feature distributions + single-vs-all scatter (optional)
    df_plot = X_full_plot[top_feats].copy()
    df_plot[target_col] = y.values
    df_plot = df_plot.dropna()
    if target_type == "categorical":
        df_plot[target_col] = df_plot[target_col].astype(str)
        for feat in top_feats:
            if feat not in X.columns:
                df_plot[feat] = df_plot[feat].astype("category")
    plot_feature_distributions(df_plot, top_feats, target_col, target_type, output_dir=str(FEATURE_PLOTS_DIR))

    try:
        y_test_enc = LabelEncoder().fit_transform(y_test)
        evaluate_and_plot_feature_performance(X_test_sel, pd.Series(y_test_enc), top_feats, output_path=PLOTS_DIR / "feature_score_scatter.png")
    except Exception as e:
        print(f"Scatter plot skipped: {e}")

    # 12) Build "encode_like_train" using the SAME fitted transformers
    def encode_like_train(df_in: pd.DataFrame):
        parts = []
        # numeric (same imputer)
        if numeric_cols:
            if imputer_num is None:
                tmp_num = df_in[numeric_cols].copy()
                tmp_num = tmp_num.fillna(tmp_num.median(numeric_only=True))
            else:
                tmp_num = pd.DataFrame(
                    imputer_num.transform(df_in[numeric_cols]),
                    columns=numeric_cols,
                    index=df_in.index,
                )
            parts.append(csr_matrix(tmp_num.values))
        # ordinal
        if high_cat_cols:
            parts.append(csr_matrix(ord_enc.transform(df_in[high_cat_cols].astype(str))))
        # one-hot
        if small_cat_cols:
            parts.append(ohe.transform(df_in[small_cat_cols].astype(str)))
        if not parts:
            return csr_matrix((len(df_in), 0))
        return hstack(parts).tocsr()

    # Encode ALL, select same top features, apply same VT
    X_enc_all = encode_like_train(X)
    X_all_sel_enc = X_enc_all[:, top_indices]
    X_all_sel_df = pd.DataFrame(X_all_sel_enc.toarray(), columns=top_feats)
    X_all_sel = vt.transform(X_all_sel_df)

    # 13) Predict for ALL + save predictions_all.csv
    if target_type == "categorical":
        pred_all = clf2.predict(X_all_sel)
        try:
            conf_all = clf2.predict_proba(X_all_sel).max(axis=1)
        except Exception:
            conf_all = np.full(len(pred_all), np.nan)

        pred_all_df = pd.DataFrame({
            "wb_id": wb_ids_all,
            "Predicted Class": pd.Categorical(pred_all),
            "Prediction Confidence": conf_all
        })
        # Attach actuals if lengths align (they should)
        if len(pred_all_df) == len(y):
            pred_all_df["Actual Class"] = y.values

        pred_all_df.to_csv(ARTIFACTS_DIR / "predictions_all.csv", index=False)
        print("Saved artifacts/predictions_all.csv")

    else:
        pred_all_df = pd.DataFrame({
            "wb_id": wb_ids_all,
            "Predicted Value": reg2.predict(X_all_sel)
        })
        pred_all_df.to_csv(ARTIFACTS_DIR / "predictions_all.csv", index=False)
        print("Saved artifacts/predictions_all.csv (continuous)")

    # 14) Save the whole inference stack
    try:
        joblib.dump(clf2, ARTIFACTS_DIR / "model.joblib")
        if ohe is not None:
            joblib.dump(ohe, ARTIFACTS_DIR / "onehot.joblib")
        if ord_enc is not None:
            joblib.dump(ord_enc, ARTIFACTS_DIR / "ordinal.joblib")
        if imputer_num is not None:
            joblib.dump(imputer_num, ARTIFACTS_DIR / "imputer_num.joblib")
        joblib.dump(vt, ARTIFACTS_DIR / "vt.joblib")

        # feature lists
        with open(ARTIFACTS_DIR / "feature_columns_prevt.json", "w") as f:
            json.dump(list(feature_names_enc), f, indent=2)
        with open(ARTIFACTS_DIR / "kept_columns.json", "w") as f:
            json.dump(list(top_feats), f, indent=2)  # columns before VT (top features)
        with open(ARTIFACTS_DIR / "top_features.json", "w") as f:
            json.dump(list(top_feats), f, indent=2)
        if target_type == "categorical" and hasattr(clf2, "classes_"):
            with open(ARTIFACTS_DIR / "classes.json", "w") as f:
                json.dump([str(c) for c in clf2.classes_], f, indent=2)
        print("Saved model + preprocessing artifacts.")
    except Exception as e:
        print(f"Warning: could not save model artifacts: {e}")

    # return for PDP (uses train matrix)
    return final_pred, (clf2 if target_type == "categorical" else reg2), X_train_sel, top_feats


def explain_model(model, X, feature_names=None, target_name="target", top_n=15):
    """Generate PDPs + text trends; save figures under plots/."""
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

    explanations = []
    for idx, fname in zip(top_idx, top_features):
        try:
            pdp_res = partial_dependence(model, X, [idx], feature_names=feature_names, kind="both")
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
        out_file = PLOTS_DIR / f"pdp_{fname.replace(' ', '_')}.png"
        plt.savefig(out_file, bbox_inches="tight")
        plt.close()

    print("\n".join(explanations))

    # 2D PDPs
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
                PartialDependenceDisplay.from_estimator(
                    model, X_pdp, [(idx1, idx2)], feature_names=feature_names, grid_resolution=25
                )
                plt.tight_layout()
                out_file = PLOTS_DIR / f"pdp_2d_{f1}_{f2}.png"
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

    print("\nDone. Viewer-ready outputs:")
    print("  - artifacts/predictions_all.csv")
    print("  - artifacts/metrics.json")
    print("  - plots/confusion_matrix.png")
    print("  - plots/feature_importance.png, plots/correlation_matrix.png, plots/cv_accuracy_boxplot.png, plots/pdp_*.png")
    print("  - artifacts/model.joblib, onehot.joblib, ordinal.joblib, imputer_num.joblib, vt.joblib")
    print("  - artifacts/feature_columns_prevt.json, kept_columns.json, top_features.json, classes.json (if categorical)")