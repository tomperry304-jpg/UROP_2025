# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 18:28:04 2025

@author: tompe
"""

import argparse
import logging
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor as PoolExecutor

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

from process_input import process_and_aggregate

if sys.platform == "linux":
    matplotlib.use("Agg")  # Use non-GUI backend on Linux servers

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(process)d %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


MAX_CATEGORIES_FOR_ONEHOT = 20
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


def determine_target_type(series, cat_threshold=15):
    """
    Determine if the target series is categorical or continuous.
    """
    if pd.api.types.is_numeric_dtype(series):
        return "categorical" if series.nunique() <= cat_threshold else "continuous"
    else:
        return "categorical"


def train_models(X_train, y_train, target_type):
    """
    Train RandomForest with hyperparameter tuning for better performance.
    """
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
        grid.fit(X_train, y_encoded)

        clf = grid.best_estimator_
        LOGGER.info("Best RandomForest params: %s", grid.best_params_)

        reg = RandomForestRegressor(
            n_estimators=500,
            max_depth=grid.best_params_["max_depth"],
            min_samples_leaf=grid.best_params_["min_samples_leaf"],
            random_state=RANDOM_STATE,
            n_jobs=-1,  # Use all available cores
        )
        reg.fit(X_train, y_encoded)

        encoder_or_bins = le

    else:
        n_bins = min(10, y_train.nunique())
        if n_bins >= 2:
            y_array = y_train.values.ravel()
            y_binned, bins = pd.qcut(
                y_array, q=n_bins, labels=False, retbins=True, duplicates="drop"
            )
            clf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE)
            clf.fit(X_train, y_binned)
            encoder_or_bins = bins
        reg = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE)
        reg.fit(X_train, y_train.values.ravel())

    return clf, reg, encoder_or_bins


def get_top_features(importances, feature_names, top_n=15):
    """
    Return the top_n feature names based on importance scores.
    """
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    top_features = [f for f, imp in feat_imp[:top_n]]
    return top_features, feat_imp[:top_n]


def plot_feature_importance(importances, feature_names, output_path, target_name=None):
    """
    Plot and save a bar chart of feature importances (top 15).
    """

    feats, imps = zip(
        *sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    )
    top_feats = list(feats[:15])
    top_imps = list(imps[:15])
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=top_imps, y=top_feats, palette="coolwarm")
    title = "Top 15 Feature Importances"
    if target_name:
        title += f" for predicting {target_name}"
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    for p in ax.patches:
        width = p.get_width()
        ax.text(
            width + 0.001, p.get_y() + p.get_height() / 2, f"{width:.2f}", va="center"
        )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_feature_distributions(df, features, target_name, target_type, output_dir):
    """
    Plot and save distribution plots for features.
    Categorical features: countplot; Continuous features: boxplot.
    Grouped by target variable (or its binned version if continuous).
    """
    os.makedirs(output_dir, exist_ok=True)
    df_plot = df.copy()
    df_plot = df_plot.loc[:, ~df_plot.columns.duplicated()]
    if target_type == "continuous":
        n_unique = df_plot[target_name].nunique()
        n_bins = min(5, n_unique)
        try:
            df_plot["target_group"] = pd.qcut(
                df_plot[target_name], q=n_bins, labels=False, duplicates="drop"
            )
        except Exception:
            df_plot["target_group"] = pd.cut(
                df_plot[target_name], bins=n_bins, labels=False
            )
        hue_col = "target_group"
    else:
        hue_col = target_name

    for feat in features:
        plt.figure(figsize=(8, 6))
        if pd.api.types.is_numeric_dtype(df_plot[feat]):
            sns.boxplot(x=hue_col, y=feat, data=df_plot)
            title = f"Boxplot of {feat} by {'Target' if target_type == 'continuous' else target_name}"
            plt.title(title)
            plt.xlabel("Target Bin" if target_type == "continuous" else target_name)
            plt.ylabel(feat)
        else:
            sns.countplot(x=feat, hue=hue_col, data=df_plot)
            title = f"Countplot of {feat} by {'Target' if target_type == 'continuous' else target_name}"
            plt.title(title)
            plt.xlabel(feat)
            plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        safe_feat = feat.replace(" ", "_")
        plot_type = (
            "boxplot" if pd.api.types.is_numeric_dtype(df_plot[feat]) else "countplot"
        )
        out_file = os.path.join(output_dir, f"{safe_feat}_{plot_type}.png")
        plt.savefig(out_file)


def save_predictions_csv(filename, wb_ids, y_true, clf_pred, reg_pred, combined_pred):
    """Write predictions with a stable join key for the map."""

    # Coerce to 1-D arrays and drop any pre-existing indices
    wb_ids_arr = (
        pd.Series(wb_ids, dtype="string")
        .reset_index(drop=True)
        .astype(str)
        .str.strip()
        .str.upper()
    )
    y_true_arr = pd.Series(y_true).reset_index(drop=True).astype(object).values.ravel()
    clf_arr = pd.Series(clf_pred).reset_index(drop=True).values.ravel()
    reg_arr = pd.Series(reg_pred).reset_index(drop=True).values.ravel()
    comb_arr = pd.Series(combined_pred).reset_index(drop=True).values.ravel()

    # Ensure equal lengths (fail early with a clear message)
    lens = [len(wb_ids_arr), len(y_true_arr), len(clf_arr), len(reg_arr), len(comb_arr)]
    if len(set(lens)) != 1:
        raise ValueError(
            f"save_predictions_csv: length mismatch {lens} "
            "(check y_test vs X_test-derived predictions)"
        )

    n = lens[0]
    df_out = pd.DataFrame(
        {
            "wb_id": wb_ids_arr,
            "True_Target": y_true_arr,
            "Pred_Classifier": clf_arr,
            "Pred_Regressor": reg_arr,
            "Pred_Combined": comb_arr,
        },
        index=pd.RangeIndex(n),
    )

    df_out.to_csv(filename, index=False)


def evaluate_and_plot_feature_performance(
    X, y, feature_names, output_path="plots/feature_score_scatter.png"
):
    os.makedirs("plots", exist_ok=True)

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(
            X,
            columns=feature_names
            if feature_names is not None
            else [f"feature_{i}" for i in range(X.shape[1])],
        )

    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def normalize_labels(series):
        return series.astype(str).str.strip().str.lower()

    y_norm = normalize_labels(y)

    for feat in feature_names:
        LOGGER.info("Evaluating feature: %s", feat)
        acc_scores, f1_scores = [], []
        for train_idx, test_idx in skf.split(X[[feat]], y_norm):
            X_train, X_test = X.iloc[train_idx][[feat]], X.iloc[test_idx][[feat]]
            y_train, y_test = y_norm.iloc[train_idx], y_norm.iloc[test_idx]

            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc_scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average="macro"))

        results.append(
            {
                "Feature": feat,
                "Accuracy": np.mean(acc_scores),
                "F1 Score": np.mean(f1_scores),
            }
        )

    acc_scores, f1_scores = [], []
    for train_idx, test_idx in skf.split(X[feature_names], y_norm):
        X_train, X_test = (
            X.iloc[train_idx][feature_names],
            X.iloc[test_idx][feature_names],
        )
        y_train, y_test = y_norm.iloc[train_idx], y_norm.iloc[test_idx]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average="macro"))

    results.append(
        {
            "Feature": "All Features",
            "Accuracy": np.mean(acc_scores),
            "F1 Score": np.mean(f1_scores),
        }
    )

    baseline_label = "2"
    y_baseline = pd.Series([baseline_label] * len(y_norm))

    LOGGER.info("Unique normalized labels: %s", y_norm.unique())
    LOGGER.info("Baseline first 5 predictions: %s", y_baseline.head())

    baseline_acc = accuracy_score(y_norm, y_baseline)
    baseline_f1 = f1_score(y_norm, y_baseline, average="macro")

    LOGGER.info("Unique labels in dataset: %s", y_norm.unique())

    results.append(
        {
            "Feature": "Predict Moderate",
            "Accuracy": baseline_acc,
            "F1 Score": baseline_f1,
        }
    )

    df_results = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))

    plt.scatter(
        df_results["Accuracy"],
        df_results["F1 Score"],
        c="skyblue",
        edgecolors="black",
        label="Features",
    )

    baseline_row = df_results[df_results["Feature"] == "Predict Moderate"].iloc[0]
    plt.scatter(
        baseline_row["Accuracy"],
        baseline_row["F1 Score"],
        color="red",
        s=80,
        label="Predict Moderate",
    )

    for i, row in df_results.iterrows():
        plt.annotate(
            row["Feature"],
            (row["Accuracy"] + 0.001, row["F1 Score"] + 0.001),
            fontsize=8,
        )

    plt.xlabel("Accuracy")
    plt.ylabel("F1 Score")
    plt.title("Accuracy vs F1 Score (Including Predict-Moderate Baseline)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    LOGGER.info(f"Scatter plot saved as '{output_path}'")


def random_forest_processing(df_classified_chem_data, target_column_name):
    if isinstance(target_column_name, list):
        target_column_name = target_column_name[0]
    if target_column_name not in df_classified_chem_data.columns:
        raise ValueError(f"Target '{target_column_name}' not in merged data.")

    # filter out rows with missing target
    df_model = df_classified_chem_data.dropna(subset=[target_column_name]).reset_index(
        drop=True
    )
    df_model = df_model.loc[:, ~df_model.columns.duplicated()]
    wb_ids_all = df_model["wb_id"].astype(str).str.strip().str.upper().copy()

    if "wb_id" in df_model.columns:
        df_model = df_model.drop(columns=["wb_id"])
    X = df_model.drop(columns=[target_column_name]).copy()
    y = df_model[target_column_name].copy()

    model_columns = set(df_model.columns.tolist())
    df_model = None

    # Fitler out 'Not Assessed' entries;
    mask = y.astype(str).str.lower().str.strip() != "not assessed"
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    wb_ids = wb_ids_all[mask].reset_index(drop=True)

    LOGGER.info("X shape: %s", X.shape)
    LOGGER.info("y shape: %s", y.shape)
    LOGGER.info("Index match: %s", X.index.equals(y.index))
    LOGGER.info(f"Target '{target_column_name}' type: %s", type(y.iloc[0]))

    target_type = determine_target_type(y)
    LOGGER.info(f"Target '{target_column_name}' is detected as {target_type}.")

    X_train, X_test, y_train, y_test, wb_train, wb_test = train_test_split(
        X, y, wb_ids, test_size=0.2, random_state=RANDOM_STATE
    )

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    small_cat_cols = []
    high_cat_cols = []
    if cat_cols:
        for col in cat_cols:
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)

        small_cat_cols = [
            col
            for col in cat_cols
            if X_train[col].nunique() <= MAX_CATEGORIES_FOR_ONEHOT
        ]
        high_cat_cols = [
            col
            for col in cat_cols
            if X_train[col].nunique() > MAX_CATEGORIES_FOR_ONEHOT
        ]

        if small_cat_cols:
            ohe = OneHotEncoder(sparse=True, handle_unknown="ignore")
            X_train_onehot = ohe.fit_transform(X_train[small_cat_cols])
            X_test_onehot = ohe.transform(X_test[small_cat_cols])
            onehot_feature_names = ohe.get_feature_names_out(small_cat_cols)
        else:
            X_train_onehot = None
            X_test_onehot = None
            onehot_feature_names = []

        if high_cat_cols:
            ord_enc = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            X_train_ord = ord_enc.fit_transform(X_train[high_cat_cols])
            X_test_ord = ord_enc.transform(X_test[high_cat_cols])
        else:
            ord_enc = None
            X_train_ord = None
            X_test_ord = None

        parts_train = []
        parts_test = []
        numeric_cols = [col for col in X_train.columns if col not in cat_cols]
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
    else:
        X_train_enc = X_train.values
        X_test_enc = X_test.values
        feature_names_enc = list(X_train.columns)
        ord_enc = None

    X_train = X_train_enc
    X_test = X_test_enc

    LOGGER.info("Training Main Random Forest model...")
    clf, reg, encoder_or_bins = train_models(X_train, y_train, target_type)
    LOGGER.info("Training Main Random Forest model complete.")

    if target_type == "categorical" and clf is not None:
        importances = clf.feature_importances_
        feature_names = feature_names_enc
    elif reg is not None:
        importances = reg.feature_importances_
        feature_names = feature_names_enc
    else:
        raise ValueError("No valid model to determine feature importances.")

    top_feats, top_importances = get_top_features(importances, feature_names, top_n=15)
    LOGGER.info("Top 15 features:")

    if target_type == "categorical":
        le_temp = LabelEncoder()
        y_test_enc = le_temp.fit_transform(y_test.values.ravel())
    else:
        y_test_enc = y_test.values.ravel()
    for feat, imp in top_importances:
        LOGGER.info(f'  feature "{feat}": {imp:.4f}')

    if high_cat_cols:
        for col in high_cat_cols:
            X[col] = X[col].astype(str)
        try:
            X[high_cat_cols] = ord_enc.transform(X[high_cat_cols])
        except Exception as e:
            LOGGER.info(
                f"Warning: could not ordinal-encode all high-cardinality features: {e}"
            )

    for feat in top_feats:
        if feat not in X.columns:
            for col in cat_cols:
                if feat.startswith(str(col) + "_"):
                    cat_val = feat[len(col) + 1 :]

                    X[feat] = (X[col].astype(str) == cat_val).astype(int)
                    break

    LOGGER.info("Plotting Correlation Matrix...")
    os.makedirs("plots", exist_ok=True)
    corr_df = X[top_feats].dropna()
    corr_matrix = corr_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Pearson Correlation Matrix of Top 15 Features")
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix.png")
    plt.show()

    LOGGER.info("Plotting Cross Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    clf_cv = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    scores = cross_val_score(
        clf_cv, X[top_feats], y.values.ravel(), cv=skf, scoring="accuracy"
    )

    plt.figure(figsize=(6, 6))
    sns.boxplot(y=scores, color="lightgreen")
    plt.title("5-fold Cross-validation Accuracy")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("plots/cv_accuracy_boxplot.png")
    plt.show()

    LOGGER.info("Plotting Features...")
    plot_feature_importance(
        importances, feature_names, "plots/feature_importance.png", target_column_name
    )
    LOGGER.info("Feature importance plot saved as 'plots/feature_importance.png'.")

    feat_index_map = {name: idx for idx, name in enumerate(feature_names_enc)}
    top_indices = [feat_index_map[f] for f in top_feats if f in feat_index_map]
    if isinstance(X_train, np.ndarray) or hasattr(X_train, "toarray"):
        X_train_sel_enc = X_train[:, top_indices]
        X_test_sel_enc = X_test[:, top_indices]
        X_train_sel = pd.DataFrame(
            X_train_sel_enc.toarray()
            if hasattr(X_train_sel_enc, "toarray")
            else X_train_sel_enc,
            columns=top_feats,
        )
        X_test_sel = pd.DataFrame(
            X_test_sel_enc.toarray()
            if hasattr(X_test_sel_enc, "toarray")
            else X_test_sel_enc,
            columns=top_feats,
        )
    else:
        X_train_sel = X_train[top_feats].copy()
        X_test_sel = X_test[top_feats].copy()

    from sklearn.feature_selection import VarianceThreshold

    vt = VarianceThreshold(threshold=1e-5)
    X_train_sel = vt.fit_transform(X_train_sel)
    X_test_sel = vt.transform(X_test_sel)

    clf2, reg2, encoder_or_bins2 = train_models(X_train_sel, y_train, target_type)

    from lightgbm import LGBMClassifier
    from sklearn.metrics import accuracy_score, f1_score

    lgbm_pred = None

    if target_type == "categorical":
        lgbm = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        from lightgbm import early_stopping, log_evaluation

        lgbm.fit(
            X_train_sel,
            y_train,
            eval_set=[(X_test_sel, y_test)],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(50), log_evaluation(20)],
        )
        lgbm_pred = lgbm.predict(X_test_sel)

        acc = accuracy_score(y_test, lgbm_pred)
        f1 = f1_score(y_test, lgbm_pred, average="macro")

        LOGGER.info(f"LightGBM Test Accuracy: {acc:.4f}")
        LOGGER.info(f"LightGBM Macro F1: {f1:.4f}")

    if target_type == "categorical":
        clf_pred_enc = clf2.predict(X_test_sel)
        reg_pred = reg2.predict(X_test_sel)

        ensemble_pred = None
        if target_type == "categorical" and lgbm_pred is not None:
            rf_pred = clf2.predict(X_test_sel)
            ensemble_pred = np.where(rf_pred == lgbm_pred, rf_pred, lgbm_pred)

            acc_ens = accuracy_score(y_test, ensemble_pred)
            f1_ens = f1_score(y_test, ensemble_pred, average="macro")

            LOGGER.info(f"Ensemble Accuracy: {acc_ens:.4f}")
            LOGGER.info(f"Ensemble Macro F1: {f1_ens:.4f}")

        avg_pred = 0.5 * (clf_pred_enc + reg_pred)
        combined_num = np.rint(avg_pred).astype(int)

        le = encoder_or_bins
        combined_num = np.clip(combined_num, 0, len(le.classes_) - 1)
        final_pred = le.inverse_transform(combined_num)
        clf_pred = le.inverse_transform(clf_pred_enc)

    else:
        reg_pred = reg2.predict(X_test_sel)
        if clf2 is not None:
            clf_pred_bins = clf2.predict(X_test_sel)
            df_temp = pd.DataFrame({"target": y_train.values.ravel()})
            n_bins = len(np.unique(clf_pred_bins))
            df_temp["bin"], bins = pd.qcut(
                df_temp["target"],
                q=n_bins,
                labels=False,
                retbins=True,
                duplicates="drop",
            )
            bin_medians = df_temp.groupby("bin")["target"].median().to_dict()
            class_pred_numeric = np.array(
                [bin_medians.get(b, np.nan) for b in clf_pred_bins]
            )
            final_pred = 0.5 * (class_pred_numeric + reg_pred)
            clf_pred = clf_pred_bins

            global ECO_CLASS
            ECO_CLASS = final_pred
        else:
            final_pred = reg_pred
            clf_pred = [None] * len(reg_pred)

    save_predictions_csv(
        "model_predictions.csv",
        wb_test,  # <- stable join key
        y_test,
        clf_pred,
        reg_pred,
        ensemble_pred if ensemble_pred is not None else final_pred,
    )

    df_plot = X[top_feats].copy()
    df_plot[target_column_name] = y.values
    df_plot = df_plot.dropna()
    if target_type == "categorical":
        df_plot[target_column_name] = df_plot[target_column_name].astype(str)

        for feat in top_feats:
            if feat not in model_columns:
                df_plot[feat] = df_plot[feat].astype("category")
    plot_feature_distributions(
        df_plot, top_feats, target_column_name, target_type, output_dir="feature_plots"
    )
    LOGGER.info("Feature distribution plots saved to 'feature_plots/' directory.")

    evaluate_and_plot_feature_performance(X_test_sel, pd.Series(y_test_enc), top_feats)
    model_used = clf2 if target_type == "categorical" else reg2
    return final_pred, model_used, X_train_sel, top_feats


def explain_model(model, X, feature_names, target_name):
    # --- make X a DataFrame so .sample works and names align ---
    if isinstance(X, np.ndarray):
        if feature_names is None or len(feature_names) != X.shape[1]:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()
        if feature_names is None:
            feature_names = list(X_df.columns)

    partial_dep_args = []
    for f1, f2 in PDP_2D_PAIRS:
        if f1 in feature_names and f2 in feature_names:
            args = [model, feature_names, X_df, f1, f2]
            partial_dep_args.append(args)

    nprocs = min(len(partial_dep_args), os.cpu_count() or 1)

    LOGGER.info(f"Calculating 2D PDPs using {nprocs} parallel processes...")
    with PoolExecutor(max_workers=nprocs) as pool:
        partial_dependencies = pool.map(calc_partial_dependence, partial_dep_args)
    LOGGER.info("Calculation of 2D PDPs complete.")

    for args, pdp_res_2d in zip(partial_dep_args, partial_dependencies):
        feat_name_1, feat_name_2 = args[-2], args[-1]
        generate_pdp_2d_plot(
            pdp_res_2d,
            feat_name_1,
            feat_name_2,
        )


def calc_partial_dependence(args):
    model, feature_names, X_df, feat_name_1, feat_name_2 = args
    idx1, idx2 = feature_names.index(feat_name_1), feature_names.index(feat_name_2)
    X_pdp = X_df
    if len(X_pdp) > 2000:
        X_pdp = X_pdp.sample(n=2000, random_state=RANDOM_STATE)
    return partial_dependence(
        model,
        X_pdp.values,
        [(idx1, idx2)],
        feature_names=feature_names,
        grid_resolution=25,
    )


def generate_pdp_2d_plot(pdp_res_2d, feat_name_1, feat_name_2):
    grid1 = pdp_res_2d["grid_values"][0]
    grid2 = pdp_res_2d["grid_values"][1]
    avg_preds_2d = pdp_res_2d["average"][0]
    Xx, Yy = np.meshgrid(grid2, grid1)
    plt.figure(figsize=(6, 5))
    plt.contourf(Xx, Yy, avg_preds_2d, cmap="viridis")
    plt.colorbar(label="Predicted")
    plt.xlabel(feat_name_2)
    plt.ylabel(feat_name_1)
    plt.title(f"2D PDP of {feat_name_1} and {feat_name_2}")
    out_file = f"plots/pdp_2d_{feat_name_1}_{feat_name_2}_fallback.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"2D PDP saved as {out_file}")


def save_1d_pdp_plots(
    model,
    X,
    feature_names=None,
    out_dir="plots",
    top_n=15,
    grid_resolution=50,
    row_subsample=None,
    preferred_classes=(),
):
    """
    Save 1D PDP images for the top_n features into `out_dir` as:
      - Regression:     pdp_1d_<feature>.png
      - Classification: pdp_1d_<feature>__class_<label>.png

    Parameters
    ----------
    model : fitted estimator
        Regressor or classifier compatible with scikit-learn PDP.
    X : array-like, shape (n_samples, n_features)
        Data with the exact columns/order used to train `model`.
    feature_names : list[str] | None
        Column names. If None or length mismatch, falls back to generic names.
    out_dir : str
        Directory to write PNG files into.
    top_n : int
        Number of features to plot (by feature_importances_ if available).
    grid_resolution : int
        Grid size for PDP computation.
    row_subsample : int | None
        If set and < n_rows, randomly subsample rows for speed.
    preferred_classes : iterable
        For classifiers, if non-empty, only these class labels (names or indices)
        are plotted; otherwise all classes are plotted.
    """

    X = np.asarray(X)
    n_rows, n_cols = X.shape

    if feature_names is None or len(feature_names) != n_cols:
        feature_names = [f"feature_{i}" for i in range(n_cols)]
    else:
        feature_names = list(feature_names)

    X_df = pd.DataFrame(X, columns=feature_names)

    if row_subsample is not None and row_subsample < n_rows:
        rs = np.random.RandomState(42)
        idx = rs.choice(n_rows, size=row_subsample, replace=False)
        X_df = X_df.iloc[idx].reset_index(drop=True)

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        if importances.shape[0] == n_cols:
            order = np.argsort(importances)[::-1]
        else:
            order = np.arange(n_cols)
    else:
        order = np.arange(n_cols)

    top_n = int(min(top_n, n_cols))
    feat_indices = order[:top_n].tolist()

    os.makedirs(out_dir, exist_ok=True)

    is_classifier = hasattr(model, "predict_proba") or hasattr(model, "classes_")

    class_targets = None
    class_labels = None
    if is_classifier and hasattr(model, "classes_"):
        class_labels = list(model.classes_)
        if preferred_classes:
            mapped = []
            for c in preferred_classes:
                if isinstance(c, (int, np.integer)) and 0 <= c < len(class_labels):
                    mapped.append(c)
                else:
                    try:
                        mapped.append(class_labels.index(c))
                    except ValueError:
                        pass
            class_targets = sorted(set(mapped))
        else:
            class_targets = list(range(len(class_labels)))

    def _fresh_ax():
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        return fig, ax

    saved_files = []
    for idx in feat_indices:
        fname = feature_names[idx]

        if is_classifier and class_targets is not None and len(class_targets) > 0:
            for targ in class_targets:
                fig, ax = _fresh_ax()
                try:
                    PartialDependenceDisplay.from_estimator(
                        model,
                        X_df,
                        features=[idx],
                        grid_resolution=grid_resolution,
                        kind="average",
                        target=targ,
                        ax=ax,
                    )
                except TypeError:
                    try:
                        PartialDependenceDisplay.from_estimator(
                            model,
                            X_df,
                            features=[idx],
                            grid_resolution=grid_resolution,
                            ax=ax,
                        )
                    except Exception:
                        plt.close(fig)
                        continue

                ax.set_title(
                    f"PDP: {fname} (class {class_labels[targ] if class_labels else targ})"
                )
                ax.set_xlabel(fname)
                out_path = os.path.join(
                    out_dir,
                    f"pdp_1d_{fname}__class_{class_labels[targ] if class_labels else targ}.png",
                )
                fig.savefig(out_path, dpi=120)
                plt.close(fig)
                saved_files.append(out_path)
        else:
            fig, ax = _fresh_ax()
            try:
                PartialDependenceDisplay.from_estimator(
                    model,
                    X_df,
                    features=[idx],
                    grid_resolution=grid_resolution,
                    kind="average",
                    ax=ax,
                )
            except TypeError:
                try:
                    PartialDependenceDisplay.from_estimator(
                        model,
                        X_df,
                        features=[idx],
                        grid_resolution=grid_resolution,
                        ax=ax,
                    )
                except Exception:
                    plt.close(fig)
                    continue

            ax.set_title(f"PDP: {fname}")
            ax.set_xlabel(fname)
            out_path = os.path.join(out_dir, f"pdp_1d_{fname}.png")
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            saved_files.append(out_path)

    return saved_files


def main():
    parser = argparse.ArgumentParser(description="Data processing and modeling")
    parser.add_argument(
        "--excel_path",
        type=str,
        default="WFD_SW_Classification_Status_and_Objectives_Cycle2_v4.xlsx",
    )
    parser.add_argument("--parquet_path", type=str, default="wims_wfd_merged.parquet")
    parser.add_argument("--target", type=str, default="Ecological Class")
    args = parser.parse_args()

    if False:
        df_classified_chem_data = process_and_aggregate(args)
        df_classified_chem_data.to_parquet("process_input.parquet", index=False)
    else:
        df_classified_chem_data = pd.read_parquet("process_input.parquet")

    target_column = args.target

    final_pred, model, X_train_used, feature_names_used = random_forest_processing(
        df_classified_chem_data, target_column
    )

    os.makedirs("model_predictions", exist_ok=True)
    out_path = os.path.join("model_predictions", "final_predictions.csv")

    explain_model(
        model,
        X_train_used,
        feature_names=feature_names_used,
        target_name=args.target,
    )

    save_1d_pdp_plots(
        model,
        X_train_used,
        feature_names_used,
        out_dir="plots",
        top_n=15,
        row_subsample=5000,
    )


if __name__ == "__main__":
    main()

# accuracy and f1 for just predicting "moderate"
# try quantiles
# softwhere testing
# alterative AI prosseing
# more interactive data visulisation "streamlit"
# orange "load in data and describe it"
# data visuliation, robustness, more usfull
# interal anual varaiblity
# intigrate it into an "app" - streamlit
