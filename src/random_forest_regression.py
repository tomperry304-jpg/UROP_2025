# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 18:28:04 2025

@author: tompe
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import csr_matrix, hstack

# Set random seeds for reproducibility
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Configurable parameters for encoding and PDP
MAX_CATEGORIES_FOR_ONEHOT = 20  # One-hot encode features with <=20 unique values (else use ordinal encoding)
PDP_2D_PAIRS = [
    ("Ammonia(N)", "Orthophospht"),
    ("pH", "Temp Water"),
    ("Nitrate-N", "N Oxidised"),
    # ("Feature1", "Feature2"),  # Example pairs for 2D PDP plots
    # Temperature & Chemistry
    ("Temp Water", "Ammonia(N)"),
    ("Temp Water", "Nitrate-N"),

    # Nutrient Interactions
    ("Orthophospht", "Nitrate-N"),
    ("Ammonia(N)", "N Oxidised"),

    # Oxygen + Nutrients
    ("Dissolved Oxygen", "Ammonia(N)"),
    ("Dissolved Oxygen", "Orthophospht"),

    # Chemistry + pH
    ("pH", "Ammonia(N)"),
    ("pH", "N Oxidised"),

    # Example for solids if available (you can remove if not in your dataset)
    ("Suspended Solids", "Orthophospht"),
    ("Conductivity", "Nitrate-N"),
    ]

def load_excel_data(excel_path):
    """
    Load all sheets from the Excel file, concatenate them, and drop columns with >=50% missing values.
    Skips the second header row which repeats labels.
    """
    all_sheets = []
    try:
        xls = pd.ExcelFile(excel_path, engine='openpyxl')
        sheets = xls.sheet_names
    except Exception as e:
        raise Exception(f"Error reading Excel file: {e}")

    for sheet in sheets:
        # Read sheet, skip the repeated header row at index 1
        df_sheet = pd.read_excel(excel_path, sheet_name=sheet, skiprows=1, engine='openpyxl')
        all_sheets.append(df_sheet)
    if not all_sheets:
        raise ValueError("No sheets found in Excel file.")
    df_excel = pd.concat(all_sheets, ignore_index=True)
    # Drop columns with 50% or more missing values
    thresh = len(df_excel) * 0.5
    df_excel = df_excel.loc[:, df_excel.isnull().sum() < thresh]
    return df_excel

def load_parquet_data(parquet_path):
    """
    Load Parquet file and drop columns with <1,000,000 non-null values.
    """
    try:
        df_parquet = pd.read_parquet(parquet_path)
    except ImportError:
        raise ImportError("PyArrow or fastparquet is required to read Parquet files. Install one to proceed.")
    except Exception as e:
        raise Exception(f"Error reading Parquet file: {e}")

    # Keep columns with at least 1,000,000 non-null entries
    non_null_counts = df_parquet.notnull().sum()
    cols_to_keep = non_null_counts[non_null_counts >= 1000000].index
    df_parquet = df_parquet.loc[:, cols_to_keep]
    return df_parquet

def aggregate_parquet(df_parquet):
    """
    Aggregate WIMS data by wb_id, taking mean of numeric fields.
    """
    numeric_cols = df_parquet.select_dtypes(include=np.number).columns.tolist()
    
    agg_df = df_parquet.groupby("wb_id")[numeric_cols].mean().reset_index()
    
    #agg_df = df_parquet.groupby("wb_id")[numeric_cols].median().reset_index()
    
    #agg_df = df_parquet.groupby("wb_id")[numeric_cols].apply(lambda x: x.mean(skipna=True) if len(x) > 10 else x.median()).reset_index()
    
    """
    from scipy.stats import trim_mean
    agg_df = df_parquet.groupby("wb_id")[numeric_cols].apply(
    lambda x: x.dropna().apply(lambda col: trim_mean(col, 0.1))
    ).reset_index()
    """
    return agg_df

def merge_datasets(df_excel, df_parquet):
    """
    Merge the two datasets using 'Water Body ID' from Excel and 'wb_id' from Parquet.
    """
    if 'Water Body ID' not in df_excel.columns:
        raise KeyError("Excel file missing 'Water Body ID'")
    if 'wb_id' not in df_parquet.columns:
        raise KeyError("Parquet file missing 'wb_id'")

    # Rename for consistency
    df_excel = df_excel.rename(columns={"Water Body ID": "wb_id"})
    df_parquet_agg = aggregate_parquet(df_parquet)
    # Perform inner join
    df_merged = pd.merge(df_excel, df_parquet_agg, on="wb_id", how="inner")
    return df_merged

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
    Train RandomForestClassifier and RandomForestRegressor on the data.
    Returns trained classifier, regressor, and encoder/bins info.
    """
    clf = None
    reg = None
    encoder_or_bins = None

    if target_type == "categorical":
        # Label-encode y for training
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_train.values.ravel())
        # Train classifier
        clf = RandomForestClassifier(random_state=RANDOM_STATE)
        clf.fit(X_train, y_encoded)
        # Train regressor on the encoded labels
        reg = RandomForestRegressor(random_state=RANDOM_STATE)
        reg.fit(X_train, y_encoded)
        encoder_or_bins = le
    else:
        # Continuous target
        n_bins = min(10, y_train.nunique())
        if n_bins >= 2:
            y_array = y_train.values.ravel()
            y_binned, bins = pd.qcut(y_array, q=n_bins, labels=False, retbins=True, duplicates='drop')
            clf = RandomForestClassifier(random_state=RANDOM_STATE)
            clf.fit(X_train, y_binned)
            encoder_or_bins = bins
        # Train regressor on actual continuous target
        reg = RandomForestRegressor(random_state=RANDOM_STATE)
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
    # Annotate bars with importance values
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.001, p.get_y() + p.get_height()/2, f"{width:.2f}", va='center')
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
        # Bin the continuous target into 5 categories for grouping
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
            # Continuous feature: use boxplot
            sns.boxplot(x=hue_col, y=feat, data=df_plot)
            title = f"Boxplot of {feat} by {'Target' if target_type=='continuous' else target_name}"
            plt.title(title)
            plt.xlabel('Target Bin' if target_type=='continuous' else target_name)
            plt.ylabel(feat)
        else:
            # Categorical feature: use countplot
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

def save_predictions_csv(filename, X, y_true, clf_pred, reg_pred, combined_pred):
    """
    Save features, true target, and predictions to CSV.
    """
    df_out = X.copy().reset_index(drop=True)
    df_out['True_Target'] = y_true.values.ravel()
    df_out['Pred_Classifier'] = clf_pred
    df_out['Pred_Regressor'] = reg_pred
    df_out['Pred_Combined'] = combined_pred
    df_out.to_csv(filename, index=False)

def evaluate_and_plot_feature_performance(X, y, feature_names, output_path="plots/feature_score_scatter.png"):
    os.makedirs("plots", exist_ok=True)
    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def normalize_labels(series):
        return series.astype(str).str.strip().str.lower()

    y_norm = normalize_labels(y)

    # --- Evaluate each single feature ---
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

        results.append({
            "Feature": feat,
            "Accuracy": np.mean(acc_scores),
            "F1 Score": np.mean(f1_scores)
        })

    # --- Evaluate all features together ---
    acc_scores, f1_scores = [], []
    for train_idx, test_idx in skf.split(X[feature_names], y_norm):
        X_train, X_test = X.iloc[train_idx][feature_names], X.iloc[test_idx][feature_names]
        y_train, y_test = y_norm.iloc[train_idx], y_norm.iloc[test_idx]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='macro'))

    results.append({
        "Feature": "All Features",
        "Accuracy": np.mean(acc_scores),
        "F1 Score": np.mean(f1_scores)
    })

    # --- Baseline: Always predict "moderate" ---
    baseline_label = "2"  # because Moderate is encoded as 2
    y_baseline = pd.Series([baseline_label] * len(y_norm))
    
    print("Unique normalized labels:", y_norm.unique())
    print("Baseline first 5 predictions:", y_baseline.head())
    
    baseline_acc = accuracy_score(y_norm, y_baseline)
    baseline_f1 = f1_score(y_norm, y_baseline, average='macro')

    print("Unique labels in dataset:", y_norm.unique())  # ✅ debug check

    results.append({
        "Feature": "Predict Moderate",
        "Accuracy": baseline_acc,
        "F1 Score": baseline_f1
    })

    # --- Plot ---
    df_results = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))

    plt.scatter(df_results["Accuracy"], df_results["F1 Score"],
                c='skyblue', edgecolors='black', label='Features')

    baseline_row = df_results[df_results["Feature"] == "Predict Moderate"].iloc[0]
    plt.scatter(baseline_row["Accuracy"], baseline_row["F1 Score"],
                color='red', s=80, label='Predict Moderate')

    for i, row in df_results.iterrows():
        plt.annotate(row["Feature"],
                     (row["Accuracy"] + 0.001, row["F1 Score"] + 0.001),
                     fontsize=8)

    plt.xlabel("Accuracy")
    plt.ylabel("F1 Score")
    plt.title("Accuracy vs F1 Score (Including Predict-Moderate Baseline)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Scatter plot saved as '{output_path}'")

def main(args):
    # Load data
    df_excel = load_excel_data(args.excel_path)
    # Drop "Overall Water Body Class" if it exists
    df_excel = df_excel.drop(columns=["Overall Water Body Class"], errors="ignore")
    print(f"Excel data: {df_excel.shape} (rows, columns)")
    df_parquet = load_parquet_data(args.parquet_path)
    print(f"Parquet data: {df_parquet.shape} (rows, columns)")

    # (Optional debugging prints of columns)
    # print("Excel columns:", df_excel.columns.tolist())
    # print("Parquet columns:", df_parquet.columns.tolist())

    df_merged = merge_datasets(df_excel, df_parquet)
    print(f"Merged data before filtering: {df_merged.shape} (rows, columns)")

    # Keep only the target column from Excel and chemical features from Parquet
    target_col = args.target
    df_excel = df_excel.rename(columns={'Water Body ID': 'wb_id'})
    target_df = df_excel[['wb_id', target_col]]
    # Example filter: keep only chemical-related columns from Parquet
    """
    chemical_keywords = ['NO3', 'NH4', 'PO4', 'Nitrate', 'Phosphate', 'Ammonia', 'Nitrogen', 'Phosphorus', 'mg', 'ug', 'conc', 'chem']
    chemical_cols = [col for col in df_parquet.columns if any(kw.lower() in col.lower() for kw in chemical_keywords)]
    wims_df = df_parquet[['wb_id'] + chemical_cols].copy()
    # Merge target and features
    df_merged = pd.merge(target_df, wims_df, on='wb_id', how='inner')
    """
    var_counts = df_parquet['variable'].value_counts()
    frequent_vars = var_counts[var_counts > 1e6].index.tolist()

    print(f"Found {len(frequent_vars)} frequent variables with > 1M records")
    print("Sample:", frequent_vars[:5])
    
    # STEP 2: Filter Parquet to only those variables
    df_filtered = df_parquet[df_parquet['variable'].isin(frequent_vars)]

    # STEP 3: Pivot data to wide format (each variable becomes a column)
    pivot_df = df_filtered.pivot_table(
        index='wb_id',
        columns='variable',
        values='result',
        aggfunc='mean'
        ).reset_index()

    # STEP 4: Merge with Excel target
    df_excel = df_excel.rename(columns={'Water Body ID': 'wb_id'})
    target_df = df_excel[['wb_id', target_col]]
    df_merged = pd.merge(target_df, pivot_df, on='wb_id', how='inner')

    # STEP 5: Drop NaNs
    df_merged = df_merged.dropna()
    
    """
    # Filter out rows with any missing values
    chemical_subset = df_merged[['wb_id', args.target] + chemical_cols].dropna(subset=[args.target])
    chemical_subset = chemical_subset.dropna()
    print(f"Merged data after filtering chemical variables: {chemical_subset.shape} (rows, columns)")
    print("Chemical columns found:", len(chemical_cols))
    print("Sample chemical columns:", chemical_cols[:5])
    df_merged = chemical_subset
    """
    target = args.target
    if isinstance(target, list):
        target = target[0]
    if target not in df_merged.columns:
        raise ValueError(f"Target '{target}' not in merged data.")

    # Drop rows with missing target (already handled above, but ensure none remain)
    df_model = df_merged.dropna(subset=[target]).reset_index(drop=True)
    df_model = df_model.loc[:, ~df_model.columns.duplicated()]
    # Drop unique ID column to reduce dimensionality and avoid overfitting
    if 'wb_id' in df_model.columns:
        df_model = df_model.drop(columns=['wb_id'])
    X = df_model.drop(columns=[target]).copy()
    y = df_model[target].copy()

    # Debugging shape and type information
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Index match:", X.index.equals(y.index))
    print(f"Target '{target}' type:", type(y.iloc[0]))

    # Determine target type (categorical vs continuous)
    target_type = determine_target_type(y)
    print(f"Target '{target}' is detected as {target_type}.")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Memory Efficiency: Encode categorical features without creating a huge dense matrix
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    small_cat_cols = []
    high_cat_cols = []
    if cat_cols:
        # Convert categorical columns to string type for uniform encoding
        for col in cat_cols:
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)
        # Decide which categorical features to one-hot encode (low cardinality) and which to ordinal encode (high cardinality)
        small_cat_cols = [col for col in cat_cols if X_train[col].nunique() <= MAX_CATEGORIES_FOR_ONEHOT]
        high_cat_cols = [col for col in cat_cols if X_train[col].nunique() > MAX_CATEGORIES_FOR_ONEHOT]
        # One-hot encode low-cardinality categoricals (sparse output)
        if small_cat_cols:
            ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')
            X_train_onehot = ohe.fit_transform(X_train[small_cat_cols])
            X_test_onehot = ohe.transform(X_test[small_cat_cols])
            onehot_feature_names = ohe.get_feature_names_out(small_cat_cols)
        else:
            X_train_onehot = None
            X_test_onehot = None
            onehot_feature_names = []
        # Ordinal encode high-cardinality categoricals
        if high_cat_cols:
            ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_train_ord = ord_enc.fit_transform(X_train[high_cat_cols])
            X_test_ord = ord_enc.transform(X_test[high_cat_cols])
        else:
            ord_enc = None
            X_train_ord = None
            X_test_ord = None
        # Combine numeric, ordinal, and one-hot parts into sparse matrices
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
        # Feature name list after encoding
        feature_names_enc = numeric_cols + high_cat_cols + list(onehot_feature_names)
    else:
        # No categorical features; proceed without encoding
        X_train_enc = X_train.values
        X_test_enc = X_test.values
        feature_names_enc = list(X_train.columns)
        ord_enc = None  # no ordinal encoding used

    # Use encoded feature matrices for model training
    X_train = X_train_enc
    X_test = X_test_enc

    # Train initial models on all features
    clf, reg, encoder_or_bins = train_models(X_train, y_train, target_type)

    # Get feature importances from the appropriate model
    if target_type == "categorical" and clf is not None:
        importances = clf.feature_importances_
        feature_names = feature_names_enc
    elif reg is not None:
        importances = reg.feature_importances_
        feature_names = feature_names_enc
    else:
        raise ValueError("No valid model to determine feature importances.")

    # Select top 15 features by importance
    top_feats, top_importances = get_top_features(importances, feature_names, top_n=15)
    print("Top 15 features:")
    # Encode y_test for evaluation if classification
    if target_type == "categorical":
        le_temp = LabelEncoder()
        y_test_enc = le_temp.fit_transform(y_test.values.ravel())
    else:
        y_test_enc = y_test.values.ravel()
    for feat, imp in top_importances:
        print(f"  {feat}: {imp:.4f}")

    # Prepare original full dataset X for analysis (correlation, PDP, etc.)
    # If ordinal encoding was used for high-cardinality features, apply it to full X for consistency
    if high_cat_cols:
        for col in high_cat_cols:
            X[col] = X[col].astype(str)
        try:
            X[high_cat_cols] = ord_enc.transform(X[high_cat_cols])
        except Exception as e:
            print(f"Warning: could not ordinal-encode all high-cardinality features: {e}")
    # Add dummy columns for any one-hot encoded features in top_feats (to include in correlation and plots)
    for feat in top_feats:
        if feat not in X.columns:
            for col in cat_cols:
                if feat.startswith(str(col) + "_"):
                    cat_val = feat[len(col)+1:]
                    # Create dummy indicator column for this category
                    X[feat] = (X[col].astype(str) == cat_val).astype(int)
                    break

    # Correlation matrix of top 15 features (numeric features and any newly added dummy features)
    os.makedirs("plots", exist_ok=True)
    corr_df = X[top_feats].dropna()
    corr_matrix = corr_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Pearson Correlation Matrix of Top 15 Features")
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix.png")
    plt.show()

    # Cross-validation accuracy boxplot (5-fold StratifiedKFold) on top features
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    clf_cv = RandomForestClassifier(random_state=RANDOM_STATE)
    scores = cross_val_score(clf_cv, X[top_feats], y.values.ravel(), cv=skf, scoring='accuracy')
    plt.figure(figsize=(6, 6))
    sns.boxplot(y=scores, color='lightgreen')
    plt.title("5-fold Cross-validation Accuracy")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("plots/cv_accuracy_boxplot.png")
    plt.show()

    # Plot feature importances and save
    plot_feature_importance(importances, feature_names, "plots/feature_importance.png", target)
    print("Feature importance plot saved as 'plots/feature_importance.png'.")

    # Subset training and test data to top features only
    feat_index_map = {name: idx for idx, name in enumerate(feature_names_enc)}
    top_indices = [feat_index_map[f] for f in top_feats if f in feat_index_map]
    if isinstance(X_train, np.ndarray) or hasattr(X_train, 'toarray'):
        # If X_train is an array or sparse matrix, slice and convert to DataFrame
        X_train_sel_enc = X_train[:, top_indices]
        X_test_sel_enc = X_test[:, top_indices]
        X_train_sel = pd.DataFrame(X_train_sel_enc.toarray() if hasattr(X_train_sel_enc, 'toarray') else X_train_sel_enc,
                                   columns=top_feats)
        X_test_sel = pd.DataFrame(X_test_sel_enc.toarray() if hasattr(X_test_sel_enc, 'toarray') else X_test_sel_enc,
                                  columns=top_feats)
    else:
        # If X_train is still a DataFrame (no encoding needed), select columns directly
        X_train_sel = X_train[top_feats].copy()
        X_test_sel = X_test[top_feats].copy()

    # Retrain models on selected top features
    clf2, reg2, encoder_or_bins2 = train_models(X_train_sel, y_train, target_type)

    # Make predictions on test set using both models
    if target_type == "categorical":
        # Classifier predicted classes (encoded) and regressor predictions (encoded)
        clf_pred_enc = clf2.predict(X_test_sel)
        reg_pred = reg2.predict(X_test_sel)
        # Combine classifier and regressor predictions (average) and round to nearest class
        avg_pred = 0.5 * (clf_pred_enc + reg_pred)
        combined_num = np.rint(avg_pred).astype(int)
        # Clip combined predictions to valid label range and invert encoding to original labels
        le = encoder_or_bins  # LabelEncoder from initial training
        combined_num = np.clip(combined_num, 0, len(le.classes_) - 1)
        final_pred = le.inverse_transform(combined_num)
        clf_pred = le.inverse_transform(clf_pred_enc)
        # reg_pred remains numeric (encoded class values)
    else:
        # Continuous target
        reg_pred = reg2.predict(X_test_sel)
        if clf2 is not None:
            # If classifier was trained (target binned), map its predictions back to numeric scale
            clf_pred_bins = clf2.predict(X_test_sel)
            df_temp = pd.DataFrame({'target': y_train.values.ravel()})
            n_bins = len(np.unique(clf_pred_bins))
            df_temp['bin'], bins = pd.qcut(df_temp['target'], q=n_bins, labels=False, retbins=True, duplicates='drop')
            bin_medians = df_temp.groupby('bin')['target'].median().to_dict()
            class_pred_numeric = np.array([bin_medians.get(b, np.nan) for b in clf_pred_bins])
            final_pred = 0.5 * (class_pred_numeric + reg_pred)
            clf_pred = clf_pred_bins
            # Store combined prediction in a global variable if needed
            global ECO_CLASS
            ECO_CLASS = final_pred
        else:
            final_pred = reg_pred
            clf_pred = [None] * len(reg_pred)

    # Save predictions and true labels to CSV
    save_predictions_csv("model_predictions.csv", X_test_sel, y_test, clf_pred, reg_pred, final_pred)
    print("Predictions saved to 'model_predictions.csv'.")

    # Plot feature distributions for top features on entire dataset
    df_plot = X[top_feats].copy()
    df_plot[target] = y.values
    df_plot = df_plot.dropna()
    if target_type == "categorical":
        df_plot[target] = df_plot[target].astype(str)
        # Treat dummy features as categorical for plotting
        for feat in top_feats:
            if feat not in df_model.columns:
                df_plot[feat] = df_plot[feat].astype('category')
    plot_feature_distributions(df_plot, top_feats, target, target_type, output_dir="feature_plots")
    print("Feature distribution plots saved to 'feature_plots/' directory.")

    # Evaluate performance of top features (accuracy vs F1 scatter)
    evaluate_and_plot_feature_performance(X_test_sel, pd.Series(y_test_enc), top_feats)
    model_used = clf2 if target_type == "categorical" else reg2
    return final_pred, model_used, X_train_sel, top_feats

from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.base import is_classifier

def explain_model(model, X, feature_names=None, target_name="target", top_n=15):
    """
    Generate Partial Dependence Plots for the top_n features of the model 
    and provide simple trend-based textual explanations of feature effects on the target.
    """
    # Ensure feature_names list
    if feature_names is None:
        try:
            feature_names = list(X.columns)
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    # Determine top features by importance if available
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:top_n]
    else:
        top_idx = np.arange(min(top_n, X.shape[1]))
    top_features = [feature_names[i] for i in top_idx]

    is_class = is_classifier(model)
    numeric_classes = False
    if is_class:
        try:
            classes = model.classes_
            numeric_classes = np.issubdtype(classes.dtype, np.number)
        except Exception:
            numeric_classes = False

    explanations = []
    # Loop through each top feature
    for idx, fname in zip(top_idx, top_features):
        # Compute partial dependence with individual predictions for std calculation
        try:
            pdp_res = partial_dependence(model, X, [idx], feature_names=feature_names, kind='both')
        except TypeError:
            pdp_res = partial_dependence(model, X, [idx], feature_names=feature_names)
        # Extract grid values and average predictions
        grid = pdp_res["values"][0] if "values" in pdp_res else pdp_res["grid_values"][0]
        avg_data = pdp_res["average"]
        avg_preds = np.ravel(avg_data[0]) if avg_data.ndim > 1 else np.ravel(avg_data)
        # Interpret monotonic trend
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

        # Plot the partial dependence with standard deviation bands
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grid, avg_preds, label="Average prediction")
        indiv_curves = pdp_res.get("individual", None)
        if indiv_curves is not None:
            if indiv_curves.ndim > 2:
                indiv_curves = indiv_curves[0]  # first output if multi-output
            std = np.std(indiv_curves, axis=0)
            ax.fill_between(grid, avg_preds - std, avg_preds + std, alpha=0.3, label="\u00B1 1 std")
        ax.set_xlabel(fname)
        ax.set_ylabel(target_name)
        ax.set_title(f"PDP of {fname}")
        if indiv_curves is not None:
            ax.legend()
        plt.tight_layout()
        plt.show()

    # Print textual explanations for each top feature
    print("\n".join(explanations))

    # 2D Partial Dependence Plots for specified feature pairs
    if PDP_2D_PAIRS:
        for f1, f2 in PDP_2D_PAIRS:
            if f1 in feature_names and f2 in feature_names:
                idx1 = feature_names.index(f1)
                idx2 = feature_names.index(f2)
                print(f"\nGenerating 2D PDP for features: {f1} and {f2}")

            # Use a smaller subset of X for faster computation
            X_pdp = X.copy()
            if len(X_pdp) > 2000:  # Limit sample size
                X_pdp = X_pdp.sample(n=2000, random_state=RANDOM_STATE)

            try:
                # You want to target a specific ecological class for PDP
                TARGET_CLASS_NAME = "Moderate"

                # Get the class index from the trained classifier
                class_index = model.classes_.tolist().index(TARGET_CLASS_NAME)

                # Use that in PDP
                display = PartialDependenceDisplay.from_estimator(
                    model,
                    X_pdp,
                    [(idx1, idx2)],
                    feature_names=feature_names,
                    grid_resolution=25,
                    target=class_index,
                )
                plt.tight_layout()
                out_file = f"plots/pdp_2d_{f1}_{f2}.png"
                plt.savefig(out_file, bbox_inches="tight")
                plt.close()
                print(f"2D PDP saved as {out_file}")
            except Exception as e:
                # Fallback if from_estimator fails
                print(f" Skipped 2D PDP for {f1} & {f2}: {e}")
                try:
                    pdp_res_2d = partial_dependence(
                        model, X_pdp, [(idx1, idx2)],
                        feature_names=feature_names,
                        grid_resolution=25
                    )
                    grid1 = pdp_res_2d['grid_values'][0]
                    grid2 = pdp_res_2d['grid_values'][1]
                    avg_preds_2d = pdp_res_2d['average'][0]
                    Xx, Yy = np.meshgrid(grid2, grid1)
                    plt.figure(figsize=(6, 5))
                    plt.contourf(Xx, Yy, avg_preds_2d, cmap='viridis')
                    plt.colorbar(label='Predicted')
                    plt.xlabel(f2)
                    plt.ylabel(f1)
                    plt.title(f"2D PDP of {f1} and {f2}")
                    out_file = f"plots/pdp_2d_{f1}_{f2}_fallback.png"
                    plt.savefig(out_file, bbox_inches="tight")
                    plt.close()
                    print(f"2D PDP saved (fallback) as {out_file}")
                except Exception as ee:
                    print(f"Failed 2D PDP fallback for {f1} & {f2}: {ee}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data processing and modeling")
    parser.add_argument("--excel_path", type=str, default="WFD_SW_Classification_Status_and_Objectives_Cycle2_v4.xlsx")
    parser.add_argument("--parquet_path", type=str, default="wims_wfd_merged.parquet")
    parser.add_argument("--target", type=str, default="Ecological Class")
    args = parser.parse_args()
    final_pred, model, X_train_used, feature_names_used = main(args)
    explain_model(model, X_train_used, feature_names=feature_names_used, target_name=args.target, top_n=15)
    
    
    
# accuracy and f1 for just predicting "moderate"
# try quantiles
# softwhere testing
# alterative AI prosseing
# more interactive data visulisation "streamlit"
# orange "load in data and describe it"
# data visuliation, robustness, more usfull
# interal anual varaiblity
# intigrate it into an "app" - streamlit