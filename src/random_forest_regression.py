# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 20:18:36 2025

@author: tompe
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 20:01:00 2025

@author: tompe
"""

"""
Script to load and merge data from Excel and Parquet files, perform feature 
selection, train Random Forest models (classifier and regressor), and output 
predictions and plots.

Usage:
    python script.py --excel_path <path> --parquet_path <path> [--target TARGET]

Requirements:
    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - seaborn
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

# Set random seeds for reproducibility
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def load_excel_data(excel_path):
    """
    Load all sheets from the Excel file, concatenate them, and drop columns with >=50% missing values.
    Skips the second header row which repeats labels.
    """
    all_sheets = []
    try:
        # Use pandas to get sheet names
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
    # Concatenate all sheets into one DataFrame
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
        if series.nunique() <= cat_threshold:
            return "categorical"
        else:
            return "continuous"
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
        # Bin the target into up to 10 bins for classification
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
    ax = plt.gca()
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
    Categorical features: countplot; Continuous: boxplot.
    Grouped by target variable (or its binned version if continuous).
    """
    os.makedirs(output_dir, exist_ok=True)
    df_plot = df.copy()
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
    """
    Evaluates each feature and all features together using StratifiedKFold and 
    plots accuracy vs F1. Saves the plot to the specified path.
    """
    import os
    os.makedirs("plots", exist_ok=True)

    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for feat in feature_names:
        acc_scores, f1_scores = [], []
        for train_idx, test_idx in skf.split(X[[feat]], y):
            X_train, X_test = X.iloc[train_idx][[feat]], X.iloc[test_idx][[feat]]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc_scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='macro'))
        results.append({"Feature": feat, "Accuracy": np.mean(acc_scores), "F1 Score": np.mean(f1_scores)})

    # Evaluate all features together
    acc_scores, f1_scores = [], []
    for train_idx, test_idx in skf.split(X[feature_names], y):
        X_train, X_test = X.iloc[train_idx][feature_names], X.iloc[test_idx][feature_names]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='macro'))
    results.append({"Feature": "All Features", "Accuracy": np.mean(acc_scores), "F1 Score": np.mean(f1_scores)})

    # Plot
    df_results = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.scatter(df_results["Accuracy"], df_results["F1 Score"], c='skyblue', edgecolors='black')
    for i, row in df_results.iterrows():
        plt.annotate(row["Feature"], (row["Accuracy"] + 0.001, row["F1 Score"] + 0.001), fontsize=8)
    plt.xlabel("Accuracy")
    plt.ylabel("F1 Score")
    plt.title("Accuracy vs F1 Score for Individual and Combined Predictors")
    plt.grid(True)
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
    
    print("Excel columns:", df_excel.columns.tolist())
    print("Parquet columns:", df_parquet.columns.tolist())
    
    # Merge on common keys
    df_merged = merge_datasets(df_excel, df_parquet)
    print(f"Merged data: {df_merged.shape} (rows, columns)")

    target = args.target
    if target not in df_merged.columns:
        raise ValueError(f"Target '{target}' not in merged data.")

    # Drop rows with missing target
    df_merged = df_merged.dropna(subset=[target]).reset_index(drop=True)
    X = df_merged.drop(columns=[target])
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].astype(str)
        X[col] = LabelEncoder().fit_transform(X[col])
    
    y = df_merged[[target]]

    # Determine target type
    target_type = determine_target_type(y[target])
    print(f"Target '{target}' is detected as {target_type}.")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Train initial models on all features
    clf, reg, encoder_or_bins = train_models(X_train, y_train, target_type)

    # Feature importances from classifier or regressor
    if target_type == "categorical" and clf is not None:
        importances = clf.feature_importances_
        feature_names = X_train.columns
    elif reg is not None:
        importances = reg.feature_importances_
        feature_names = X_train.columns
    else:
        raise ValueError("No valid model to determine feature importances.")

    # Select top 15 features
    top_feats, top_importances = get_top_features(importances, feature_names, top_n=15)
    print("Top 15 features:")
    if target_type == "categorical":
        le = LabelEncoder()
        y_test_enc = le.fit_transform(y_test.values.ravel())
    else:
        y_test_enc = y_test.values.ravel()
    
    for feat, imp in top_importances:
        print(f"  {feat}: {imp:.4f}")

    # Correlation matrix of top 15 features
    os.makedirs("plots", exist_ok=True)
    corr_df = X[top_feats].dropna()
    corr_matrix = corr_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Pearson Correlation Matrix of Top 15 Features")
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix.png")
    plt.show()

    # Cross-validation accuracy boxplot (5-fold StratifiedKFold)
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

    # Plot feature importances (highlighting top features for target prediction)
    plot_feature_importance(importances, feature_names, "plots/feature_importance.png", target)
    print("Feature importance plot saved as 'plots/feature_importance.png'.")

    # Subset data to top features
    X_train_sel = X_train[top_feats]
    X_test_sel = X_test[top_feats]

    # Retrain models on selected features
    clf2, reg2, encoder_or_bins2 = train_models(X_train_sel, y_train, target_type)

    # Predictions
    if target_type == "categorical":
        # Predict class (encoded)
        clf_pred_enc = clf2.predict(X_test_sel)
        reg_pred = reg2.predict(X_test_sel)
        # Combine predictions: average and round
        avg_pred = 0.5 * (clf_pred_enc + reg_pred)
        combined_num = np.rint(avg_pred).astype(int)
        # Clip to valid range
        le = encoder_or_bins  # LabelEncoder from train
        combined_num = np.clip(combined_num, 0, len(le.classes_) - 1)
        final_pred = le.inverse_transform(combined_num)
        clf_pred = le.inverse_transform(clf_pred_enc)
        # Keep reg_pred numeric as float (predicted encoded class values)
    else:
        # Continuous target
        reg_pred = reg2.predict(X_test_sel)
        if clf2 is not None:
            # Map classifier's binned predictions to numeric via median of bins
            clf_pred_bins = clf2.predict(X_test_sel)
            df_temp = pd.DataFrame({'target': y_train.values.ravel()})
            n_bins = len(np.unique(clf_pred_bins))
            df_temp['bin'], bins = pd.qcut(df_temp['target'], q=n_bins, labels=False, retbins=True, duplicates='drop')
            bin_medians = df_temp.groupby('bin')['target'].median().to_dict()
            class_pred_numeric = np.array([bin_medians.get(b, np.nan) for b in clf_pred_bins])
            final_pred = 0.5 * (class_pred_numeric + reg_pred)
            clf_pred = clf_pred_bins
            
            global ECO_CLASS
            ECO_CLASS = final_pred
        else:
            final_pred = reg_pred
            clf_pred = [None] * len(reg_pred)

    # Save predictions and true labels
    save_predictions_csv("model_predictions.csv", X_test_sel, y_test, clf_pred, reg_pred, final_pred)
    print("Predictions saved to 'model_predictions.csv'.")

    # Plot feature distributions on the merged dataset for top features
    df_plot = df_merged[top_feats + [target]].dropna()
    if target_type == "categorical":
        df_plot[target] = df_plot[target].astype(str)
    plot_feature_distributions(df_plot, top_feats, target, target_type, output_dir="feature_plots")
    print("Feature distribution plots saved to 'feature_plots/' directory.")

    evaluate_and_plot_feature_performance(X_test_sel, pd.Series(y_test_enc), top_feats)
    #return final_pred
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
        # Compute partial dependence (average)
        try:
            pdp_res = partial_dependence(model, X, [idx], feature_names=feature_names, kind='average')
        except TypeError:
            pdp_res = partial_dependence(model, X, [idx], feature_names=feature_names)
        # Extract grid values and average predictions
        if "values" in pdp_res:
            grid = pdp_res["values"][0]
        else:
            grid = pdp_res["grid_values"][0]
        avg_data = pdp_res["average"]
        if avg_data.ndim > 1:
            avg_preds = np.ravel(avg_data[0])
        else:
            avg_preds = np.ravel(avg_data)
        
        # Interpret trend
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
        
        # Plot the partial dependence
        try:
            display = PartialDependenceDisplay.from_estimator(model, X, [idx], feature_names=feature_names)
            # Label the plot
            # The axes_ array is 2D even for one feature; access first element
            ax = display.axes_[0][0] if hasattr(display.axes_, 'shape') else display.axes_[0]
            ax.set_title(f"PDP of {fname}")
            plt.tight_layout()
            plt.show()
        except Exception:
            # Fallback manual plot if needed
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(grid, avg_preds, marker='o')
            ax.set_xlabel(fname)
            ax.set_ylabel(target_name)
            ax.set_title(f"PDP of {fname}")
            plt.tight_layout()
            plt.show()
    
    # Print textual explanations
    print("\n".join(explanations))





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data processing and modeling")
    parser.add_argument("--excel_path", type=str,
                        default="WFD_SW_Classification_Status_and_Objectives_Cycle2_v4.xlsx")
    parser.add_argument("--parquet_path", type=str,
                        default="wims_wfd_merged.parquet")
    parser.add_argument("--target", type=str, default="Ecological Class")
    args = parser.parse_args()
    
    ECO_CLASS, model, X_train_used, feature_names_used = main(args)
    explain_model(model, X_train_used, feature_names=feature_names_used, target_name=args.target, top_n=15)
    
#multi varaible pdp plots
#only using chemicl data
# presentations    






"""
import pandas as pd
# Load the data
wims_wfd_merged = pd.read_parquet(r"\path\to\wims_wfd_merged.parquet")

# Use only the (e.g.,) variables with > million samples
counts = wims_wfd_merged.variable.value_counts()
counts = counts[counts>1e6]
wims_wfd_merged = wims_wfd_merged.loc[wims_wfd_merged.variable.isin(counts.index)]

# Aggregate data to catchment-year-variable scale (try varying from mean, median, quantile)
grouped = wims_wfd_merged.groupby(["wb_id","year","variable"]).result.mean().reset_index()

# Format for sklearn
pivoted = grouped.pivot(index=["wb_id","year"],columns=["variable"],values="result")
print(pivoted.reset_index())
"""