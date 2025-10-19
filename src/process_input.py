import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _load_classification_data(excel_path):
    """
    Load all sheets from the Excel file, concatenate them, and drop columns with >=50% missing values.
    Skips the second header row which repeats labels.
    """
    all_sheets = []
    try:
        xls = pd.ExcelFile(excel_path, engine="openpyxl")
        sheets = xls.sheet_names
    except Exception as e:
        raise Exception(f"Error reading Excel file: {e}")

    for sheet in sheets:
        # Read sheet, skip the repeated header row at index 1
        df_sheet = pd.read_excel(
            excel_path, sheet_name=sheet, skiprows=1, engine="openpyxl"
        )
        all_sheets.append(df_sheet)
    if not all_sheets:
        raise ValueError("No sheets found in Excel file.")
    df_excel = pd.concat(all_sheets, ignore_index=True)
    # Drop columns with 50% or more missing values
    thresh = len(df_excel) * 0.5
    df_excel = df_excel.loc[:, df_excel.isnull().sum() < thresh]
    return df_excel


def _aggregate_parquet(df_parquet):
    """
    Aggregate WIMS data by wb_id with richer features for better predictive power.
    """
    numeric_cols = df_parquet.select_dtypes(include=np.number).columns.tolist()

    agg_df = df_parquet.groupby("wb_id")[numeric_cols].agg(
        [
            "mean",
            "median",
            "std",
            "min",
            "max",
            lambda x: x.quantile(0.25),  # Q1
            lambda x: x.quantile(0.75),
        ]  # Q3
    )

    agg_df.columns = [
        "_".join(col).strip() if isinstance(col, tuple) else col
        for col in agg_df.columns
    ]
    agg_df = agg_df.rename(
        columns={
            **{
                col: col.replace("<lambda_0>", "Q1").replace("<lambda_1>", "Q3")
                for col in agg_df.columns
            }
        }
    )

    # Compute IQR = Q3 - Q1
    for num_col in numeric_cols:
        q1_col = f"{num_col}_<lambda_0>"
        q3_col = f"{num_col}_<lambda_1>"
        if q1_col in agg_df.columns and q3_col in agg_df.columns:
            agg_df[f"{num_col}_IQR"] = agg_df[q3_col] - agg_df[q1_col]

    df_parquet["dynamic_threshold"] = df_parquet.groupby("variable")[
        "result"
    ].transform(lambda x: x.mean() + 2 * x.std())
    df_parquet["above_threshold"] = (
        df_parquet["result"] > df_parquet["dynamic_threshold"]
    ).astype(int)

    agg_extra = (
        df_parquet.groupby("wb_id")["above_threshold"]
        .agg(["sum", "mean"])
        .reset_index()
    )
    agg_extra.rename(
        columns={"sum": "count_above_thr", "mean": "prop_above_thr"}, inplace=True
    )

    if "date" in df_parquet.columns:
        df_parquet["ordinal_time"] = pd.to_datetime(df_parquet["date"]).map(
            pd.Timestamp.toordinal
        )
        slope_df = (
            df_parquet.groupby("wb_id")
            .apply(
                lambda g: np.polyfit(g["ordinal_time"], g["result"], 1)[0]
                if len(g) > 1
                else 0
            )
            .reset_index(name="trend_slope")
        )
    else:
        slope_df = pd.DataFrame(
            {"wb_id": df_parquet["wb_id"].unique(), "trend_slope": 0}
        )

    agg_df = agg_df.reset_index()
    agg_df = agg_df.merge(agg_extra, on="wb_id", how="left")
    agg_df = agg_df.merge(slope_df, on="wb_id", how="left")

    return agg_df


def _load_chemical_data(parquet_path):
    """
    Load Parquet file and drop columns with <1,000,000 non-null values.
    """
    try:
        df_parquet = pd.read_parquet(parquet_path)
    except ImportError:
        raise ImportError(
            "PyArrow or fastparquet is required to read Parquet files. Install one to proceed."
        )
    except Exception as e:
        raise Exception(f"Error reading Parquet file: {e}")

    # Keep columns with at least 1,000,000
    non_null_counts = df_parquet.notnull().sum()
    cols_to_keep = non_null_counts[non_null_counts >= 1000000].index
    df_parquet = df_parquet.loc[:, cols_to_keep]
    return df_parquet


def process_and_aggregate(args):
    """Load and process input data from classification (Excel) and chemical (Parquet) files,
    then merge them."""
    df_excel = _load_classification_data(args.excel_path)

    df_excel = df_excel.drop(columns=["Overall Water Body Class"], errors="ignore")
    LOGGER.info(f"Excel data: {df_excel.shape} (rows, columns)")
    df_parquet = _load_chemical_data(args.parquet_path)
    LOGGER.info(f"Parquet data: {df_parquet.shape} (rows, columns)")

    if "Water Body ID" not in df_excel.columns:
        raise KeyError("Excel file missing 'Water Body ID'")
    if "wb_id" not in df_parquet.columns:
        raise KeyError("Parquet file missing 'wb_id'")

    target_col = args.target
    df_excel = df_excel.rename(columns={"Water Body ID": "wb_id"})
    target_df = df_excel[["wb_id", target_col]]

    var_counts = df_parquet["variable"].value_counts()
    frequent_vars = var_counts[var_counts > 1e6].index.tolist()

    LOGGER.info(f"Found {len(frequent_vars)} frequent variables with > 1M records")
    LOGGER.info("Sample: %s", frequent_vars[:5])

    df_filtered = df_parquet[df_parquet["variable"].isin(frequent_vars)]

    pivot_df = df_filtered.pivot_table(
        index="wb_id", columns="variable", values="result", aggfunc="mean"
    ).reset_index()

    df_excel = df_excel.rename(columns={"Water Body ID": "wb_id"})
    target_df = df_excel[["wb_id", target_col]]
    df_merged = pd.merge(target_df, pivot_df, on="wb_id", how="inner")

    LOGGER.info(f"Merged data: {df_merged.shape} (rows, columns)")
    df_merged = df_merged.dropna()
    LOGGER.info(f"Merged data without nulls: {df_merged.shape} (rows, columns)")

    return df_merged
