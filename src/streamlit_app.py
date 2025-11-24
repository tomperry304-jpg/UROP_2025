"""Model plots dashboard + UK catchment predictions map (single file).

- When run normally (Spyder / python), it launches Streamlit and shows:
  • Feature plots from ./feature_plots and ./plots
  • An embedded Folium map of your predictions joined to polygons
  • Also saves out/merged.geojson and out/map.html

- When run with --cli it only builds the map artifacts (no Streamlit).

This fixes the Streamlit "set_page_config can only be called once" error
by moving ALL st.* calls inside render_app() and calling it only from
the Streamlit process.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd

# Import Streamlit (avoid calling st.* at module import time)
import streamlit as st
import streamlit.components.v1 as components

# -------------------------- DEFAULT PATHS & SETTINGS --------------------------
# Define default input file paths, output file paths, prediction column,
# and other constants
SCRIPT_DIR = Path(__file__).resolve().parent

# Default geometry and CSV file paths (adjust if needed)
DEFAULT_GEO_PATH = Path(r"C:\UROP\WFD_River_Water_Body_Catchments_Cycle_2.zip")

DEFAULT_CSV_PATH = Path(
    r"C:\Users\tompe\Documents\UROP_local\src\model_predictions.csv"
)

# DEFAULT_CSV_PATH = Path(
#    r"C:\Users\tompe\OneDrive - Imperial College London\Year 2\UROP\"
#    r"Project\model_predictions\final_predictions.csv"
# )

# Default prediction column name for map colouring
DEFAULT_PRED_COL = "Pred_Combined"

# Output file paths
OUT_DIR = SCRIPT_DIR / "out"
OUT_MERGED_GEO = OUT_DIR / "merged.geojson"
OUT_MAP_HTML = OUT_DIR / "map.html"

# Web map simplification tolerance in meters (0 disables)
SIMPLIFY_TOL_M = 60

# Directories containing plot images
FEATURE_DIR = SCRIPT_DIR / "feature_plots"
PLOTS_DIR = SCRIPT_DIR / "plots"

# Filename patterns to categorize plots
PATTERNS = {
    "summary": ["correlation", "feature_importance", "feature_score", "scatter"],
    "pdp_1d": ["pdp_1d_"],
    "pdp_2d": ["pdp_2d_"],
}

# Colour palette for map (categorical or quantile bins)
PALETTE = [
    "#3182bd",
    "#41ab5d",
    "#de2d26",
    "#756bb1",
    "#8c510a",
    "#fc8d62",
    "#66c2a5",
    "#a6d854",
    "#e78ac3",
    "#bdbdbd",
]


# -------------------------- UTILITY FUNCTIONS --------------------------
# Utility functions for geometry data loading and processing
# (file path resolution, geometry reading, CRS normalization, simplification)
def resolve_geo_path(raw: str | Path, search_dir: Path) -> Path:
    """Resolve a path to an existing geometry file.

    Handles relative paths and missing extensions.
    """
    cand = Path(raw)
    if not cand.is_absolute():
        cand = (search_dir / cand).resolve()
    if cand.exists():
        return cand

    stem = cand.with_suffix("")
    for p in [
        stem.with_suffix(".zip"),
        stem.with_suffix(".shp"),
        stem,
        stem.with_suffix(".geojson"),
        stem.with_suffix(".json"),
    ]:
        if p.exists():
            return p

    for p in search_dir.glob(stem.name + "*"):
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find geometry at '{raw}' (searched around {search_dir})."
    )


def read_geometry_any(path_like: str | Path, script_dir: Path) -> gpd.GeoDataFrame:
    """Read geometry data from various formats into a GeoDataFrame.

    Supports zip, shapefile, GeoJSON formats. Returns WGS84 CRS.
    """
    p = resolve_geo_path(path_like, script_dir)

    if p.suffix.lower() in (".geojson", ".json"):
        gdf = gpd.read_file(p)
    elif p.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(td)
            shp = next(iter(Path(td).rglob("*.shp")), None)
            if shp is None:
                raise ValueError("ZIP has no .shp inside.")
            gdf = gpd.read_file(str(shp))
    elif p.is_dir():
        shp = next(iter(p.rglob("*.shp")), None)
        if shp is None:
            raise ValueError(f"Folder '{p}' contains no .shp.")
        gdf = gpd.read_file(str(shp))
    elif p.suffix.lower() == ".shp":
        gdf = gpd.read_file(str(p))
    else:
        raise ValueError(f"Unsupported geometry path: {p}")

    # normalise CRS
    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
    else:
        gdf = gdf.to_crs(4326)

    return gdf


def simplify_for_web(gdf: gpd.GeoDataFrame, tol_m: float) -> gpd.GeoDataFrame:
    """Simplify geometry for web visualization (reduce detail to tolerance)."""
    if tol_m <= 0:
        return gdf
    tmp = gdf.copy()
    try:
        tmp = tmp.to_crs(3857)
        tmp["geometry"] = tmp.geometry.simplify(tol_m, preserve_topology=True)
        tmp = tmp.to_crs(4326)
    except Exception:
        return gdf
    return tmp


# -------------------------- PREDICTION DATA JOINING --------------------------
# Functions to normalize identifiers and join prediction CSV data to geometry
def norm_series(s: pd.Series) -> pd.Series:
    """Normalize a Series.

    Strips whitespace, lowercases, and removes non-alphanumeric characters.
    """
    return (
        pd.Series(s, dtype="string")
        .str.strip()
        .str.lower()
        .map(lambda x: re.sub(r"[^0-9a-z]+", "", x) if pd.notna(x) else x)
    )


def find_best_join(
    gdf: gpd.GeoDataFrame, df: pd.DataFrame
) -> tuple[str | None, str | None, int]:
    """Find fields with greatest overlap of normalized values.

    Compares GeoDataFrame and DataFrame fields.
    """
    geo_fields = [c for c in gdf.columns if c != gdf.geometry.name]
    csv_fields = df.columns.tolist()
    best = (None, None, -1)

    norm_geo = {g: set(norm_series(gdf[g]).dropna().unique()) for g in geo_fields}
    norm_csv = {c: set(norm_series(df[c]).dropna().unique()) for c in csv_fields}

    for g in geo_fields:
        gset = norm_geo.get(g, set())
        if not gset:
            continue
        for c in csv_fields:
            cset = norm_csv.get(c, set())
            if not cset:
                continue
            overlap = len(gset & cset)
            if overlap > best[2]:
                best = (g, c, overlap)
    return best


def join_predictions(
    gdf: gpd.GeoDataFrame,
    df: pd.DataFrame,
    pred_col: str,
    logger: Callable[[str], None] | None = print,
) -> gpd.GeoDataFrame:
    """Join prediction data with geometry data.

    Prefer exact join on "wb_id". If no match, use heuristic text matching
    on other fields, and as a last resort, join by row order.
    """
    log = logger or (lambda *_: None)

    # Normalise CSV headers (strip BOM/whitespace)
    df.rename(columns=lambda c: str(c).strip().lstrip("\ufeff"), inplace=True)

    # 1) Force exact join on wb_id if possible
    if "wb_id" in gdf.columns and "wb_id" in df.columns:
        log("Forcing exact join on 'wb_id'…")
        left = gdf.copy()
        right = df.copy()
        left["wb_id"] = left["wb_id"].astype(str).str.strip().str.upper()
        right["wb_id"] = right["wb_id"].astype(str).str.strip().str.upper()
        df_join = right[["wb_id", pred_col]].copy()
        merged = left.merge(df_join, on="wb_id", how="left")
        log(f"Exact wb_id overlap: {merged[pred_col].notna().sum()}/{len(merged)}")
        return merged

    # 2) Best heuristic join
    g_field, c_field, overlap = find_best_join(gdf, df)
    log(
        f"Heuristic match → geometry: '{g_field}'  CSV: '{c_field}'  "
        f"(overlap: {overlap})"
    )

    if g_field and c_field and overlap > 0:
        left = gdf.copy()
        right = df.copy()
        left[g_field] = left[g_field].astype(str).str.strip()
        right[c_field] = right[c_field].astype(str).str.strip()
        right = right[[c_field, pred_col]].dropna().copy()
        right.rename(columns={c_field: g_field}, inplace=True)
        merged = left.merge(right, on=g_field, how="left")
        log(
            f"Strict overlap (non-null): {merged[pred_col].notna().sum()}/{len(merged)}"
        )
        return merged

    # 3) Loose normalised join
    log("No strict overlap. Trying loose match (case/space/punct removed)…")
    geom_fields = [c for c in gdf.columns if c != gdf.geometry.name]
    if not geom_fields:
        raise ValueError("No candidate geometry fields to join on.")
    g_field = max(geom_fields, key=lambda c: gdf[c].nunique(dropna=True))

    id_like = [c for c in df.columns if df[c].nunique(dropna=True) > 50]
    c_field = id_like[0] if id_like else df.columns[0]

    left = gdf.copy()
    right = df.copy()
    left["_norm_key"] = norm_series(left[g_field])
    right["_norm_key"] = norm_series(right[c_field])

    merged = left.merge(right[["_norm_key", pred_col]], on="_norm_key", how="left")
    nn = merged[pred_col].notna().sum()
    log(f"Loose match overlap (non-null): {nn}/{len(merged)}")
    if nn > 0:
        return merged

    # 4) Final fallback: positional (warn)
    log("⚠️  No key overlap at all. Falling back to positional join (row order).")
    merged = gdf.copy()
    merged[pred_col] = pd.Series(df[pred_col].values[: len(gdf)], index=merged.index)
    return merged


# -------------------------- MAP BUILDING FUNCTIONS --------------------------
# Functions to build the Folium map and orchestrate map generation
def build_map(
    merged_gdf: gpd.GeoDataFrame,
    pred_col: str,
    name_cols: list[str] | None = None,
) -> folium.Map:
    """Create a Folium map with features colored by the given prediction column."""
    if name_cols is None:
        name_cols = ["wb_name", "wb_id", "rbd_name", "rbd_id", "OC_NAME", "id"]

    union_geom = merged_gdf.geometry.unary_union
    centroid = union_geom.centroid
    m = folium.Map(
        location=[float(centroid.y), float(centroid.x)],
        zoom_start=6,
        tiles="cartodbpositron",
    )

    geom_types = set(merged_gdf.geom_type.astype(str).str.lower().unique())
    is_polygonal = any(t in geom_types for t in ["polygon", "multipolygon"])
    is_numeric = pd.api.types.is_numeric_dtype(merged_gdf[pred_col])

    series = merged_gdf[pred_col]
    # Determine color mapping scheme (numeric vs categorical values)
    if is_numeric:
        vals = pd.to_numeric(series, errors="coerce")
        finite = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty or float(finite.min()) == float(finite.max()):
            labels = ["No variation"]

            def pick_color(v: float | str | None) -> str:
                return PALETTE[-1]
        else:
            q = np.unique(np.quantile(finite, np.linspace(0, 1, 6)))
            if len(q) < 6:
                q = np.linspace(float(finite.min()), float(finite.max()), 6)
            labels = [f"[{q[i]:.3g}, {q[i+1]:.3g}]" for i in range(5)]

            def pick_color(v: float | str | None) -> str:
                if pd.isna(v):
                    return PALETTE[-1]
                v_float = float(v) if isinstance(v, int | float) else 0.0
                for i in range(5):
                    if (v_float >= q[i]) and (
                        v_float < q[i + 1] or (i == 4 and v_float <= q[i + 1])
                    ):
                        return PALETTE[i]
                return PALETTE[-1]
    else:
        cats = pd.Categorical(series.astype(str)).categories.tolist()
        labels = cats if cats else ["No data"]
        cmap = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(labels)}

        def pick_color(v: float | str | None) -> str:
            if v is None:
                return PALETTE[-1]
            return cmap.get(str(v), PALETTE[-1])

    def best_name(props: dict[str, object]) -> str:
        for k in name_cols:
            if k in props and props[k] not in (None, "", "nan"):
                return str(props[k])
        return str(props.get("id", "Feature"))

    def _get_property_value(feat: dict[str, object], key: str) -> str | None:
        """Extract property value from feature dict."""
        props = feat.get("properties")
        if isinstance(props, dict):
            return str(props.get(key))
        return None

    gj = json.loads(merged_gdf.to_json())
    for f in gj["features"]:
        props = f.get("properties", {})
        props["_tip"] = (
            f"{best_name(props)} — {pred_col}: {props.get(pred_col, 'No data')}"
        )

    def style_polygon(feat: dict[str, object]) -> dict[str, str | float]:
        props = feat.get("properties", {})
        if not isinstance(props, dict):
            props = {}
        v = props.get(pred_col)
        c = pick_color(pd.to_numeric(v, errors="coerce") if is_numeric else v)
        return {"fillColor": c, "color": "#555", "weight": 1, "fillOpacity": 0.7}

    folium.GeoJson(
        gj,
        style_function=style_polygon
        if is_polygonal
        else (
            lambda f: {
                "color": pick_color(
                    _get_property_value(f, pred_col) if isinstance(f, dict) else None
                ),
                "weight": 4,
            }
        ),
        tooltip=folium.GeoJsonTooltip(
            fields=["_tip"], aliases=[""], labels=False, sticky=True
        ),
        highlight_function=(lambda _: {"weight": 3, "color": "#000"})
        if is_polygonal
        else None,
    ).add_to(m)

    # Legend
    legend_html = [
        "<div style='position: fixed; bottom: 20px; left: 20px; z-index: 9999; "
        "background: white; padding: 10px; border:1px solid #888; "
        "box-shadow: 0 1px 4px rgba(0,0,0,0.2); font-size: 12px;'>",
        f"<b>{pred_col}</b><br>",
    ]
    labels_to_show = (
        sorted({str(v) for v in series.dropna().unique()}) if not is_numeric else labels
    )
    for i, lbl in enumerate(labels_to_show):
        color = PALETTE[i % len(PALETTE)]
        legend_html.append(
            f"<div><span style='display:inline-block;width:12px;height:12px;"
            f"background:{color};border:1px solid #555;margin-right:6px;'></span>"
            f"{lbl}</div>"
        )
    legend_html.append("</div>")
    m.get_root().html.add_child(folium.Element("".join(legend_html)))
    return m


def build_and_save_map(
    geo_path: Path,
    csv_path: Path,
    pred_col: str,
    out_geo: Path,
    out_html: Path,
    logger: Callable[[str], None] | None = print,
) -> tuple[gpd.GeoDataFrame, folium.Map]:
    """Read input files, join predictions, build map, and save output files."""
    log = logger or (lambda *_: None)
    # Load input files and join prediction data
    log(f"Reading geometry: {geo_path}")
    gdf = read_geometry_any(geo_path, SCRIPT_DIR)
    log(f"Geometry features: {len(gdf)}")

    log(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    merged = join_predictions(gdf, df, pred_col, logger=log)

    out_geo.parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing merged GeoJSON → {out_geo}")
    merged.to_file(out_geo, driver="GeoJSON")

    # Simplify geometry and create Folium map
    map_gdf = simplify_for_web(merged, SIMPLIFY_TOL_M)
    log("Building Folium map…")
    fmap = build_map(map_gdf, pred_col)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(out_html)
    log(f"Saved map → {out_html}")
    return merged, fmap


# -------------------------- STREAMLIT APP --------------------------
# Streamlit app components for interactive plots and map interface
def imgs_in(folder: Path) -> list[Path]:
    """Return a sorted list of PNG images in the given folder."""
    if not folder.exists():
        return []
    hits = list(folder.glob("*.[pP][nN][gG]"))
    unique = {str(p.resolve()).lower(): p for p in hits}
    return sorted(unique.values(), key=lambda p: p.name.lower())


def caption(p: Path) -> str:
    """Format a file path stem as a title (underscores to spaces)."""
    return p.stem.replace("_", " ").title()


def filter_by_patterns(paths: list[Path], patterns: list[str]) -> list[Path]:
    """Filter a list of files by a set of substrings (case-insensitive)."""
    if not patterns:
        return paths
    pats = [s.lower() for s in patterns]
    out = []
    for p in paths:
        stem = p.stem.lower()
        if any(s in stem for s in pats):
            out.append(p)
    return out


def render_app() -> None:
    """Render the Streamlit dashboard UI for plots and map building."""
    # Set page config at the start to avoid Streamlit config errors
    st.set_page_config(page_title="Model Plots Dashboard", layout="centered")

    st.title("Model Analysis Plots Dashboard")
    st.write(
        "Feature plots plus curated images from `feature_plots/` and `plots/`, "
        "and a predictions map."
    )

    # --- Feature plots
    feat_imgs = imgs_in(FEATURE_DIR)
    all_plots = imgs_in(PLOTS_DIR)

    with st.expander("Feature Plots (Distributions & Boxplots)", expanded=False):
        if feat_imgs:
            for p in feat_imgs:
                st.image(str(p), caption=caption(p), use_column_width=True)
        else:
            st.write("_No images found in `feature_plots`._")

    with st.expander(
        "Other Analysis Plots (Feature Importance & Correlation)", expanded=False
    ):
        section_imgs = filter_by_patterns(all_plots, PATTERNS["summary"])
        if section_imgs:
            for p in section_imgs:
                st.image(str(p), caption=caption(p), use_column_width=True)
        else:
            st.write("_No matching images in `plots` for this section._")

    with st.expander("Other Analysis Plots (1D PDP)", expanded=False):
        section_imgs = filter_by_patterns(all_plots, PATTERNS["pdp_1d"])
        if section_imgs:
            for p in section_imgs:
                st.image(str(p), caption=caption(p), use_column_width=True)
        else:
            st.write("_No matching 1D PDP images in `plots`._")

    with st.expander("Other Analysis Plots (2D PDP)", expanded=False):
        section_imgs = filter_by_patterns(all_plots, PATTERNS["pdp_2d"])
        if section_imgs:
            for p in section_imgs:
                st.image(str(p), caption=caption(p), use_column_width=True)
        else:
            st.write("_No matching 2D PDP images in `plots`._")

    st.markdown("---")
    st.header("Catchment Predictions Map")

    geo_path = st.text_input(
        "Geometry path (.zip/.shp/.geojson)", str(DEFAULT_GEO_PATH)
    )
    csv_path = st.text_input("Predictions CSV path", str(DEFAULT_CSV_PATH))
    pred_col = st.text_input("Prediction column", DEFAULT_PRED_COL)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Build / Refresh Map", type="primary"):
            log_lines = []

            def slog(msg: str) -> None:
                log_lines.append(str(msg))

            try:
                merged, fmap = build_and_save_map(
                    Path(geo_path),
                    Path(csv_path),
                    pred_col,
                    OUT_MERGED_GEO,
                    OUT_MAP_HTML,
                    logger=slog,
                )
                st.success(
                    f"Join success (non-null): "
                    f"{merged[pred_col].notna().sum()}/{len(merged)}"
                )
                # embed map
                components.html(fmap.get_root().render(), height=720, scrolling=False)
                with st.expander("Build log"):
                    for ln in log_lines:
                        st.text(ln)
                st.caption(
                    f"Saved: {OUT_MERGED_GEO.name} and {OUT_MAP_HTML.name} in {OUT_DIR}"
                )
            except Exception as e:
                st.error(f"Build failed: {e}")

    with col2:
        st.write("Output folder:", str(OUT_DIR))
        st.write("Merged GeoJSON:", str(OUT_MERGED_GEO))
        st.write("HTML Map:", str(OUT_MAP_HTML))


# -------------------------- CLI MAIN --------------------------
# Command-line interface: parse arguments and run map generation without UI
def run_cli(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the map generation (no Streamlit UI)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--geo", default=str(DEFAULT_GEO_PATH))
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--pred-col", dest="pred_col", default=DEFAULT_PRED_COL)
    parser.add_argument("--out-geo", dest="out_geo", default=str(OUT_MERGED_GEO))
    parser.add_argument("--out-html", dest="out_html", default=str(OUT_MAP_HTML))
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    def plog(msg: str) -> None:
        print(msg)

    merged, _ = build_and_save_map(
        Path(args.geo),
        Path(args.csv),
        args.pred_col,
        Path(args.out_geo),
        Path(args.out_html),
        logger=plog,
    )
    print(
        f"Join success (non-null): {merged[args.pred_col].notna().sum()}/{len(merged)}"
    )
    return None


# -------------------------- ENTRY POINT --------------------------
# Determine mode (CLI vs Streamlit) and execute accordingly
if __name__ == "__main__":
    # If "--cli" flag is present, run CLI mode and exit
    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        run_cli(sys.argv[1:])
        sys.exit(0)

    # Not launched from Streamlit: start a new Streamlit process
    if os.environ.get("STREAMLIT_LAUNCHED_FROM_SCRIPT") != "1":
        env = os.environ.copy()
        env["STREAMLIT_LAUNCHED_FROM_SCRIPT"] = "1"
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            __file__,
            "--server.fileWatcherType=none",
            "--browser.gatherUsageStats=false",
        ]
        subprocess.Popen(cmd, env=env, close_fds=True)
        raise SystemExit(0)

    # Running inside Streamlit (launched from script): render app UI
    render_app()
