#!/usr/bin/env python3
"""
Command-line demo for the MEE344 Electricity Consumption Forecasting project.

This script mirrors the main notebook's forecasting setup:
- target: hourly consumption at time t
- features: calendar features, lagged consumption, lagged production, rolling historical statistics
- no same-time consumption or same-time production features are used

Example usage from the repository root:
    python src/app.py compare
    python src/app.py predict "2025-08-20 14:00"
    python src/app.py predict "2025-08-20 14:00" --model decision_tree
    python src/app.py figures
    python src/app.py open-figure final_model_rmse_comparison
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - only triggered on missing dependency
    raise SystemExit(
        "xgboost is not installed. Install dependencies with: pip install -r requirements.txt"
    ) from exc


PRODUCTION_FILENAME = "Gercek_Zamanli_Uretim-01062025-01092025(in).csv"
CONSUMPTION_FILENAME = "Gercek_Zamanli_Tuketim-01062025-01092025(in).csv"
ZERO_SOURCE_COLUMNS = {"Nafta", "LNG"}


@dataclass(frozen=True)
class DatasetBundle:
    df_model: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]
    split_index: int


@dataclass(frozen=True)
class ModelResult:
    name: str
    rmse: float
    mae: float
    r2: float


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from cwd or from this file location."""
    candidates: list[Path] = []
    if start is not None:
        candidates.append(start.resolve())
    candidates.extend([Path.cwd().resolve(), Path(__file__).resolve().parent])

    for base in list(candidates):
        candidates.extend(base.parents)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "data" / "raw").exists() or (candidate / "notebooks").exists():
            return candidate

    return Path.cwd().resolve()


def find_data_file(project_root: Path, filename: str) -> Path:
    """Locate a raw data file using repository paths and notebook-check fallbacks."""
    candidates = [
        project_root / "data" / "raw" / filename,
        project_root / filename,
        Path.cwd() / filename,
        Path.cwd() / "data" / "raw" / filename,
        Path("/mnt/data") / filename,  # useful when tested outside the repo
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Checked:\n{checked}")


def parse_datetime_columns(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        df["Tarih"].astype(str) + " " + df["Saat"].astype(str),
        format="%d.%m.%Y %H:%M",
        errors="coerce",
    )


def load_data(project_root: Path) -> pd.DataFrame:
    """Load, parse, merge, and clean the production/consumption datasets."""
    production_path = find_data_file(project_root, PRODUCTION_FILENAME)
    consumption_path = find_data_file(project_root, CONSUMPTION_FILENAME)

    production = pd.read_csv(
        production_path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )
    production["datetime"] = parse_datetime_columns(production)
    production = production.sort_values("datetime").reset_index(drop=True)

    consumption_raw = pd.read_csv(
        consumption_path,
        sep=";",
        encoding="cp1254",
        dtype=str,
    )
    consumption_col = [col for col in consumption_raw.columns if col not in ["Tarih", "Saat"]][0]
    consumption = consumption_raw.rename(columns={consumption_col: "Tuketim"}).copy()
    consumption["Tuketim"] = (
        consumption["Tuketim"]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    consumption["datetime"] = parse_datetime_columns(consumption)
    consumption = consumption.sort_values("datetime").reset_index(drop=True)

    df = pd.merge(
        production,
        consumption[["datetime", "Tuketim"]],
        on="datetime",
        how="inner",
    )
    df = df.sort_values("datetime").reset_index(drop=True)

    if df["datetime"].isna().any():
        raise ValueError("At least one timestamp could not be parsed.")
    if df["datetime"].duplicated().any():
        raise ValueError("Duplicate timestamps found after merging data files.")

    return df


def build_features(df: pd.DataFrame) -> DatasetBundle:
    """Reproduce the notebook's leakage-safe feature matrix."""
    df_fe = df.copy()
    df_fe["target"] = df_fe["Tuketim"]

    # Calendar features
    df_fe["hour"] = df_fe["datetime"].dt.hour
    df_fe["dayofweek"] = df_fe["datetime"].dt.dayofweek
    df_fe["day"] = df_fe["datetime"].dt.day
    df_fe["month"] = df_fe["datetime"].dt.month
    df_fe["is_weekend"] = df_fe["dayofweek"].isin([5, 6]).astype(int)
    df_fe["hour_sin"] = np.sin(2 * np.pi * df_fe["hour"] / 24)
    df_fe["hour_cos"] = np.cos(2 * np.pi * df_fe["hour"] / 24)
    df_fe["dow_sin"] = np.sin(2 * np.pi * df_fe["dayofweek"] / 7)
    df_fe["dow_cos"] = np.cos(2 * np.pi * df_fe["dayofweek"] / 7)

    calendar_features = [
        "hour",
        "dayofweek",
        "day",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    # Historical consumption features
    consumption_lags = [1, 2, 3, 24, 48, 168]
    for lag in consumption_lags:
        df_fe[f"Tuketim_lag_{lag}"] = df_fe["Tuketim"].shift(lag)
    consumption_lag_features = [f"Tuketim_lag_{lag}" for lag in consumption_lags]

    rolling_windows = [3, 6, 24]
    consumption_rolling_features: list[str] = []
    for window in rolling_windows:
        mean_name = f"Tuketim_roll_{window}_mean"
        std_name = f"Tuketim_roll_{window}_std"
        df_fe[mean_name] = df_fe["Tuketim"].shift(1).rolling(window).mean()
        df_fe[std_name] = df_fe["Tuketim"].shift(1).rolling(window).std()
        consumption_rolling_features.extend([mean_name, std_name])

    df_fe["Tuketim_diff_1"] = df_fe["Tuketim_lag_1"] - df_fe["Tuketim_lag_2"]
    df_fe["Tuketim_diff_24"] = df_fe["Tuketim_lag_1"] - df_fe["Tuketim_lag_24"]
    trend_features = ["Tuketim_diff_1", "Tuketim_diff_24"]

    # Historical production features
    production_total_lags = [1, 2, 3, 24, 48, 168]
    for lag in production_total_lags:
        df_fe[f"Toplam_lag_{lag}"] = df_fe["Toplam"].shift(lag)
    production_total_lag_features = [f"Toplam_lag_{lag}" for lag in production_total_lags]

    source_cols = [
        col
        for col in df.columns
        if col not in ["Tarih", "Saat", "datetime", "Toplam", "Tuketim"]
        and col not in ZERO_SOURCE_COLUMNS
    ]
    source_lag_features: list[str] = []
    for col in source_cols:
        feature_name = f"{col}_lag_1"
        df_fe[feature_name] = df_fe[col].shift(1)
        source_lag_features.append(feature_name)

    feature_names = (
        calendar_features
        + consumption_lag_features
        + consumption_rolling_features
        + trend_features
        + production_total_lag_features
        + source_lag_features
    )

    required_cols = ["target"] + feature_names
    df_model = df_fe.dropna(subset=required_cols).copy()
    X = df_model[feature_names].copy()
    y = df_model["target"].copy()

    # Leakage checks: no current-hour consumption or production columns are model inputs.
    forbidden_cols = ["Tuketim", "Toplam"] + source_cols
    leaked_cols = [col for col in forbidden_cols if col in X.columns]
    if leaked_cols:
        raise AssertionError(f"Leakage columns found in feature matrix: {leaked_cols}")

    split_index = int(len(df_model) * 0.85)
    return DatasetBundle(df_model=df_model, X=X, y=y, feature_names=feature_names, split_index=split_index)


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> ModelResult:
    pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))
    return ModelResult(name=name, rmse=rmse, mae=mae, r2=r2)


def train_models(bundle: DatasetBundle) -> dict[str, object]:
    """Train the same baseline/tuned models used in the notebook."""
    X_train = bundle.X.iloc[: bundle.split_index]
    y_train = bundle.y.iloc[: bundle.split_index]

    models: dict[str, object] = {
        "decision_tree_baseline": DecisionTreeRegressor(random_state=42),
        "decision_tree": DecisionTreeRegressor(
            random_state=42,
            max_depth=10,
            max_features=None,
            min_samples_leaf=5,
            min_samples_split=2,
        ),
        "xgboost_baseline": XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
        ),
        "xgboost": XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=5.0,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
        ),
    }

    for model in models.values():
        model.fit(X_train, y_train)
    return models


def print_results_table(results: Iterable[ModelResult]) -> None:
    rows = list(results)
    if not rows:
        print("No results.")
        return

    name_width = max(len("Model"), *(len(row.name) for row in rows))
    print(f"{'Model':<{name_width}}  {'RMSE':>10}  {'MAE':>10}  {'R2':>8}")
    print(f"{'-' * name_width}  {'-' * 10}  {'-' * 10}  {'-' * 8}")
    for row in rows:
        print(f"{row.name:<{name_width}}  {row.rmse:10.3f}  {row.mae:10.3f}  {row.r2:8.4f}")


def command_compare(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.root) if args.root else None)
    df = load_data(root)
    bundle = build_features(df)
    models = train_models(bundle)

    X_test = bundle.X.iloc[bundle.split_index :]
    y_test = bundle.y.iloc[bundle.split_index :]

    results = [
        evaluate_model("Decision Tree Baseline", models["decision_tree_baseline"], X_test, y_test),
        evaluate_model("Decision Tree Tuned", models["decision_tree"], X_test, y_test),
        evaluate_model("XGBoost Baseline", models["xgboost_baseline"], X_test, y_test),
        evaluate_model("XGBoost Tuned", models["xgboost"], X_test, y_test),
    ]

    print("Final chronological test-period comparison")
    print(f"Rows after feature engineering: {len(bundle.df_model)}")
    print(f"Train/validation rows: {bundle.split_index}")
    print(f"Test rows: {len(bundle.df_model) - bundle.split_index}")
    print()
    print_results_table(results)


def parse_user_datetime(text: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(text, errors="coerce")
    if pd.isna(timestamp):
        raise argparse.ArgumentTypeError(
            f"Could not parse datetime: {text!r}. Try a format such as '2025-08-20 14:00'."
        )
    return pd.Timestamp(timestamp).floor("h")


def command_predict(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.root) if args.root else None)
    df = load_data(root)
    bundle = build_features(df)
    models = train_models(bundle)

    timestamp = parse_user_datetime(args.datetime)
    matches = bundle.df_model.index[bundle.df_model["datetime"] == timestamp].tolist()
    if not matches:
        min_dt = bundle.df_model["datetime"].min()
        max_dt = bundle.df_model["datetime"].max()
        raise SystemExit(
            f"Timestamp {timestamp} is not predictable from the available feature matrix.\n"
            f"Available predictable range: {min_dt} to {max_dt}.\n"
            "The first 168 hours are unavailable because weekly lag features are required."
        )

    row_label = matches[0]
    row_pos = bundle.df_model.index.get_loc(row_label)
    feature_row = bundle.X.loc[[row_label]]
    actual = float(bundle.df_model.loc[row_label, "target"])

    model_key = args.model
    model = models[model_key]
    predicted = float(model.predict(feature_row)[0])
    absolute_error = abs(predicted - actual)
    percent_error = absolute_error / actual * 100 if actual else float("nan")

    period = "final test period" if row_pos >= bundle.split_index else "train/validation period"
    model_label = "Tuned XGBoost" if model_key == "xgboost" else "Tuned Decision Tree"

    print(f"Timestamp: {timestamp}")
    print(f"Model: {model_label}")
    print(f"Period: {period}")
    print(f"Predicted consumption: {predicted:,.2f} MWh")
    print(f"Actual consumption:    {actual:,.2f} MWh")
    print(f"Absolute error:        {absolute_error:,.2f} MWh")
    print(f"Percent error:         {percent_error:.2f}%")


def get_figures_dir(root: Path) -> Path:
    return root / "figures" / "results"


def list_figure_files(root: Path) -> list[Path]:
    figures_dir = get_figures_dir(root)
    if not figures_dir.exists():
        return []
    return sorted(
        [path for path in figures_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".csv"}]
    )


def command_figures(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.root) if args.root else None)
    files = list_figure_files(root)
    if not files:
        print(f"No figure/result files found in {get_figures_dir(root)}")
        return

    print(f"Files in {get_figures_dir(root)}:")
    for path in files:
        print(f"  {path.name}")


def open_file(path: Path) -> None:
    system = platform.system().lower()
    if system == "windows":
        os.startfile(path)  # type: ignore[attr-defined]
    elif system == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def resolve_figure(root: Path, query: str) -> Path:
    files = list_figure_files(root)
    if not files:
        raise FileNotFoundError(f"No figure/result files found in {get_figures_dir(root)}")

    query_path = Path(query)
    candidates = []
    if query_path.suffix:
        candidates.append(query_path.name)
    else:
        candidates.extend([f"{query}.png", f"{query}.jpg", f"{query}.jpeg", f"{query}.csv"])

    by_name = {path.name: path for path in files}
    for candidate in candidates:
        if candidate in by_name:
            return by_name[candidate]

    lowered = query.lower()
    fuzzy = [path for path in files if lowered in path.stem.lower() or lowered in path.name.lower()]
    if len(fuzzy) == 1:
        return fuzzy[0]
    if len(fuzzy) > 1:
        options = "\n".join(f"  - {path.name}" for path in fuzzy)
        raise SystemExit(f"Multiple matching files. Be more specific:\n{options}")

    options = "\n".join(f"  - {path.name}" for path in files)
    raise FileNotFoundError(f"No file matched {query!r}. Available files:\n{options}")


def command_open_figure(args: argparse.Namespace) -> None:
    root = find_project_root(Path(args.root) if args.root else None)
    path = resolve_figure(root, args.name)
    print(f"Opening: {path}")
    open_file(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI demo for the MEE344 electricity consumption forecasting project.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Optional repository root. Usually not needed when running from the project folder.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser("compare", help="Train the notebook models and print test metrics.")
    compare_parser.set_defaults(func=command_compare)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict consumption for a timestamp inside the available dataset.",
    )
    predict_parser.add_argument("datetime", help="Timestamp, e.g. '2025-08-20 14:00'.")
    predict_parser.add_argument(
        "--model",
        choices=["xgboost", "decision_tree"],
        default="xgboost",
        help="Tuned model to use for the prediction. Default: xgboost.",
    )
    predict_parser.set_defaults(func=command_predict)

    figures_parser = subparsers.add_parser("figures", help="List saved result figures/files.")
    figures_parser.set_defaults(func=command_figures)

    open_parser = subparsers.add_parser("open-figure", help="Open a saved figure/result file by name.")
    open_parser.add_argument("name", help="File stem or filename, e.g. 'xgboost_residuals'.")
    open_parser.set_defaults(func=command_open_figure)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
