#!/usr/bin/env python3
"""Train and recursively forecast weekly sales per warehouse/article.

The model is trained directly on weekly targets; it does NOT sum predictions from a
separate daily model. Historical daily sales are aggregated into Monday-Sunday weeks.
The target is trained on log1p(weekly_quantity), then converted back with expm1.

Default input candidates:
  data/raw/sales_dataset_hakaton_train.csv
  data/sales_dataset_hakaton_train.csv
  sales_dataset_hakaton_train.csv

Run from the repository root:
  python training/forecast_weekly.py
or:
  python training/forecast_weekly.py --sales data/raw/sales_dataset_hakaton_train.csv --horizon-weeks 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor
except ImportError as exc:  # pragma: no cover - depends on environment
    raise SystemExit(
        "CatBoost is not installed. Activate the project's Python environment and run: "
        "pip install -r requirements.txt"
    ) from exc

LOG = logging.getLogger("weekly_forecast")
KEYS = ["warehouse_name", "article"]
LAGS = (1, 2, 3, 4, 8, 13, 26, 52)
ROLLING_WINDOWS = (4, 8, 13)
STATIC_CANDIDATES = ("category_lvl_1", "category_lvl_2", "characteristic")
DATE_ALIASES = ("date", "sale_date", "datetime", "day", "dt", "Дата")
QTY_ALIASES = ("quantity", "qty", "sales_qty", "units_sold", "sales", "Количество")
WAREHOUSE_ALIASES = ("warehouse_name", "warehouse", "warehouse_id", "store_id", "склад")
ARTICLE_ALIASES = ("article", "article_id", "sku", "item_id", "артикул")


def _norm_name(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _find_column(columns: Iterable[object], aliases: Iterable[str], override: str | None, logical_name: str) -> str:
    cols = list(columns)
    if override:
        if override not in cols:
            raise ValueError(f"Column for {logical_name!r} was set to {override!r}, but it is not in the CSV. Columns: {cols}")
        return override
    by_norm = {_norm_name(c): c for c in cols}
    for alias in aliases:
        if alias in cols:
            return alias
        found = by_norm.get(_norm_name(alias))
        if found is not None:
            return found
    raise ValueError(
        f"Could not identify the {logical_name} column. Available columns: {cols}. "
        f"Pass it explicitly with the relevant --*-col argument."
    )


def _choose_sales_path(explicit: str | None, root: Path) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f"Sales file not found: {path}")
    candidates = [
        root / "data" / "raw" / "sales_dataset_hakaton_train.csv",
        root / "data" / "sales_dataset_hakaton_train.csv",
        root / "sales_dataset_hakaton_train.csv",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    pretty = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not find sales CSV. Checked:\n" + pretty +
        "\nPass the correct location with --sales PATH."
    )


def _read_csv(path: Path) -> pd.DataFrame:
    # Sniff delimiter first so both comma- and semicolon-separated exports work.
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception as exc:
        raise RuntimeError(f"Unable to read sales CSV {path}: {exc}") from exc


def _mode_or_first(series: pd.Series) -> str:
    values = series.dropna().astype(str).str.strip()
    values = values[values.ne("")]
    if values.empty:
        return "__MISSING__"
    modes = values.mode()
    return str(modes.iloc[0] if not modes.empty else values.iloc[0])


def load_and_aggregate(path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    LOG.info("Reading %s", path)
    raw = _read_csv(path)
    if raw.empty:
        raise ValueError("Sales CSV is empty.")

    date_col = _find_column(raw.columns, DATE_ALIASES, args.date_col, "date")
    qty_col = _find_column(raw.columns, QTY_ALIASES, args.quantity_col, "quantity")
    warehouse_col = _find_column(raw.columns, WAREHOUSE_ALIASES, args.warehouse_col, "warehouse")
    article_col = _find_column(raw.columns, ARTICLE_ALIASES, args.article_col, "article")

    rename = {date_col: "date", qty_col: "quantity", warehouse_col: "warehouse_name", article_col: "article"}
    raw = raw.rename(columns=rename)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce", dayfirst=args.dayfirst)
    raw["quantity"] = pd.to_numeric(raw["quantity"], errors="coerce")
    raw = raw.dropna(subset=["date", "quantity", *KEYS]).copy()
    raw["warehouse_name"] = raw["warehouse_name"].astype(str).str.strip()
    raw["article"] = raw["article"].astype(str).str.strip()
    raw = raw[(raw["warehouse_name"] != "") & (raw["article"] != "")]
    if raw.empty:
        raise ValueError("No valid rows remained after parsing dates, quantities and keys.")

    # Keep only stable categorical attributes available from sales data.
    static_cols = [c for c in STATIC_CANDIDATES if c in raw.columns]
    for col in static_cols:
        raw[col] = raw[col].replace(r"^\s*$", np.nan, regex=True)
    if static_cols:
        metadata = raw.groupby(KEYS, sort=False)[static_cols].agg(_mode_or_first).reset_index()
    else:
        metadata = raw[KEYS].drop_duplicates().copy()

    # Calendar weeks are Monday through Sunday. A weekly target is the sum of actual
    # daily quantities within the week, not a sum of already-generated daily forecasts.
    raw["week_start"] = raw["date"].dt.to_period("W-SUN").dt.start_time
    weekly = raw.groupby(KEYS + ["week_start"], as_index=False, sort=False)["quantity"].sum()

    if (weekly["quantity"] < 0).any():
        n_negative = int((weekly["quantity"] < 0).sum())
        raise ValueError(
            f"Found {n_negative} negative weekly targets. log1p requires non-negative targets. "
            "Confirm how returns/corrections should be handled before training; the script will not silently clip them."
        )

    latest_observed_date = raw["date"].max().normalize()
    last_week = weekly["week_start"].max()
    # Mark the trailing calendar week as incomplete when the source ends before Sunday.
    # It can be used as recent history for recursion, but not as a supervised weekly target.
    last_week_end = pd.Timestamp(last_week) + pd.Timedelta(days=6)
    trailing_week_is_partial = latest_observed_date < last_week_end
    observed_days_in_trailing_week = int((latest_observed_date - pd.Timestamp(last_week)).days + 1)
    first_by_pair = weekly.groupby(KEYS, sort=False)["week_start"].min().reset_index(name="first_week")
    # Build a complete weekly history for each pair from its first observed week to the
    # last observed week. A missing sales row inside that active interval is treated as 0.
    parts: list[pd.DataFrame] = []
    for row in first_by_pair.itertuples(index=False, name=None):
        warehouse, article, first_week = row
        dates = pd.date_range(pd.Timestamp(first_week), pd.Timestamp(last_week), freq="7D")
        parts.append(pd.DataFrame({"warehouse_name": warehouse, "article": article, "week_start": dates}))
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.merge(weekly, on=KEYS + ["week_start"], how="left", validate="one_to_one")
    panel["quantity"] = panel["quantity"].fillna(0.0).astype(float)
    panel["week_complete"] = panel["week_start"] + pd.Timedelta(days=6) <= latest_observed_date
    if trailing_week_is_partial:
        # Use a simple run-rate estimate for lag features only. The incomplete week is
        # explicitly excluded from all target training and holdout metrics below.
        partial_mask = panel["week_start"].eq(pd.Timestamp(last_week))
        panel.loc[partial_mask, "quantity"] *= 7.0 / max(observed_days_in_trailing_week, 1)
    panel = panel.merge(metadata, on=KEYS, how="left", validate="many_to_one")
    for col in static_cols:
        panel[col] = panel[col].fillna("__MISSING__").astype(str)
    panel = panel.sort_values(KEYS + ["week_start"]).reset_index(drop=True)

    LOG.info("Rows in source: %s", f"{len(raw):,}")
    LOG.info("Daily rows aggregated into weekly target rows: %s", f"{len(weekly):,}")
    LOG.info("Complete weekly panel rows: %s; pairs: %s; weeks: %s", f"{len(panel):,}", panel[KEYS].drop_duplicates().shape[0], panel["week_start"].nunique())
    return panel, static_cols


def make_features(panel: pd.DataFrame, static_cols: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = panel.sort_values(KEYS + ["week_start"]).reset_index(drop=True).copy()
    grouped = frame.groupby(KEYS, sort=False)["quantity"]

    iso = frame["week_start"].dt.isocalendar()
    frame["week_of_year"] = iso.week.astype(int)
    frame["month"] = frame["week_start"].dt.month.astype(int)
    frame["year"] = frame["week_start"].dt.year.astype(int)
    frame["week_sin"] = np.sin(2 * np.pi * frame["week_of_year"] / 52.18)
    frame["week_cos"] = np.cos(2 * np.pi * frame["week_of_year"] / 52.18)

    for lag in LAGS:
        frame[f"qty_lag_{lag}"] = grouped.shift(lag)
    for window in ROLLING_WINDOWS:
        # shift before rolling to avoid using the current week's target in its own features
        frame[f"qty_roll_mean_{window}"] = grouped.transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
        frame[f"qty_roll_max_{window}"] = grouped.transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).max())
        frame[f"qty_roll_zero_share_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(1).eq(0).rolling(w, min_periods=1).mean()
        )

    cat_cols = KEYS + static_cols
    num_cols = [
        "week_of_year", "month", "year", "week_sin", "week_cos",
        *[f"qty_lag_{lag}" for lag in LAGS],
        *[f"qty_roll_mean_{w}" for w in ROLLING_WINDOWS],
        *[f"qty_roll_max_{w}" for w in ROLLING_WINDOWS],
        *[f"qty_roll_zero_share_{w}" for w in ROLLING_WINDOWS],
    ]
    feature_cols = cat_cols + num_cols
    for col in cat_cols:
        frame[col] = frame[col].fillna("__MISSING__").astype(str)
    frame[num_cols] = frame[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame, feature_cols, cat_cols


def score_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = float(np.sum(np.abs(actual)))
    actual_sum = float(np.sum(actual))
    return {
        "n_rows": int(len(actual)),
        "mae": float(np.mean(np.abs(actual - predicted))) if len(actual) else None,
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))) if len(actual) else None,
        "wape": float(np.sum(np.abs(actual - predicted)) / denom) if denom > 0 else None,
        "actual_total": actual_sum,
        "predicted_total": float(np.sum(predicted)),
        "total_bias_pct": float((np.sum(predicted) - actual_sum) / actual_sum) if actual_sum != 0 else None,
    }


def _build_model(iterations: int, thread_count: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=100,
        thread_count=thread_count,
        allow_writing_files=False,
    )


def run(args: argparse.Namespace) -> Path:
    root = Path(args.project_root).expanduser().resolve() if args.project_root else Path.cwd().resolve()
    sales_path = _choose_sales_path(args.sales, root)
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    panel, static_cols = load_and_aggregate(sales_path, args)
    feat, feature_cols, cat_cols = make_features(panel, static_cols)
    # Only complete weeks may be supervised targets or enter temporal holdout metrics.
    weeks = sorted(pd.Timestamp(w) for w in feat.loc[feat["week_complete"], "week_start"].drop_duplicates())
    if len(weeks) < 20:
        raise ValueError(f"Only {len(weeks)} weekly periods found. At least 20 are needed for train/validation/test splitting.")

    # Chronological holdout: last 8 weeks test; previous 8 weeks validation.
    validation_weeks = set(weeks[-16:-8])
    test_weeks = set(weeks[-8:])
    train_mask = feat["week_complete"] & ~feat["week_start"].isin(validation_weeks | test_weeks) & feat["quantity"].notna()
    val_mask = feat["week_complete"] & feat["week_start"].isin(validation_weeks) & feat["quantity"].notna()
    test_mask = feat["week_complete"] & feat["week_start"].isin(test_weeks) & feat["quantity"].notna()
    train, val, test = feat.loc[train_mask], feat.loc[val_mask], feat.loc[test_mask]
    if train.empty or val.empty or test.empty:
        raise ValueError("Temporal split produced an empty train, validation or test set.")

    LOG.info("Split rows: train=%s, validation=%s, test=%s", f"{len(train):,}", f"{len(val):,}", f"{len(test):,}")
    model = _build_model(args.iterations, args.thread_count)
    model.fit(
        train[feature_cols], np.log1p(train["quantity"].to_numpy()),
        cat_features=cat_cols,
        eval_set=(val[feature_cols], np.log1p(val["quantity"].to_numpy())),
        early_stopping_rounds=80,
        use_best_model=True,
    )

    holdout_log = model.predict(test[feature_cols])
    holdout_pred = np.maximum(0.0, np.expm1(holdout_log))
    holdout = test[KEYS + ["week_start", "quantity"]].copy()
    holdout = holdout.rename(columns={"quantity": "actual_weekly_quantity"})
    holdout["predicted_weekly_quantity"] = holdout_pred
    holdout.to_csv(out_dir / "weekly_holdout_predictions.csv", index=False)
    metrics = {
        "sales_source": str(sales_path),
        "target": "weekly sum of daily quantity per warehouse/article",
        "target_transform": "log1p; predictions back-transformed with expm1 and clipped at zero",
        "week_definition": "Monday-Sunday; week_start is Monday",
        "trailing_week_incomplete": bool(not bool(panel.loc[panel["week_start"].eq(panel["week_start"].max()), "week_complete"].all())),
        "incomplete_trailing_week_used_for_training": False,
        "n_pairs": int(panel[KEYS].drop_duplicates().shape[0]),
        "n_historical_weeks": int(len(weeks)),
        "train_week_end": str(max(set(weeks) - validation_weeks - test_weeks)),
        "validation_weeks": [str(w.date()) for w in sorted(validation_weeks)],
        "test_weeks": [str(w.date()) for w in sorted(test_weeks)],
        "best_iteration": int(model.get_best_iteration() if model.get_best_iteration() is not None and model.get_best_iteration() >= 0 else args.iterations),
        "test_metrics": score_predictions(test["quantity"].to_numpy(), holdout_pred),
        "feature_columns": feature_cols,
    }
    (out_dir / "weekly_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Test metrics: %s", json.dumps(metrics["test_metrics"], ensure_ascii=False))

    # Refit a final model on all available historical weeks using the selected iteration count.
    best_iters = int(metrics["best_iteration"])
    final_model = _build_model(max(100, best_iters), args.thread_count)
    final_rows = feat.loc[feat["week_complete"] & feat["quantity"].notna()]
    final_model.fit(final_rows[feature_cols], np.log1p(final_rows["quantity"].to_numpy()), cat_features=cat_cols)
    final_model.save_model(str(out_dir / "catboost_weekly.cbm"))

    # Direct weekly recursive forecasts: predict week t+1, append that prediction as the
    # lag for t+2, and repeat. We never create daily predictions to sum afterwards.
    history = panel.copy().sort_values(KEYS + ["week_start"]).reset_index(drop=True)
    pair_meta = history[KEYS + static_cols].drop_duplicates(KEYS).reset_index(drop=True)
    last_week = pd.Timestamp(history["week_start"].max())
    forecast_frames: list[pd.DataFrame] = []
    for step in range(1, args.horizon_weeks + 1):
        future_week = last_week + pd.Timedelta(days=7 * step)
        future = pair_meta.copy()
        future["week_start"] = future_week
        future["quantity"] = np.nan
        future["week_complete"] = True
        history = pd.concat([history, future], ignore_index=True, sort=False)
        history = history.sort_values(KEYS + ["week_start"]).reset_index(drop=True)
        future_mask = history["week_start"].eq(future_week)
        recursive_feat, _, _ = make_features(history, static_cols)
        x_future = recursive_feat.loc[future_mask, feature_cols]
        pred_log = final_model.predict(x_future)
        pred = np.maximum(0.0, np.expm1(pred_log))
        history.loc[future_mask, "quantity"] = pred
        out = history.loc[future_mask, KEYS + ["week_start"] + static_cols].copy()
        out["forecast_week"] = step
        out["predicted_weekly_quantity"] = pred
        forecast_frames.append(out)
        LOG.info("Forecasted week %s/%s (%s) for %s warehouse/article pairs", step, args.horizon_weeks, future_week.date(), len(out))

    forecast = pd.concat(forecast_frames, ignore_index=True)
    forecast = forecast.sort_values(["week_start", "warehouse_name", "article"]).reset_index(drop=True)
    forecast.to_csv(out_dir / "weekly_forecast.csv", index=False)
    summary = forecast.groupby("week_start", as_index=False).agg(
        forecast_total=("predicted_weekly_quantity", "sum"),
        pair_count=("article", "size"),
    )
    summary.to_csv(out_dir / "weekly_forecast_summary.csv", index=False)
    LOG.info("Saved weekly forecast: %s", out_dir / "weekly_forecast.csv")
    LOG.info("Saved weekly summary: %s", out_dir / "weekly_forecast_summary.csv")
    LOG.info("Output directory: %s", out_dir)
    return out_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sales", help="Path to sales CSV. Defaults to a set of conventional paths under project root.")
    parser.add_argument("--project-root", help="Repository root (defaults to current working directory).")
    parser.add_argument("--output-dir", default="data/artifacts/forecast/weekly", help="Forecast output directory, relative to project root unless absolute.")
    parser.add_argument("--horizon-weeks", type=int, default=10, help="Number of future weekly buckets to forecast (default: 10).")
    parser.add_argument("--iterations", type=int, default=600, help="Maximum CatBoost iterations before early stopping (default: 600).")
    parser.add_argument("--thread-count", type=int, default=-1, help="CatBoost threads; -1 uses all available cores.")
    parser.add_argument("--date-col", help="Override detected date column name.")
    parser.add_argument("--quantity-col", help="Override detected quantity column name.")
    parser.add_argument("--warehouse-col", help="Override detected warehouse column name.")
    parser.add_argument("--article-col", help="Override detected article column name.")
    parser.add_argument("--dayfirst", action="store_true", help="Parse ambiguous dates as day/month/year.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args(argv)
    if args.horizon_weeks < 1:
        print("--horizon-weeks must be >= 1", file=sys.stderr)
        return 2
    try:
        out_dir = run(args)
    except Exception as exc:
        LOG.exception("Weekly forecast failed: %s", exc)
        return 1
    print(f"Weekly forecast artifacts written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
