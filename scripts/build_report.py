"""Собрать автономный HTML-отчёт из сохранённых результатов и выборок FE.

Запуск из корня: python scripts/build_report.py
Модели не переобучаются. Распределения считаются по всем строкам трёх выборок;
в HTML попадают только агрегаты и графики, без исходных строк Parquet/CSV.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data/artifacts"
SPLITS = ("train", "valid", "test")


def records(relative: str) -> list[dict]:
    """Пропуски CSV превращаем в JSON null, даты сохраняем строками."""
    return json.loads(pd.read_csv(ART / relative).to_json(orient="records", double_precision=9))


def feature_descriptions() -> dict:
    result = {}
    for line in (ROOT / "features/FEATURES.md").read_text().splitlines():
        match = re.match(r"\| `([^`]+)` \| (.*?) \| (.*?) \|", line)
        if match:
            name, meaning, available = match.groups()
            result[name] = {"meaning": meaning, "available": available}
    return result


def distribution_payload(manifest: dict, labels: dict, groups: dict) -> list[dict]:
    columns = ["quantity", *manifest["categorical"], *manifest["crosses"], *manifest["numeric"]]
    frames = {s: pd.read_parquet(ROOT / f"data/processed/{s}.parquet", columns=columns) for s in SPLITS}
    descriptions = feature_descriptions()
    membership = {c: g for g, cols in manifest["groups"].items() for c in cols}
    membership.update({c: "identity" for c in manifest["categorical"]})
    membership.update({c: "crosses" for c in manifest["crosses"]})
    result = []
    for name in columns:
        categorical = name in manifest["categorical"] + manifest["crosses"]
        group = membership.get(name, "target")
        entry = {
            "name": name, "label": labels.get(name, name),
            "group": groups.get(group, "Цель: не подаётся в модель"),
            "kind": "category" if categorical else "numeric",
            **descriptions.get(name, {"meaning": "Число проданных единиц на пару склад × артикул за день.",
                                      "available": "Ответ текущего дня: цель, не вход модели"}),
            "stats": {}, "views": {},
        }
        if categorical:
            series = {s: frames[s][name].astype("string").fillna("Пропуск") for s in SPLITS}
            # Общий список — 15 самых частых категорий ОБУЧЕНИЯ, остальные в одну.
            top = series["train"].value_counts().head(15).index.tolist()
            categories = [*top, "Остальные"]
            counts = {}
            for s, values in series.items():
                vc = values.value_counts()
                chosen = [int(vc.get(c, 0)) for c in top]
                counts[s] = [*chosen, len(values) - sum(chosen)]
                entry["stats"][s] = {
                    "n": len(values), "missing": float(frames[s][name].isna().mean()),
                    "unique": int(values.nunique()),
                }
            entry["views"]["full"] = {"labels": categories, "counts": counts}
        else:
            series = {s: pd.to_numeric(frames[s][name], errors="coerce").to_numpy(dtype=float) for s in SPLITS}
            finite = {s: a[np.isfinite(a)] for s, a in series.items()}
            combined_unique = np.unique(np.concatenate([np.unique(a) for a in finite.values()]))
            for s, a in finite.items():
                entry["stats"][s] = {
                    "n": len(series[s]), "missing": 1 - len(a) / len(series[s]),
                    "mean": float(np.mean(a)) if len(a) else None,
                    "median": float(np.median(a)) if len(a) else None,
                    "p99": float(np.quantile(a, .99)) if len(a) else None,
                    "min": float(np.min(a)) if len(a) else None,
                    "max": float(np.max(a)) if len(a) else None,
                    "zero": float(np.sum(a == 0) / len(series[s])),
                    "unique": int(len(np.unique(a))),
                }
            if len(combined_unique) <= 24:
                values = combined_unique.tolist()
                view = {"labels": values, "discrete": True, "counts": {}, "outside": {s: 0 for s in SPLITS}}
                for s, a in finite.items():
                    view["counts"][s] = [int(np.sum(a == v)) for v in values]
                entry["views"] = {"full": view, "central": view}
            else:
                train = finite["train"]
                lo, hi = np.quantile(train, [.005, .995]) if len(train) else (0., 1.)
                full_lo = min(float(a.min()) for a in finite.values() if len(a))
                full_hi = max(float(a.max()) for a in finite.values() if len(a))
                if hi <= lo:
                    hi = lo + 1
                for key, low, high in [("central", float(lo), float(hi)), ("full", full_lo, full_hi)]:
                    if high <= low:
                        high = low + 1
                    edges = np.linspace(low, high, 37)
                    view = {"edges": edges.tolist(), "counts": {}, "outside": {}}
                    for s, a in finite.items():
                        view["counts"][s] = np.histogram(a, bins=edges)[0].tolist()
                        view["outside"][s] = int(np.sum((a < low) | (a > high)))
                    entry["views"][key] = view
        result.append(entry)
        print(f"Распределение: {name}", flush=True)
    return result


def main() -> None:
    # Внешний каталог подписей уже используется во всех ноутбуках.
    import sys
    sys.path.insert(0, str(ROOT))
    from features.catalog import LABELS, GROUP_LABELS, MODEL_LABELS, identifier_label

    manifest = json.loads((ART / "features/feature_manifest.json").read_text())
    features = distribution_payload(manifest, LABELS, GROUP_LABELS)
    for feature in features:
        if feature["kind"] == "category":
            feature["views"]["full"]["display"] = [identifier_label(x) for x in feature["views"]["full"]["labels"]]

    payload = {
        "features": features, "manifest": manifest, "modelLabels": MODEL_LABELS,
        "labels": LABELS, "groups": GROUP_LABELS,
        "quality": json.loads((ART / "research/quality_summary.json").read_text()),
        "splits": records("features/split_summary.csv"),
        "paired": records("training/paired_comparison.csv"),
        "metrics": records("training/all_metrics.csv"),
        "ablation": records("training/catboost_ablation.csv"),
        "importance": records("training/catboost_fe_importance.csv"),
        "dailyPredictions": records("training/daily_predictions.csv"),
        "segments": records("training/error_segments.csv"),
        "warehouses": records("training/warehouse_metrics.csv"),
        "timePanel": records("time_analysis/daily_time_panel.csv"),
        "weekly": records("time_analysis/weekly_effect_intervals.csv"),
        "timeForecasts": records("time_analysis/forecast_summary.csv"),
        "timeFolds": records("time_analysis/forecast_fold_metrics.csv"),
        "tricks": records("tricks/validation_experiments.csv"),
        "tricksTest": records("tricks/fixed_test_results.csv"),
        "recipe": json.loads((ART / "tricks/selected_recipe.json").read_text()),
    }

    gallery = []
    for folder in ("research", "features", "training", "tricks", "time_analysis"):
        for p in sorted((ART / folder).glob("*.png")):
            gallery.append({"group": folder, "file": p.name, "source": p.relative_to(ROOT).as_posix(),
                            "image": "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()})
    payload["gallery"] = gallery
    payload["meta"] = {
        "builtAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sourceCommit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip(),
        "featureCount": len(features) - 1,
        "galleryCount": len(gallery),
        "sources": [],
    }
    for p in sorted(ART.rglob("*")):
        if p.is_file() and p.suffix in (".csv", ".json", ".png"):
            payload["meta"]["sources"].append({"path": p.relative_to(ROOT).as_posix(),
                                               "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    for s in SPLITS:
        p = ROOT / f"data/processed/{s}.parquet"
        payload["meta"]["sources"].append({"path": p.relative_to(ROOT).as_posix(),
                                           "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})

    # Экранируем закрывающий тег script в потенциальных строковых значениях.
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    template = (ROOT / "scripts/report_template.html").read_text()
    html = template.replace("__REPORT_DATA__", data).replace("__PLOTLY_JS__", get_plotlyjs())
    destination = ROOT / "reports/iteration_00.html"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(html)
    print(f"Готово: {destination} ({destination.stat().st_size / 2**20:.1f} МиБ)")


if __name__ == "__main__":
    main()
