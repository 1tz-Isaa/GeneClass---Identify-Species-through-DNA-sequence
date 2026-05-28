from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import json
import os

import numpy as np

from run_group_model_search import (
    config_grid,
    fit_predict,
    load_rows,
    choose_group_split,
    row_subset,
)


BASE = Path(__file__).resolve().parent
OUT = BASE / "runs" / "virus_model_search"
OUT.mkdir(parents=True, exist_ok=True)

REPEATS = int(os.getenv("VIRUS_ROBUST_REPEATS", "25"))
TOP_TUNE_CONFIGS = int(os.getenv("VIRUS_ROBUST_TOP_TUNE_CONFIGS", "24"))


def load_tune_rankings() -> dict[int, dict]:
    path = OUT / "virus_model_search_results.csv"
    rows: dict[int, dict] = {}
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            config_id = int(row["config_id"])
            for key in ["tune_accuracy", "tune_f1_macro", "tune_mcc"]:
                row[key] = float(row[key])
            rows[config_id] = row
    return rows


def candidate_configs() -> list[tuple[int, dict]]:
    configs = list(enumerate(config_grid("virus"), start=1))
    tune_rows = load_tune_rankings()

    candidate_ids = set()
    if tune_rows:
        ranked = sorted(
            tune_rows.values(),
            key=lambda row: (row["tune_f1_macro"], row["tune_mcc"], row["tune_accuracy"]),
            reverse=True,
        )
        candidate_ids.update(int(row["config_id"]) for row in ranked[:TOP_TUNE_CONFIGS])

    for config_id, config in configs:
        # Include longer-fragment candidates even if one tune split ranked them low.
        if int(config["fragment_len"]) == 5000 and (int(config["kmer_min"]), int(config["kmer_max"])) in {
            (5, 5),
            (6, 6),
            (6, 8),
        }:
            candidate_ids.add(config_id)

    return [(config_id, config) for config_id, config in configs if config_id in candidate_ids]


def evaluate_config(rows: list[dict], labels: list[str], groups: list[str], config_id: int, config: dict) -> dict:
    repeat_rows = []
    for seed in range(100, 100 + REPEATS):
        train_idx, val_idx = choose_group_split(labels, groups, test_size=0.30, random_state=seed)
        train_rows = row_subset(rows, train_idx)
        val_rows = row_subset(rows, val_idx)
        metrics, _ = fit_predict(train_rows, val_rows, config)
        repeat_rows.append(
            {
                "seed": seed,
                "validation_samples": len(val_rows),
                "validation_genera": len({r["genus"] for r in val_rows}),
                **metrics,
            }
        )

    summary = {
        "config_id": config_id,
        **config,
        "repeat_count": REPEATS,
    }
    for metric in ["accuracy", "balanced_accuracy", "f1_macro", "mcc"]:
        values = [float(row[metric]) for row in repeat_rows]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_median"] = float(np.median(values))
        summary[f"{metric}_min"] = float(np.min(values))
        summary[f"{metric}_max"] = float(np.max(values))

    return summary


def final_holdout_metrics(rows: list[dict], labels: list[str], groups: list[str], config: dict) -> tuple[dict, list[dict]]:
    train_idx, holdout_idx = choose_group_split(labels, groups, test_size=0.30, random_state=42)
    train_rows = row_subset(rows, train_idx)
    holdout_rows = row_subset(rows, holdout_idx)
    metrics, predictions = fit_predict(train_rows, holdout_rows, config)
    prediction_rows = [
        {**row, "prediction": pred, "correct": pred == row["genus"]}
        for row, pred in zip(holdout_rows, predictions)
    ]
    metrics = {
        **metrics,
        "train_samples": len(train_rows),
        "holdout_samples": len(holdout_rows),
        "holdout_genera": len({row["genus"] for row in holdout_rows}),
        "holdout_class_counts": dict(sorted(Counter(row["genus"] for row in holdout_rows).items())),
    }
    return metrics, prediction_rows


def main() -> None:
    rows = load_rows("virus")
    labels = [row["genus"] for row in rows]
    groups = [row["species_group"] for row in rows]
    candidates = candidate_configs()
    print(
        f"[virus robust] samples={len(rows)} genera={len(set(labels))} "
        f"species_groups={len(set(groups))} candidates={len(candidates)} repeats={REPEATS}",
        flush=True,
    )

    summaries = []
    for index, (config_id, config) in enumerate(candidates, start=1):
        summary = evaluate_config(rows, labels, groups, config_id, config)
        summaries.append(summary)
        print(
            f"[virus robust] {index}/{len(candidates)} cfg={config_id} "
            f"mean_acc={summary['accuracy_mean']:.3f} "
            f"mean_f1={summary['f1_macro_mean']:.3f} "
            f"mean_mcc={summary['mcc_mean']:.3f}",
            flush=True,
        )

    summaries.sort(
        key=lambda row: (row["f1_macro_mean"], row["mcc_mean"], row["accuracy_mean"]),
        reverse=True,
    )
    best = summaries[0]
    best_config = {
        key: best[key]
        for key in [
            "fragment_len",
            "pad",
            "kmer_min",
            "kmer_max",
            "min_df",
            "max_features",
            "class_weight",
            "C",
            "max_iter",
        ]
    }
    holdout, predictions = final_holdout_metrics(rows, labels, groups, best_config)

    eval_path = OUT / "virus_robust_config_evaluation.csv"
    with eval_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    pred_path = OUT / "virus_robust_holdout_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)

    payload = {
        "selection_rule": "Candidate configs are selected from top tune results plus long-fragment virus configs; final config is selected by repeated group-aware holdout mean F1 Macro, then mean MCC, then mean accuracy.",
        "data": {
            "samples": len(rows),
            "genera": len(set(labels)),
            "species_groups": len(set(groups)),
        },
        "best_repeated_summary": best,
        "best_config": best_config,
        "final_holdout_metrics": holdout,
    }
    summary_path = OUT / "virus_robust_selection_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[virus robust] best config id={best['config_id']} {json.dumps(best_config, sort_keys=True)}")
    print(
        f"[virus robust] repeated mean: acc={best['accuracy_mean']:.3f} "
        f"f1={best['f1_macro_mean']:.3f} mcc={best['mcc_mean']:.3f}"
    )
    print(
        f"[virus robust] final holdout: acc={holdout['accuracy']:.3f} "
        f"f1={holdout['f1_macro']:.3f} mcc={holdout['mcc']:.3f}"
    )
    print(f"[virus robust] wrote {summary_path}")


if __name__ == "__main__":
    main()
