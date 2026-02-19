"""Entrypoint for training sequence classifiers.

This file keeps runtime flow simple; heavy logic is split into:
- training/config.py
- training/core.py
- training/artifacts.py
"""

import csv
import json
import warnings
from collections import Counter
from datetime import datetime, timezone
from time import perf_counter

from sklearn.metrics import log_loss

from training.artifacts import (
    append_history_csv,
    build_report,
    save_model_bundle,
    send_email_report,
    write_classification_report_csv,
    write_confusion_matrix_csv,
)
from training.config import format_target_table, load_config
from training.core import (
    build_model,
    calc_metrics,
    prepare_dataset,
    run_cv,
    split_dataset,
    write_genus_accuracy_timeline,
)


# Hide noisy sklearn warning when class coverage differs across group folds.
warnings.filterwarnings(
    "ignore",
    message=".*y_pred contains classes not in y_true.*",
    category=UserWarning,
)


def main() -> None:
    run_started = perf_counter()
    started_utc = datetime.now(timezone.utc)
    cfg = load_config()

    if cfg.show_target_table:
        print(format_target_table())
    print(f"[TARGET] input='{cfg.train_target_input}' resolved='{cfg.train_target}'")

    # 1) Data load + cleanup.
    ds = prepare_dataset(cfg)
    sequences = ds["sequences"]
    labels = ds["labels"]
    genera = ds["genera"]
    groups = ds["groups"]

    # 2) Train/validation split.
    train_idx, val_idx, use_stratify = split_dataset(labels, groups, cfg)

    x_train = [sequences[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    x_val = [sequences[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]
    val_genera = [genera[i] for i in val_idx]

    # 3) Fit model + compute metrics.
    model = build_model(cfg)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)

    train_metrics = calc_metrics(y_train, train_pred)
    val_metrics = calc_metrics(y_val, val_pred)

    try:
        proba = model.predict_proba(x_val)
        val_metrics["log_loss"] = float(log_loss(y_val, proba, labels=model.classes_))
    except Exception:
        val_metrics["log_loss"] = float("nan")

    cv_metrics = run_cv(sequences, labels, groups, cfg)

    # 4) Persist artifacts, summary, and registry models.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.runs_root / f"run_{run_id}_{cfg.train_target}_{cfg.label_level}"
    run_dir.mkdir(parents=True, exist_ok=True)

    labels_sorted = sorted(set(labels))
    class_counts = Counter(labels)
    validation_classes_present = len(set(y_val))
    validation_classes_missing = len(class_counts) - validation_classes_present

    if validation_classes_missing > 0:
        print(
            f"[WARN] Validation has missing classes: {validation_classes_missing} "
            f"(present {validation_classes_present}/{len(class_counts)})"
        )

    with (run_dir / "class_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "count"])
        for label, count in sorted(class_counts.items()):
            writer.writerow([label, count])

    write_confusion_matrix_csv(run_dir / "confusion_matrix_val.csv", y_val, val_pred, labels_sorted)
    write_classification_report_csv(run_dir / "classification_report_val.csv", y_val, val_pred, labels_sorted)

    with (run_dir / "predictions_val.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["y_true", "y_pred", "genus"])
        for true_label, pred_label, genus in zip(y_val, val_pred, val_genera):
            writer.writerow([true_label, pred_label, genus])

    timeline_stats = write_genus_accuracy_timeline(
        run_dir / "accuracy_timeline_by_genus.csv",
        val_genera=val_genera,
        y_true=y_val,
        y_pred=val_pred,
        show_check_progress=cfg.show_check_progress,
    )

    model_bundle = save_model_bundle(
        model=model,
        run_dir=run_dir,
        run_id=run_id,
        val_accuracy=val_metrics["accuracy"],
        cfg=cfg,
    )

    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": started_utc.isoformat(),
        "duration_seconds": perf_counter() - run_started,
        "config": {
            "target": cfg.train_target,
            "label_level": cfg.label_level,
            "root_folder": ds["root_folder"],
            "kingdom": ds["kingdom"],
            "split_mode": cfg.split_mode,
            "dedup_exact": cfg.dedup_exact,
            "filter_bad_headers": cfg.filter_bad_headers,
            "min_seq_len": cfg.min_seq_len,
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "cv_folds": cfg.cv_folds,
            "group_split_tries": cfg.group_split_tries,
            "use_stratify": use_stratify,
                "model": {
                    "type": "kmer_lr",
                    "kmer_min": cfg.kmer_min,
                    "kmer_max": cfg.kmer_max,
                    "kmer_min_df": cfg.kmer_min_df,
                    "kmer_max_features": cfg.kmer_max_features,
                    "lr_c": cfg.lr_c,
                    "lr_max_iter": cfg.lr_max_iter,
                    "lr_solver": cfg.lr_solver,
                    "lr_class_weight": cfg.lr_class_weight,
                },
            },
        "data": {
            "n_samples_total": len(labels),
            "n_classes_total": len(class_counts),
            "n_validation_classes_present": validation_classes_present,
            "n_validation_classes_missing": validation_classes_missing,
            "n_train": len(y_train),
            "n_validation": len(y_val),
            "n_validation_genera": timeline_stats["genera_checked"],
            "n_groups_total": len(set(groups)),
            "n_groups_train": len(set(groups[i] for i in train_idx)),
            "n_groups_validation": len(set(groups[i] for i in val_idx)),
            "min_class_count_total": min(class_counts.values()),
        },
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
        },
        "cv": cv_metrics,
        "artifacts": {
            "run_dir": str(run_dir.resolve()),
            "class_counts_csv": str((run_dir / "class_counts.csv").resolve()),
            "confusion_matrix_csv": str((run_dir / "confusion_matrix_val.csv").resolve()),
            "classification_report_csv": str((run_dir / "classification_report_val.csv").resolve()),
            "predictions_csv": str((run_dir / "predictions_val.csv").resolve()),
            "accuracy_timeline_csv": str((run_dir / "accuracy_timeline_by_genus.csv").resolve()),
            "run_model": model_bundle["run_model"],
            "latest_model": model_bundle["latest_model"],
            "best_model": model_bundle["best_model"],
            "best_val_accuracy": model_bundle["best_val_accuracy"],
            "is_new_best": model_bundle["is_new_best"],
        },
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    append_history_csv(
        cfg.runs_root / "history.csv",
        {
            "run_id": run_id,
            "timestamp_utc": summary["timestamp_utc"],
            "started_utc": summary["started_utc"],
            "duration_seconds": summary["duration_seconds"],
            "target": cfg.train_target,
            "label_level": cfg.label_level,
            "split_mode": cfg.split_mode,
            "dedup_exact": cfg.dedup_exact,
            "test_size": cfg.test_size,
            "kmer_min": cfg.kmer_min,
            "kmer_max": cfg.kmer_max,
            "kmer_min_df": cfg.kmer_min_df,
            "kmer_max_features": cfg.kmer_max_features,
            "lr_c": cfg.lr_c,
            "lr_max_iter": cfg.lr_max_iter,
            "lr_solver": cfg.lr_solver,
            "lr_class_weight": cfg.lr_class_weight,
            "n_samples_total": summary["data"]["n_samples_total"],
            "n_classes_total": summary["data"]["n_classes_total"],
            "n_validation_classes_present": summary["data"]["n_validation_classes_present"],
            "n_validation_classes_missing": summary["data"]["n_validation_classes_missing"],
            "n_validation_genera": summary["data"]["n_validation_genera"],
            "n_train": summary["data"]["n_train"],
            "n_validation": summary["data"]["n_validation"],
            "n_groups_total": summary["data"]["n_groups_total"],
            "n_groups_train": summary["data"]["n_groups_train"],
            "n_groups_validation": summary["data"]["n_groups_validation"],
            "min_class_count_total": summary["data"]["min_class_count_total"],
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_precision_macro": train_metrics["precision_macro"],
            "train_recall_macro": train_metrics["recall_macro"],
            "train_f1_macro": train_metrics["f1_macro"],
            "train_precision_weighted": train_metrics["precision_weighted"],
            "train_recall_weighted": train_metrics["recall_weighted"],
            "train_f1_weighted": train_metrics["f1_weighted"],
            "train_mcc": train_metrics["mcc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_precision_weighted": val_metrics["precision_weighted"],
            "val_recall_weighted": val_metrics["recall_weighted"],
            "val_f1_weighted": val_metrics["f1_weighted"],
            "val_mcc": val_metrics["mcc"],
            "val_log_loss": val_metrics.get("log_loss", ""),
            "cv_enabled": cv_metrics.get("enabled", False),
            "cv_splits": cv_metrics.get("n_splits", ""),
            "cv_accuracy_mean": cv_metrics.get("accuracy_mean", ""),
            "cv_accuracy_std": cv_metrics.get("accuracy_std", ""),
            "cv_balanced_accuracy_mean": cv_metrics.get("balanced_accuracy_mean", ""),
            "cv_balanced_accuracy_std": cv_metrics.get("balanced_accuracy_std", ""),
            "cv_f1_macro_mean": cv_metrics.get("f1_macro_mean", ""),
            "cv_f1_macro_std": cv_metrics.get("f1_macro_std", ""),
            "cv_f1_weighted_mean": cv_metrics.get("f1_weighted_mean", ""),
            "cv_f1_weighted_std": cv_metrics.get("f1_weighted_std", ""),
            "run_model_path": model_bundle["run_model"],
            "latest_model_path": model_bundle["latest_model"],
            "best_model_path": model_bundle["best_model"],
            "is_new_best": model_bundle["is_new_best"],
            "run_dir": str(run_dir.resolve()),
        },
    )

    report = build_report(summary)
    print(report)
    send_email_report(report, cfg)


if __name__ == "__main__":
    main()
