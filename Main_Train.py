"""Entrypoint for training sequence classifiers.

This file keeps runtime flow simple; heavy logic is split into:
- training/config.py
- training/core.py
- training/artifacts.py
"""

import csv
from dataclasses import replace
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
    calc_metrics,
    fit_model,
    prepare_dataset,
    rna_genus_to_family,
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
    print(
        f"[PRESET] train_preset='{cfg.train_preset}' enable_cv={int(cfg.enable_cv)} "
        f"auto_tune_lr_c={int(cfg.auto_tune_lr_c)} "
        f"train_fragment_len={cfg.train_fragment_len} "
        f"max_seq_len={cfg.max_seq_len} max_samples_total={cfg.max_samples_total} "
        f"max_samples_per_label={cfg.max_samples_per_label}"
    )
    if cfg.train_target == "rna":
        print(
            "[RNA] "
            f"min_unique_genomes_per_label={cfg.rna_min_unique_genomes_per_label} "
            f"use_family_fragment_len={int(cfg.rna_use_family_fragment_len)} "
            f"family_top_k={cfg.rna_family_top_k} "
            f"hierarchical_weight={cfg.rna_hierarchical_weight:.2f} "
            f"collapse_nested_species={int(cfg.rna_collapse_nested_species)}"
        )

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
    groups_train = [groups[i] for i in train_idx]
    x_val = [sequences[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]
    val_genera = [genera[i] for i in val_idx]

    # 3) Fit model + compute metrics.
    model, fit_info = fit_model(x_train, y_train, groups_train, cfg)
    effective_cfg = replace(cfg, lr_c=float(fit_info.get("selected_lr_c", cfg.lr_c)))

    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)

    train_metrics = calc_metrics(y_train, train_pred)
    val_metrics = calc_metrics(y_val, val_pred)
    family_train_metrics = None
    family_val_metrics = None
    y_val_family = None
    family_val_pred = None
    family_labels_sorted = None
    family_top_confusions = []
    global_only_train_metrics = None
    global_only_val_metrics = None
    hierarchy_delta_vs_global = None

    try:
        proba = model.predict_proba(x_val)
        val_metrics["log_loss"] = float(log_loss(y_val, proba, labels=model.classes_))
    except Exception:
        val_metrics["log_loss"] = float("nan")

    if fit_info.get("hierarchical_family_mode") and hasattr(model, "family_model"):
        try:
            y_train_family = [rna_genus_to_family(x) for x in y_train]
            y_val_family = [rna_genus_to_family(x) for x in y_val]
            family_train_pred = model.family_model.predict(x_train)
            family_val_pred = model.family_model.predict(x_val)
            family_train_metrics = calc_metrics(y_train_family, family_train_pred)
            family_val_metrics = calc_metrics(y_val_family, family_val_pred)
            family_labels_sorted = sorted(set(y_val_family) | set(family_val_pred))

            confusion_counts = Counter()
            for true_family, pred_family in zip(y_val_family, family_val_pred):
                if true_family != pred_family:
                    confusion_counts[(true_family, pred_family)] += 1
            family_top_confusions = [
                {"true_family": t, "pred_family": p, "count": int(c)}
                for (t, p), c in confusion_counts.most_common(5)
            ]
        except Exception as exc:
            fit_info["family_eval_error"] = str(exc)

    if fit_info.get("hierarchical_family_mode") and hasattr(model, "global_model") and model.global_model is not None:
        try:
            global_only_train_pred = model.global_model.predict(x_train)
            global_only_val_pred = model.global_model.predict(x_val)
            global_only_train_metrics = calc_metrics(y_train, global_only_train_pred)
            global_only_val_metrics = calc_metrics(y_val, global_only_val_pred)
            hierarchy_delta_vs_global = {
                "accuracy_delta": float(val_metrics["accuracy"] - global_only_val_metrics["accuracy"]),
                "balanced_accuracy_delta": float(
                    val_metrics["balanced_accuracy"] - global_only_val_metrics["balanced_accuracy"]
                ),
                "f1_macro_delta": float(val_metrics["f1_macro"] - global_only_val_metrics["f1_macro"]),
            }
        except Exception as exc:
            fit_info["global_only_eval_error"] = str(exc)

    cv_metrics = run_cv(sequences, labels, groups, effective_cfg)

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
    family_confusion_path = ""
    if family_val_pred is not None and y_val_family is not None and family_labels_sorted:
        family_confusion_file = run_dir / "family_confusion_matrix_val.csv"
        write_confusion_matrix_csv(family_confusion_file, y_val_family, family_val_pred, family_labels_sorted)
        family_confusion_path = str(family_confusion_file.resolve())

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
        cfg=effective_cfg,
    )

    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": started_utc.isoformat(),
        "duration_seconds": perf_counter() - run_started,
        "config": {
            "target": effective_cfg.train_target,
            "train_preset": effective_cfg.train_preset,
            "label_level": effective_cfg.label_level,
            "root_folder": ds["root_folder"],
            "kingdom": ds["kingdom"],
            "split_mode": effective_cfg.split_mode,
            "enable_cv": effective_cfg.enable_cv,
            "dedup_exact": effective_cfg.dedup_exact,
            "filter_bad_headers": effective_cfg.filter_bad_headers,
            "min_seq_len": effective_cfg.min_seq_len,
            "train_fragment_len": effective_cfg.train_fragment_len,
            "max_seq_len": effective_cfg.max_seq_len,
            "max_samples_total": effective_cfg.max_samples_total,
            "max_samples_per_label": effective_cfg.max_samples_per_label,
            "rna_min_unique_genomes_per_label": effective_cfg.rna_min_unique_genomes_per_label,
            "rna_min_samples_per_label": effective_cfg.rna_min_samples_per_label,
            "rna_use_family_fragment_len": effective_cfg.rna_use_family_fragment_len,
            "rna_family_top_k": effective_cfg.rna_family_top_k,
            "rna_hierarchical_weight": effective_cfg.rna_hierarchical_weight,
            "rna_collapse_nested_species": effective_cfg.rna_collapse_nested_species,
            "test_size": effective_cfg.test_size,
            "random_state": effective_cfg.random_state,
            "cv_folds": effective_cfg.cv_folds,
            "group_split_tries": effective_cfg.group_split_tries,
            "cpu_jobs": effective_cfg.cpu_jobs,
            "auto_tune_lr_c": effective_cfg.auto_tune_lr_c,
            "auto_tune_max_samples": effective_cfg.auto_tune_max_samples,
            "auto_tune_holdout_size": effective_cfg.auto_tune_holdout_size,
            "fit_info": fit_info,
            "use_stratify": use_stratify,
            "model": {
                "type": "kmer_lr",
                "kmer_min": effective_cfg.kmer_min,
                "kmer_max": effective_cfg.kmer_max,
                "kmer_min_df": effective_cfg.kmer_min_df,
                "kmer_max_features": effective_cfg.kmer_max_features,
                "lr_c": effective_cfg.lr_c,
                "lr_max_iter": effective_cfg.lr_max_iter,
                "lr_tol": effective_cfg.lr_tol,
                "lr_solver": effective_cfg.lr_solver,
                "lr_class_weight": effective_cfg.lr_class_weight,
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
            "n_labels_dropped_min_unique_genomes": int(
                len(ds.get("dropped_labels_min_unique_genomes", {}))
            ),
            "n_samples_dropped_min_unique_genomes": int(ds.get("dropped_samples_min_unique_genomes", 0)),
            "dropped_labels_min_unique_genomes": ds.get("dropped_labels_min_unique_genomes", {}),
            "n_labels_dropped_min_samples_per_label": int(
                len(ds.get("dropped_labels_min_samples_per_label", {}))
            ),
            "n_samples_dropped_min_samples_per_label": int(ds.get("dropped_samples_min_samples_per_label", 0)),
            "dropped_labels_min_samples_per_label": ds.get("dropped_labels_min_samples_per_label", {}),
        },
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "family_train": family_train_metrics,
            "family_validation": family_val_metrics,
            "family_top_confusions": family_top_confusions,
            "global_only_train": global_only_train_metrics,
            "global_only_validation": global_only_val_metrics,
            "hierarchy_delta_vs_global": hierarchy_delta_vs_global,
        },
        "cv": cv_metrics,
        "artifacts": {
            "run_dir": str(run_dir.resolve()),
            "class_counts_csv": str((run_dir / "class_counts.csv").resolve()),
            "confusion_matrix_csv": str((run_dir / "confusion_matrix_val.csv").resolve()),
            "classification_report_csv": str((run_dir / "classification_report_val.csv").resolve()),
            "predictions_csv": str((run_dir / "predictions_val.csv").resolve()),
            "accuracy_timeline_csv": str((run_dir / "accuracy_timeline_by_genus.csv").resolve()),
            "family_confusion_matrix_csv": family_confusion_path,
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
            "train_preset": effective_cfg.train_preset,
            "label_level": effective_cfg.label_level,
            "split_mode": effective_cfg.split_mode,
            "enable_cv": effective_cfg.enable_cv,
            "dedup_exact": effective_cfg.dedup_exact,
            "test_size": effective_cfg.test_size,
            "train_fragment_len": effective_cfg.train_fragment_len,
            "max_seq_len": effective_cfg.max_seq_len,
            "max_samples_total": effective_cfg.max_samples_total,
            "max_samples_per_label": effective_cfg.max_samples_per_label,
            "rna_min_unique_genomes_per_label": effective_cfg.rna_min_unique_genomes_per_label,
            "rna_min_samples_per_label": effective_cfg.rna_min_samples_per_label,
            "kmer_min": effective_cfg.kmer_min,
            "kmer_max": effective_cfg.kmer_max,
            "kmer_min_df": effective_cfg.kmer_min_df,
            "kmer_max_features": effective_cfg.kmer_max_features,
            "lr_c": effective_cfg.lr_c,
            "lr_max_iter": effective_cfg.lr_max_iter,
            "lr_tol": effective_cfg.lr_tol,
            "lr_solver": effective_cfg.lr_solver,
            "lr_class_weight": effective_cfg.lr_class_weight,
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
            "n_labels_dropped_min_unique_genomes": summary["data"]["n_labels_dropped_min_unique_genomes"],
            "n_samples_dropped_min_unique_genomes": summary["data"]["n_samples_dropped_min_unique_genomes"],
            "n_labels_dropped_min_samples_per_label": summary["data"]["n_labels_dropped_min_samples_per_label"],
            "n_samples_dropped_min_samples_per_label": summary["data"]["n_samples_dropped_min_samples_per_label"],
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
            "family_val_accuracy": (
                family_val_metrics["accuracy"] if isinstance(family_val_metrics, dict) else ""
            ),
            "global_only_val_accuracy": (
                global_only_val_metrics["accuracy"] if isinstance(global_only_val_metrics, dict) else ""
            ),
            "hierarchy_delta_vs_global_acc": (
                hierarchy_delta_vs_global["accuracy_delta"] if isinstance(hierarchy_delta_vs_global, dict) else ""
            ),
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
