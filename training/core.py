import csv
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from dataset_loader import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline

from training.config import TARGET_CONFIG, TrainConfig, format_target_table

BAD_HEADER_PATTERNS = (
    "patent",
    "synthetic construct",
    "cloning vector",
    "vector",
    "plasmid",
)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def clean_sequence(seq: str) -> str:
    return re.sub(r"[^A-Za-z]", "", seq).upper()


def is_bad_header(header: str) -> bool:
    h = (header or "").lower()
    return any(token in h for token in BAD_HEADER_PATTERNS)


def build_model(cfg: TrainConfig):
    return make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(cfg.kmer_min, cfg.kmer_max),
            lowercase=False,
            min_df=cfg.kmer_min_df,
            max_features=cfg.kmer_max_features,
            sublinear_tf=True,
        ),
        LogisticRegression(
            max_iter=cfg.lr_max_iter,
            solver=cfg.lr_solver,
            C=cfg.lr_c,
            class_weight=cfg.lr_class_weight,
            n_jobs=None,
        ),
    )


def prepare_dataset(cfg: TrainConfig):
    if cfg.train_target not in TARGET_CONFIG:
        raise ValueError(
            f"Invalid TRAIN_TARGET input='{cfg.train_target_input}' resolved='{cfg.train_target}'.\n"
            f"{format_target_table()}"
        )

    root_folder = TARGET_CONFIG[cfg.train_target]["root"]
    kingdom_name = TARGET_CONFIG[cfg.train_target]["kingdom"]

    data = load_dataset(root_folder, show_progress=cfg.show_file_progress)

    rows = []
    for item in data:
        if item["kingdom"] != kingdom_name:
            continue

        seq = clean_sequence(item["sequence"])
        if not seq or len(seq) < cfg.min_seq_len:
            continue

        header = item.get("header", "")
        if cfg.filter_bad_headers and is_bad_header(header):
            continue

        label = item[cfg.label_level]
        genus = item["genus"]
        species = item["species"]
        group = f"{genus}/{species}"
        rows.append((seq, label, genus, group))

    if cfg.dedup_exact:
        deduped = []
        seen = set()
        for seq, label, genus, group in rows:
            key = (seq, label)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((seq, label, genus, group))
        rows = deduped

    if not rows:
        raise ValueError(f"No data found for target={cfg.train_target} in {root_folder}/{kingdom_name}")

    labels = [row[1] for row in rows]
    if len(set(labels)) < 2:
        raise ValueError(f"Need at least 2 classes to train, found: {sorted(set(labels))}")

    return {
        "root_folder": root_folder,
        "kingdom": kingdom_name,
        "sequences": [row[0] for row in rows],
        "labels": labels,
        "genera": [row[2] for row in rows],
        "groups": [row[3] for row in rows],
    }


def split_dataset(labels, groups, cfg: TrainConfig):
    all_idx = list(range(len(labels)))
    total_classes = len(set(labels))

    if cfg.split_mode == "group_species" and len(set(groups)) >= 2:
        n_splits = max(1, cfg.group_split_tries)
        splitter = GroupShuffleSplit(n_splits=n_splits, test_size=cfg.test_size, random_state=cfg.random_state)

        best = None
        best_cov = -1
        best_balance = float("inf")
        target_val_size = int(round(len(labels) * cfg.test_size))

        for train_idx, val_idx in splitter.split(all_idx, labels, groups=groups):
            val_labels = {labels[i] for i in val_idx}
            cov = len(val_labels)
            balance = abs(len(val_idx) - target_val_size)

            if cov > best_cov or (cov == best_cov and balance < best_balance):
                best = (train_idx.tolist(), val_idx.tolist())
                best_cov = cov
                best_balance = balance

            if cov == total_classes:
                break

        if best is not None:
            return best[0], best[1], False

    min_class_count = min(Counter(labels).values())
    use_stratify = min_class_count >= 2
    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=labels if use_stratify else None,
    )
    return list(train_idx), list(val_idx), use_stratify


def calc_metrics(y_true, y_pred):
    eval_labels = sorted(set(y_true) | set(y_pred))

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=eval_labels,
        average="macro",
        zero_division=0,
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=eval_labels,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f_weighted),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
    }


def run_cv(sequences, labels, groups, cfg: TrainConfig):
    class_counts = Counter(labels)
    min_class_count = min(class_counts.values())

    if cfg.split_mode == "group_species":
        n_splits = min(cfg.cv_folds, len(set(groups)))
        if n_splits < 2:
            return {"enabled": False, "reason": "Not enough groups for GroupKFold"}
        cv = GroupKFold(n_splits=n_splits)
        cv_groups = groups
    else:
        n_splits = min(cfg.cv_folds, min_class_count)
        if n_splits < 2:
            return {"enabled": False, "reason": "Not enough samples for StratifiedKFold"}
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
        cv_groups = None

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }

    scores = cross_validate(
        build_model(cfg),
        sequences,
        labels,
        cv=cv,
        groups=cv_groups,
        scoring=scoring,
        return_train_score=False,
        n_jobs=None,
        error_score="raise",
    )

    out = {"enabled": True, "n_splits": int(n_splits)}
    for metric_name in scoring:
        arr = scores[f"test_{metric_name}"]
        out[f"{metric_name}_mean"] = float(arr.mean())
        out[f"{metric_name}_std"] = float(arr.std())
    return out


def write_genus_accuracy_timeline(path: Path, val_genera, y_true, y_pred, show_check_progress: bool):
    genus_stats = {}
    for genus, true_label, pred_label in zip(val_genera, y_true, y_pred):
        if genus not in genus_stats:
            genus_stats[genus] = {"total": 0, "correct": 0}
        genus_stats[genus]["total"] += 1
        if true_label == pred_label:
            genus_stats[genus]["correct"] += 1

    cumulative_total = 0
    cumulative_correct = 0
    started = perf_counter()
    total_genera = len(genus_stats)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "genus",
                "genus_samples",
                "genus_correct",
                "genus_accuracy",
                "cumulative_samples",
                "cumulative_correct",
                "cumulative_accuracy",
                "elapsed_seconds",
                "timestamp_utc",
            ]
        )

        cumulative_acc = 0.0
        for step, (genus, stats) in enumerate(genus_stats.items(), start=1):
            total = int(stats["total"])
            correct = int(stats["correct"])
            genus_acc = (correct / total) if total else 0.0

            cumulative_total += total
            cumulative_correct += correct
            cumulative_acc = cumulative_correct / cumulative_total if cumulative_total else 0.0

            if show_check_progress:
                print(
                    f"[CHECK] {step}/{total_genera} genus={genus} "
                    f"genus_acc={genus_acc:.4f} cumulative_acc={cumulative_acc:.4f}"
                )

            writer.writerow(
                [
                    step,
                    genus,
                    total,
                    correct,
                    genus_acc,
                    cumulative_total,
                    cumulative_correct,
                    cumulative_acc,
                    perf_counter() - started,
                    datetime.now(timezone.utc).isoformat(),
                ]
            )

    return {"genera_checked": total_genera, "final_cumulative_accuracy": cumulative_acc}
