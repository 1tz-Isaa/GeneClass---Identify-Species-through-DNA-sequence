from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import csv
import json
import re
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

BASE = Path(__file__).resolve().parent
BAD_HEADER_PATTERNS = (
    "patent",
    " jp ",
    " kr ",
    "synthetic construct",
    "cloning vector",
    "vector",
    "plasmid",
    "unverified",
    "partial genome",
    "partial sequence",
    "partial cds",
    "rna construct",
    "composition",
    "oligonucleotide",
    "extracellular vesicle",
    "vaccine against",
    "circular rna",
)

TARGETS = {
    "bacteria": {
        "root": BASE / "Database" / "bacteria_genus",
        "out": BASE / "runs" / "bacteria_model_search",
        "min_len": 200,
        "test_size": 0.30,
        "tune_test_size": 0.40,
        "repeats": 25,
    },
    "virus": {
        "root": BASE / "Database" / "rna_genus",
        "out": BASE / "runs" / "virus_model_search",
        "min_len": 300,
        "test_size": 0.30,
        "tune_test_size": 0.40,
        "repeats": 25,
    },
}


def read_fasta(path: Path):
    header = None
    seq_lines = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            yield header, "".join(seq_lines)


def clean_sequence(seq: str) -> str:
    return re.sub(r"[^ACGTUNacgtun]", "", seq).upper()


def center_fragment(seq: str, length: int, pad: bool) -> str:
    if length <= 0:
        return seq
    if len(seq) > length:
        start = (len(seq) - length) // 2
        return seq[start : start + length]
    if pad:
        return seq + ("N" * (length - len(seq)))
    return seq


def load_rows(target: str):
    cfg = TARGETS[target]
    root = cfg["root"]
    rows = []
    seen = set()
    for fp in sorted(root.glob("*/*/*.fasta")):
        genus = fp.parts[-3]
        species = fp.parts[-2]
        group = f"{genus}/{species}"
        for header, raw in read_fasta(fp):
            header_l = header.lower()
            if any(token in f" {header_l} " for token in BAD_HEADER_PATTERNS):
                continue
            if species == "SARS-CoV" and (
                "sars-cov-2" in header_l or "coronavirus 2" in header_l
            ):
                continue
            seq = clean_sequence(raw)
            if len(seq) < int(cfg["min_len"]):
                continue
            key = (genus, seq)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "path": str(fp),
                    "header": header,
                    "seq": seq,
                    "genus": genus,
                    "species_group": group,
                }
            )
    return rows


def choose_group_split(labels, groups, test_size: float, random_state: int, tries: int = 300):
    idx = np.arange(len(labels))
    total_classes = len(set(labels))
    target_n = round(len(labels) * test_size)
    best = None
    best_key = None
    for seed in range(random_state, random_state + 9):
        splitter = GroupShuffleSplit(n_splits=tries, test_size=test_size, random_state=seed)
        for train_idx, val_idx in splitter.split(idx, labels, groups=groups):
            train_classes = {labels[i] for i in train_idx}
            val_classes = {labels[i] for i in val_idx}
            coverage = len(val_classes)
            train_coverage = len(train_classes)
            balance = abs(len(val_idx) - target_n)
            val_counts = Counter(labels[i] for i in val_idx)
            min_val_class = min(val_counts.values()) if val_counts else 0
            key = (coverage, train_coverage, min_val_class, -balance)
            if best_key is None or key > best_key:
                best = (train_idx, val_idx)
                best_key = key
            if coverage == total_classes and train_coverage == total_classes and balance <= 3:
                return train_idx, val_idx
    if best is None:
        raise RuntimeError("Could not create group split")
    return best


def metrics(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    _, _, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f_macro),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def make_xy(rows, config):
    X = [center_fragment(r["seq"], int(config["fragment_len"]), bool(config["pad"])) for r in rows]
    y = [r["genus"] for r in rows]
    return X, y


def fit_predict(train_rows, val_rows, config):
    X_train, y_train = make_xy(train_rows, config)
    X_val, y_val = make_xy(val_rows, config)
    model = make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(int(config["kmer_min"]), int(config["kmer_max"])),
            lowercase=False,
            min_df=int(config["min_df"]),
            max_features=int(config["max_features"]),
            sublinear_tf=True,
            dtype=np.float32,
        ),
        OneVsRestClassifier(
            LogisticRegression(
                max_iter=int(config["max_iter"]),
                solver="liblinear",
                C=float(config["C"]),
                tol=0.001,
                class_weight=config["class_weight"],
                random_state=42,
            )
        ),
    )
    model.fit(X_train, y_train)
    pred = list(model.predict(X_val))
    return metrics(y_val, pred), pred


def config_grid(target: str):
    configs = []
    if target == "bacteria":
        fragments = [(0, False), (1000, False), (1200, False), (1500, False)]
        kmers = [(4, 6), (5, 5), (5, 7), (5, 8)]
        features = [25000, 50000, 80000]
    elif target == "virus":
        fragments = [(0, False), (1200, False), (2500, False), (5000, False)]
        kmers = [(5, 5), (5, 7), (6, 6), (6, 8)]
        features = [25000, 50000, 100000]
    else:
        raise ValueError(f"Unsupported target: {target}")

    for fragment_len, pad in fragments:
        for kmer_min, kmer_max in kmers:
            for max_features in features:
                for class_weight in ["balanced", None]:
                    configs.append(
                        {
                            "fragment_len": fragment_len,
                            "pad": pad,
                            "kmer_min": kmer_min,
                            "kmer_max": kmer_max,
                            "min_df": 2,
                            "max_features": max_features,
                            "class_weight": class_weight,
                            "C": 3.0,
                            "max_iter": 1200,
                        }
                    )
    return configs


def row_subset(rows, idxs):
    return [rows[int(i)] for i in idxs]


def run_target(target: str):
    target_cfg = TARGETS[target]
    out_dir = target_cfg["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(target)
    labels = [r["genus"] for r in rows]
    groups = [r["species_group"] for r in rows]
    total_genera = len(set(labels))

    final_train_idx, final_holdout_idx = choose_group_split(
        labels, groups, test_size=float(target_cfg["test_size"]), random_state=42
    )
    final_train_rows = row_subset(rows, final_train_idx)
    holdout_rows = row_subset(rows, final_holdout_idx)

    tune_labels = [r["genus"] for r in final_train_rows]
    tune_groups = [r["species_group"] for r in final_train_rows]
    tune_train_idx, tune_val_idx = choose_group_split(
        tune_labels,
        tune_groups,
        test_size=float(target_cfg["tune_test_size"]),
        random_state=2026,
    )
    tune_train_rows = row_subset(final_train_rows, tune_train_idx)
    tune_val_rows = row_subset(final_train_rows, tune_val_idx)

    search_rows = []
    configs = config_grid(target)
    print(f"[{target}] loaded {len(rows)} samples across {total_genera} genera")
    print(
        f"[{target}] final holdout: {len(holdout_rows)} samples, "
        f"{len(set(r['genus'] for r in holdout_rows))}/{total_genera} genera"
    )
    print(
        f"[{target}] tune validation: {len(tune_val_rows)} samples, "
        f"{len(set(r['genus'] for r in tune_val_rows))}/{total_genera} genera"
    )
    print(f"[{target}] testing {len(configs)} configs", flush=True)

    for i, config in enumerate(configs, start=1):
        tune_metrics, _ = fit_predict(tune_train_rows, tune_val_rows, config)
        row = {
            "config_id": i,
            **config,
            "tune_samples": len(tune_val_rows),
            "tune_genera": len(set(r["genus"] for r in tune_val_rows)),
            **{f"tune_{k}": v for k, v in tune_metrics.items()},
        }
        search_rows.append(row)
        if i % 24 == 0:
            best = sorted(search_rows, key=lambda r: (r["tune_f1_macro"], r["tune_mcc"], r["tune_accuracy"]), reverse=True)[0]
            print(
                f"[{target}] {i}/{len(configs)} best tune f1={best['tune_f1_macro']:.3f} "
                f"acc={best['tune_accuracy']:.3f} cfg={best['config_id']}",
                flush=True,
            )

    search_rows.sort(key=lambda r: (r["tune_f1_macro"], r["tune_mcc"], r["tune_accuracy"]), reverse=True)
    best_tune = search_rows[0]
    best_config = {
        k: best_tune[k]
        for k in [
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

    final_metrics, holdout_pred = fit_predict(final_train_rows, holdout_rows, best_config)
    holdout_true = [r["genus"] for r in holdout_rows]
    holdout_out = [{**r, "prediction": pred, "correct": pred == r["genus"]} for r, pred in zip(holdout_rows, holdout_pred)]

    repeated = []
    for seed in range(100, 100 + int(target_cfg["repeats"])):
        train_idx, val_idx = choose_group_split(labels, groups, test_size=float(target_cfg["test_size"]), random_state=seed)
        train_rows = row_subset(rows, train_idx)
        val_rows = row_subset(rows, val_idx)
        repeat_metrics, _ = fit_predict(train_rows, val_rows, best_config)
        repeated.append(
            {
                "seed": seed,
                "train_samples": len(train_rows),
                "validation_samples": len(val_rows),
                "validation_genera": len(set(r["genus"] for r in val_rows)),
                **repeat_metrics,
            }
        )

    with (out_dir / f"{target}_model_search_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(search_rows[0].keys()))
        writer.writeheader()
        writer.writerows(search_rows)

    with (out_dir / f"{target}_final_holdout_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(holdout_out[0].keys()))
        writer.writeheader()
        writer.writerows(holdout_out)

    with (out_dir / f"{target}_repeated_group_holdout.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(repeated[0].keys()))
        writer.writeheader()
        writer.writerows(repeated)

    repeated_summary = {}
    for key in ["accuracy", "f1_macro", "mcc"]:
        vals = [r[key] for r in repeated]
        repeated_summary[key] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    summary = {
        "target": target,
        "data": {
            "samples": len(rows),
            "genera": total_genera,
            "species_groups": len(set(groups)),
            "final_train_samples": len(final_train_rows),
            "final_holdout_samples": len(holdout_rows),
            "final_holdout_genera": len(set(holdout_true)),
            "tune_train_samples": len(tune_train_rows),
            "tune_validation_samples": len(tune_val_rows),
            "tune_validation_genera": len(set(r["genus"] for r in tune_val_rows)),
        },
        "selection_rule": "Select by tune F1 Macro, then tune MCC, then tune accuracy. Final holdout tested once after selection.",
        "best_tune": best_tune,
        "best_config": best_config,
        "final_holdout_metrics": final_metrics,
        "final_holdout_class_counts": dict(sorted(Counter(holdout_true).items())),
        "repeated_group_holdout_summary": repeated_summary,
    }
    (out_dir / f"{target}_model_search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{target}] best config: {json.dumps(best_config, sort_keys=True)}")
    print(
        f"[{target}] tune: acc={best_tune['tune_accuracy']:.3f} "
        f"f1={best_tune['tune_f1_macro']:.3f} mcc={best_tune['tune_mcc']:.3f}"
    )
    print(
        f"[{target}] final holdout: acc={final_metrics['accuracy']:.3f} "
        f"f1={final_metrics['f1_macro']:.3f} mcc={final_metrics['mcc']:.3f}"
    )
    print(
        f"[{target}] repeated mean: acc={repeated_summary['accuracy']['mean']:.3f} "
        f"f1={repeated_summary['f1_macro']['mean']:.3f} mcc={repeated_summary['mcc']['mean']:.3f}"
    )
    print(f"[{target}] wrote {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=sorted(TARGETS))
    args = parser.parse_args()
    run_target(args.target)


if __name__ == "__main__":
    main()
