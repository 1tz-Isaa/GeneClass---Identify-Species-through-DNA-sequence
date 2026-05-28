"""Train app-facing genus models from the local Database folders.

These models are for inference/demo use in Judge_App and Central_Code_Reader.
Benchmark metrics should still come from run_controlled_benchmark.py.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import re

import numpy as np
from joblib import dump
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
import warnings


BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "runs" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
        "kmer_range": (5, 8),
        "max_features": 25000,
    },
    "fungi": {
        "root": BASE / "Database" / "fungi_genus",
        "kmer_range": (5, 8),
        "max_features": 25000,
    },
    "rna": {
        "root": BASE / "Database" / "rna_genus",
        "kmer_range": (5, 5),
        "max_features": 25000,
    },
}

RNA_GENUS_TO_FAMILY = {
    "Alphacoronavirus": "Coronaviridae",
    "Betacoronavirus": "Coronaviridae",
    "Enterovirus": "Picornaviridae",
    "Rhinovirus": "Picornaviridae",
    "Influenzavirus": "Orthomyxoviridae",
    "Hantavirus": "Hantaviridae",
    "Orthobunyavirus": "Peribunyaviridae",
    "Reovirus": "Reoviridae",
    "Arenavirus": "Arenaviridae",
    "Metapneumovirus": "Pneumoviridae",
    "Orthopneumovirus": "Pneumoviridae",
    "Respirovirus": "Paramyxoviridae",
}

RNA_FAMILY_CONFIG = {
    "kmer_range": (6, 8),
    "max_features": 30000,
}


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def read_fasta(path: Path):
    header = None
    seq_lines: list[str] = []
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


def split_windows(
    seq: str,
    window_size: int = 1500,
    stride: int = 750,
    min_len: int = 120,
    max_windows: int = 8,
) -> list[str]:
    if len(seq) <= window_size:
        return [seq] if len(seq) >= min_len else []

    windows = []
    start = 0
    while start < len(seq) and len(windows) < max_windows:
        window = seq[start : start + window_size]
        if len(window) < min_len:
            break
        windows.append(window)
        if start + window_size >= len(seq):
            break
        start += stride

    tail = seq[-window_size:]
    if len(tail) >= min_len and (not windows or windows[-1] != tail):
        windows.append(tail)
    return windows


def load_target_rows(root: Path):
    rows = []
    seen = set()
    for fasta_path in sorted(root.glob("*/*/*.fasta")):
        genus = fasta_path.parts[-3]
        species = fasta_path.parts[-2]
        for header, raw in read_fasta(fasta_path):
            header_l = header.lower()
            if any(token in f" {header_l} " for token in BAD_HEADER_PATTERNS):
                continue
            if species == "SARS-CoV" and (
                "sars-cov-2" in header_l or "coronavirus 2" in header_l
            ):
                continue
            seq = clean_sequence(raw)
            if len(seq) < 200:
                continue
            key = (genus, seq)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "genus": genus,
                    "species": species,
                    "header": header,
                    "sequence": seq,
                    "path": str(fasta_path),
                }
            )
    return rows


def build_training_examples(rows: list[dict], label_field: str = "genus") -> tuple[list[str], list[str]]:
    X = []
    y = []
    seen_examples = set()
    for row in rows:
        label = row[label_field]
        seq = row["sequence"]
        candidates = [seq]
        candidates.extend(split_windows(seq))
        for item in candidates:
            key = (label, item)
            if key in seen_examples:
                continue
            seen_examples.add(key)
            X.append(item)
            y.append(label)
    return X, y


def build_lr_model(config: dict):
    return make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=tuple(config["kmer_range"]),
            lowercase=False,
            min_df=2,
            max_features=int(config["max_features"]),
            sublinear_tf=True,
            dtype=np.float32,
        ),
        OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                C=2.0,
                tol=0.001,
                class_weight="balanced",
                random_state=42,
            )
        ),
    )


def train_label_model(
    rows: list[dict],
    target: str,
    label_name: str,
    label_field: str,
    config: dict,
) -> dict:
    X, y = build_training_examples(rows, label_field=label_field)
    class_counts = Counter(row[label_field] for row in rows)
    if len(class_counts) < 2:
        raise RuntimeError(f"Need at least 2 classes for {target} {label_name}, found {sorted(class_counts)}")

    model = build_lr_model(config)
    model.fit(X, y)

    best_path = MODEL_DIR / f"best_{target}_{label_name}_kmer_lr_app.joblib"
    latest_path = MODEL_DIR / f"latest_{target}_{label_name}_kmer_lr_app.joblib"
    dump(model, best_path)
    dump(model, latest_path)

    summary = {
        "target": target,
        "label_name": label_name,
        "model_paths": [str(best_path), str(latest_path)],
        "source_sequences": len(rows),
        "training_examples": len(X),
        "classes": len(class_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "kmer_range": list(config["kmer_range"]),
        "max_features": int(config["max_features"]),
        "class_weight": "balanced",
        "filtered_header_patterns": list(BAD_HEADER_PATTERNS),
    }
    (MODEL_DIR / f"{target}_{label_name}_app_model_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def train_target(target: str, config: dict) -> list[dict]:
    root = config["root"]
    rows = load_target_rows(root)
    if not rows:
        raise RuntimeError(f"No training rows found under {root}")

    for row in rows:
        row["family"] = RNA_GENUS_TO_FAMILY.get(row["genus"], row["genus"])

    summaries = []
    genus_summary = train_label_model(rows, target, "genus", "genus", config)
    genus_summary["root"] = str(root)
    genus_summary["genera"] = genus_summary["classes"]
    (MODEL_DIR / f"{target}_app_model_summary.json").write_text(
        json.dumps(genus_summary, indent=2),
        encoding="utf-8",
    )
    summaries.append(genus_summary)

    if target == "rna":
        family_summary = train_label_model(rows, target, "family", "family", RNA_FAMILY_CONFIG)
        family_summary["root"] = str(root)
        family_summary["families"] = family_summary["classes"]
        summaries.append(family_summary)

    return summaries


def main() -> None:
    summaries = []
    for target, config in TARGETS.items():
        print(f"Training {target} app model from {config['root']}", flush=True)
        target_summaries = train_target(target, config)
        summaries.extend(target_summaries)
        summary = target_summaries[0]
        print(
            f"  saved {summary['classes']} genus classes, "
            f"{summary['source_sequences']} source sequences, "
            f"{summary['training_examples']} training examples",
            flush=True,
        )
        for extra in target_summaries[1:]:
            print(
                f"  saved {extra['classes']} {extra['label_name']} classes, "
                f"{extra['training_examples']} training examples",
                flush=True,
            )

    (MODEL_DIR / "app_model_summaries.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
