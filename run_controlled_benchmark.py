
from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import os
import re
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, matthews_corrcoef

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE = Path(__file__).resolve().parent
OUT_DIR = Path(os.getenv("BENCHMARK_OUT_DIR", BASE / "runs" / "controlled_benchmark_10_trials"))
if not OUT_DIR.is_absolute():
    OUT_DIR = BASE / OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
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

def env_root(name: str, default: Path) -> Path:
    value = os.getenv(name)
    root = Path(value) if value else default
    return root if root.is_absolute() else BASE / root

ROOTS = {
    "bacteria": env_root("BACTERIA_BENCHMARK_ROOT", BASE / "Database" / "bacteria_genus"),
    "fungi": env_root("FUNGI_BENCHMARK_ROOT", BASE / "Database" / "fungi_genus"),
    "virus": env_root("VIRUS_BENCHMARK_ROOT", BASE / "Database" / "rna_genus"),
}

TRIALS = [
    {"trial": 1, "kmer_min": 4, "kmer_max": 6, "fragment_len": 1200, "class_weight": None, "max_features": 50000, "min_samples": 0, "description": "4-6 k-mers, 1200 fragment, no class weighting"},
    {"trial": 2, "kmer_min": 5, "kmer_max": 5, "fragment_len": 1200, "class_weight": None, "max_features": 50000, "min_samples": 0, "description": "Fixed 5-mer, 1200 fragment, no class weighting"},
    {"trial": 3, "kmer_min": 4, "kmer_max": 6, "fragment_len": 1200, "class_weight": "balanced", "max_features": 50000, "min_samples": 0, "description": "4-6 k-mers, 1200 fragment, balanced class weighting"},
    {"trial": 4, "kmer_min": 5, "kmer_max": 5, "fragment_len": 1200, "class_weight": "balanced", "max_features": 50000, "min_samples": 0, "description": "Fixed 5-mer, 1200 fragment, balanced class weighting"},
    {"trial": 5, "kmer_min": 4, "kmer_max": 6, "fragment_len": 1400, "class_weight": "balanced", "max_features": 50000, "min_samples": 0, "description": "4-6 k-mers, 1400 fragment, balanced class weighting"},
    {"trial": 6, "kmer_min": 5, "kmer_max": 5, "fragment_len": 1400, "class_weight": "balanced", "max_features": 50000, "min_samples": 0, "description": "Fixed 5-mer, 1400 fragment, balanced class weighting"},
    {"trial": 7, "kmer_min": 5, "kmer_max": 8, "fragment_len": 1200, "class_weight": "balanced", "max_features": 12000, "min_samples": 0, "description": "5-8 k-mers, 1200 fragment, balanced class weighting with compact feature cap"},
    {"trial": 8, "kmer_min": 4, "kmer_max": 6, "fragment_len": 1200, "class_weight": "balanced", "max_features": 25000, "min_samples": 0, "description": "4-6 k-mers, 1200 fragment, lower TF-IDF feature limit"},
    {"trial": 9, "kmer_min": 5, "kmer_max": 5, "fragment_len": 1200, "class_weight": "balanced", "max_features": 25000, "min_samples": 0, "description": "Fixed 5-mer, 1200 fragment, lower TF-IDF feature limit"},
    {"trial": 10, "kmer_min": 5, "kmer_max": 5, "fragment_len": 1200, "class_weight": "balanced", "max_features": 50000, "min_samples": 0, "virus_min_samples": 30, "description": "Final filtered virus trial using fixed 5-mer features and a minimum 30 samples per genus"},
]

def read_fasta(path: Path):
    header = None
    seq = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq)
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header is not None:
            yield header, "".join(seq)

def clean_sequence(seq: str) -> str:
    # Retain A/C/G/T/U and N. N represents an ambiguous or unknown nucleotide.
    return re.sub(r"[^ACGTUNacgtun]", "", seq).upper()

def normalize_fragment(seq: str, length: int) -> str:
    if length <= 0:
        return seq
    if len(seq) > length:
        start = (len(seq) - length) // 2
        return seq[start:start + length]
    return seq + ("N" * (length - len(seq)))

def load_rows(group: str, trial: dict):
    root = ROOTS[group]
    rows = []
    for fp in sorted(root.glob("*/*/*.fasta")):
        genus = fp.parts[-3]
        species = fp.parts[-2]
        species_group = f"{genus}/{species}"
        for header, raw in read_fasta(fp):
            header_l = header.lower()
            if any(token in f" {header_l} " for token in BAD_HEADER_PATTERNS):
                continue
            if species == "SARS-CoV" and (
                "sars-cov-2" in header_l or "coronavirus 2" in header_l
            ):
                continue
            raw_clean = clean_sequence(raw)
            if len(raw_clean) < 200:
                continue
            rows.append((normalize_fragment(raw_clean, int(trial["fragment_len"])), genus, species_group, raw_clean))

    deduped = []
    seen = set()
    for seq, label, species_group, raw in rows:
        key = (raw, label)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((seq, label, species_group, raw))
    rows = deduped

    min_samples = int(trial.get("virus_min_samples", trial.get("min_samples", 0)) if group == "virus" else trial.get("min_samples", 0))
    dropped = {}
    if min_samples > 0:
        counts = Counter(label for _, label, _, _ in rows)
        dropped = {label: int(count) for label, count in counts.items() if count < min_samples}
        rows = [row for row in rows if row[1] not in dropped]
    return rows, dropped

def choose_group_split(labels, groups, test_size=0.3, random_state=42, tries=90):
    idx = np.arange(len(labels))
    total_classes = len(set(labels))
    target_val = round(len(labels) * test_size)
    splitter = GroupShuffleSplit(n_splits=tries, test_size=test_size, random_state=random_state)
    best = None
    best_coverage = -1
    best_balance = 10**9
    for train_idx, val_idx in splitter.split(idx, labels, groups=groups):
        coverage = len({labels[i] for i in val_idx})
        balance = abs(len(val_idx) - target_val)
        if coverage > best_coverage or (coverage == best_coverage and balance < best_balance):
            best = (train_idx, val_idx)
            best_coverage = coverage
            best_balance = balance
        if coverage == total_classes:
            break
    return best

def compute_metrics(y_true, y_pred):
    eval_labels = sorted(set(y_true) | set(y_pred))
    _, _, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=eval_labels, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f_macro),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

def run_trial(group: str, trial: dict):
    rows, dropped = load_rows(group, trial)
    if len(rows) < 2 or len({row[1] for row in rows}) < 2:
        raise RuntimeError(f"Not enough data for {group} trial {trial['trial']}")
    X = [row[0] for row in rows]
    y = [row[1] for row in rows]
    species_groups = [row[2] for row in rows]
    train_idx, val_idx = choose_group_split(y, species_groups)

    model = make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(int(trial["kmer_min"]), int(trial["kmer_max"])),
            lowercase=False,
            min_df=2,
            max_features=int(trial["max_features"]),
            sublinear_tf=True,
            dtype=np.float32,
        ),
        LogisticRegression(
            max_iter=5000,
            solver="saga",
            C=2.0,
            tol=0.001,
            class_weight=trial["class_weight"],
            random_state=42,
        ),
    )
    model.fit([X[i] for i in train_idx], [y[i] for i in train_idx])
    y_true = [y[i] for i in val_idx]
    y_pred = list(model.predict([X[i] for i in val_idx]))
    metrics = compute_metrics(y_true, y_pred)
    all_genera = set(y)
    val_genera = set(y_true)
    return {
        "group": group,
        "trial": int(trial["trial"]),
        "description": trial["description"],
        "kmer_range": f"{trial['kmer_min']}-{trial['kmer_max']}",
        "fragment_length": int(trial["fragment_len"]),
        "class_weight": trial["class_weight"] or "none",
        "max_features": int(trial["max_features"]),
        "min_samples_per_genus": int(trial.get("virus_min_samples", trial.get("min_samples", 0)) if group == "virus" else trial.get("min_samples", 0)),
        "samples": len(y),
        "genera": len(all_genera),
        "train_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "validation_genera_present": len(val_genera),
        "validation_genera_total": len(all_genera),
        "validation_coverage": f"{len(val_genera)}/{len(all_genera)}",
        "groups_total": len(set(species_groups)),
        "groups_train": len({species_groups[i] for i in train_idx}),
        "groups_validation": len({species_groups[i] for i in val_idx}),
        "dropped_low_sample_genera": json.dumps(dropped, sort_keys=True),
        **metrics,
    }

def main():
    rows = []
    groups = [g.strip() for g in os.getenv("BENCHMARK_GROUPS", "bacteria,fungi,virus").split(",") if g.strip()]
    for group in groups:
        print(f"Running {group}...", flush=True)
        for trial in TRIALS:
            result = run_trial(group, trial)
            rows.append(result)
            print(
                f"  trial {result['trial']}: acc={result['accuracy']:.3f} "
                f"f1={result['f1_macro']:.3f} mcc={result['mcc']:.3f} "
                f"n={result['samples']} genera={result['genera']} coverage={result['validation_coverage']}"
                ,
                flush=True,
            )

    fields = [
        "group", "trial", "description", "kmer_range", "fragment_length", "class_weight", "max_features",
        "min_samples_per_genus", "samples", "genera", "train_samples", "validation_samples",
        "validation_coverage", "validation_genera_present", "validation_genera_total",
        "groups_total", "groups_train", "groups_validation", "accuracy", "balanced_accuracy", "f1_macro", "mcc",
        "dropped_low_sample_genera",
    ]
    csv_path = OUT_DIR / "controlled_10_trial_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "controlled_10_trial_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    best = []
    for group in groups:
        candidates = [row for row in rows if row["group"] == group]
        chosen = sorted(candidates, key=lambda r: (r["f1_macro"], r["mcc"], r["accuracy"]), reverse=True)[0]
        best.append(chosen)
    best_path = OUT_DIR / "best_by_group.csv"
    with best_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(best)
    print(f"Wrote {csv_path}")
    print(f"Wrote {best_path}")

if __name__ == "__main__":
    main()
