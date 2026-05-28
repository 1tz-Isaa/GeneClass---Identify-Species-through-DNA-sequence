from __future__ import annotations

from collections import Counter
from pathlib import Path
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
from sklearn.svm import LinearSVC


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

BASE = Path(__file__).resolve().parent
OUT = BASE / "runs" / "virus_deep_model_search"
OUT.mkdir(parents=True, exist_ok=True)

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


def center_fragment(seq: str, length: int) -> str:
    if length <= 0 or len(seq) <= length:
        return seq
    start = (len(seq) - length) // 2
    return seq[start : start + length]


def segment_tag(header: str) -> str:
    h = header.lower()
    m = re.search(r"\bsegment\s*[:=]?\s*([0-9]+|[a-z])\b", h)
    if m:
        return f"segment_{m.group(1).lower()}"
    for token in ["pb2", "pb1", "pa", "ha", "np", "na", "m1", "m2", "ns1", "ns2"]:
        if re.search(rf"\b{re.escape(token)}\b", h):
            return f"gene_{token}"
    for token in ["nucleoprotein", "polymerase", "glycoprotein", "matrix", "hemagglutinin", "neuraminidase"]:
        if token in h:
            return "gene_" + token.replace(" ", "_")
    return "segment_unknown"


def length_tag(seq: str) -> str:
    n = len(seq)
    if n < 1000:
        return "len_short"
    if n < 3000:
        return "len_segment"
    if n < 10000:
        return "len_mid"
    return "len_genome"


def load_rows():
    rows = []
    seen = set()
    root = BASE / "Database" / "rna_genus"
    for fp in sorted(root.glob("*/*/*.fasta")):
        genus = fp.parts[-3]
        species = fp.parts[-2]
        group = f"{genus}/{species}"
        for header, raw in read_fasta(fp):
            header_l = header.lower()
            if any(token in f" {header_l} " for token in BAD_HEADER_PATTERNS):
                continue
            if species == "SARS-CoV" and ("sars-cov-2" in header_l or "coronavirus 2" in header_l):
                continue
            seq = clean_sequence(raw)
            if len(seq) < 300:
                continue
            key = (genus, seq)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "genus": genus,
                    "species_group": group,
                    "seq": seq,
                    "header": header,
                    "segment": segment_tag(header),
                    "length_tag": length_tag(seq),
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
            balance = abs(len(val_idx) - target_n)
            val_counts = Counter(labels[i] for i in val_idx)
            min_val_class = min(val_counts.values()) if val_counts else 0
            key = (len(val_classes), len(train_classes), min_val_class, -balance)
            if best_key is None or key > best_key:
                best = (train_idx, val_idx)
                best_key = key
            if len(val_classes) == total_classes and len(train_classes) == total_classes and balance <= 3:
                return train_idx, val_idx
    if best is None:
        raise RuntimeError("Could not create group split")
    return best


def metric_dict(y_true, y_pred):
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


def feature_text(row: dict, config: dict) -> str:
    seq = center_fragment(row["seq"], int(config["fragment_len"]))
    mode = config["feature_mode"]
    if mode == "raw":
        return seq
    if mode == "segment_prefix":
        return f"{row['segment']}|{seq}"
    if mode == "length_segment_prefix":
        return f"{row['length_tag']}|{row['segment']}|{seq}"
    raise ValueError(f"Unknown feature_mode {mode}")


def build_model(config: dict):
    if config["estimator"] == "linear_svc":
        clf = LinearSVC(C=float(config["C"]), class_weight=config["class_weight"], max_iter=8000, random_state=42)
    else:
        clf = OneVsRestClassifier(
            LogisticRegression(
                max_iter=1600,
                solver="liblinear",
                C=float(config["C"]),
                tol=0.001,
                class_weight=config["class_weight"],
                random_state=42,
            )
        )
    return make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(int(config["kmer_min"]), int(config["kmer_max"])),
            lowercase=False,
            min_df=int(config["min_df"]),
            max_features=int(config["max_features"]),
            sublinear_tf=True,
            dtype=np.float32,
        ),
        clf,
    )


def row_subset(rows, idxs):
    return [rows[int(i)] for i in idxs]


def configs():
    for estimator in ["lr", "linear_svc"]:
        for feature_mode in ["raw", "segment_prefix", "length_segment_prefix"]:
            for fragment_len in [0, 2500, 5000]:
                for kmer_min, kmer_max in [(5, 5), (5, 7), (6, 8)]:
                    yield {
                        "estimator": estimator,
                        "feature_mode": feature_mode,
                        "fragment_len": fragment_len,
                        "kmer_min": kmer_min,
                        "kmer_max": kmer_max,
                        "min_df": 2,
                        "max_features": 60000,
                        "class_weight": "balanced",
                        "C": 1.0 if estimator == "linear_svc" else 3.0,
                    }


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for name, vals in rows_by_config(rows).items():
        summary[name] = {}
        for metric in ["accuracy", "balanced_accuracy", "f1_macro", "mcc"]:
            arr = [float(v[metric]) for v in vals]
            summary[name][metric] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
    return summary


def rows_by_config(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(str(row["config_name"]), []).append(row)
    return out


def config_name(config: dict) -> str:
    return (
        f"{config['estimator']}_{config['feature_mode']}_"
        f"{config['fragment_len']}bp_{config['kmer_min']}_{config['kmer_max']}"
    )


def main() -> None:
    rows = load_rows()
    labels = [r["genus"] for r in rows]
    groups = [r["species_group"] for r in rows]
    cfgs = list(configs())
    output_rows = []
    print(f"[virus deep] rows={len(rows)} genera={len(set(labels))} configs={len(cfgs)}")
    for cfg_i, cfg in enumerate(cfgs, start=1):
        name = config_name(cfg)
        for seed in range(500, 515):
            train_idx, val_idx = choose_group_split(labels, groups, test_size=0.30, random_state=seed)
            train_rows = row_subset(rows, train_idx)
            val_rows = row_subset(rows, val_idx)
            model = build_model(cfg)
            model.fit([feature_text(r, cfg) for r in train_rows], [r["genus"] for r in train_rows])
            y_true = [r["genus"] for r in val_rows]
            y_pred = list(model.predict([feature_text(r, cfg) for r in val_rows]))
            m = metric_dict(y_true, y_pred)
            output_rows.append(
                {
                    "config_name": name,
                    "seed": seed,
                    "n_train": len(train_rows),
                    "n_val": len(val_rows),
                    **cfg,
                    **m,
                }
            )
        current = summarize([r for r in output_rows if r["config_name"] == name])[name]
        print(
            f"[virus deep] {cfg_i}/{len(cfgs)} {name}: "
            f"acc={current['accuracy']['mean']:.3f} f1={current['f1_macro']['mean']:.3f} "
            f"mcc={current['mcc']['mean']:.3f}",
            flush=True,
        )

    csv_path = OUT / "virus_deep_model_search.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    summary = summarize(output_rows)
    ranked = sorted(
        summary.items(),
        key=lambda item: (
            item[1]["f1_macro"]["mean"],
            item[1]["mcc"]["mean"],
            item[1]["accuracy"]["mean"],
        ),
        reverse=True,
    )
    (OUT / "virus_deep_model_search_summary.json").write_text(
        json.dumps({"ranked": ranked, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print("[virus deep] top configs:")
    for name, vals in ranked[:8]:
        print(
            f"  {name}: acc={vals['accuracy']['mean']:.3f} "
            f"f1={vals['f1_macro']['mean']:.3f} mcc={vals['mcc']['mean']:.3f}"
        )
    print(f"[virus deep] wrote {csv_path}")


if __name__ == "__main__":
    main()
