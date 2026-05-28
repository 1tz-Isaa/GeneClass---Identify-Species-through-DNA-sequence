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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

BASE = Path(__file__).resolve().parent
OUT = BASE / "runs" / "virus_svc_focused_search"
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


def feature_text(row: dict, cfg: dict) -> str:
    seq = center_fragment(row["seq"], int(cfg["fragment_len"]))
    mode = cfg["feature_mode"]
    if mode == "raw":
        return seq
    if mode == "segment_prefix":
        return f"{row['segment']}|{seq}"
    if mode == "length_segment_prefix":
        return f"{row['length_tag']}|{row['segment']}|{seq}"
    raise ValueError(mode)


def build_model(cfg: dict):
    return make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(int(cfg["kmer_min"]), int(cfg["kmer_max"])),
            lowercase=False,
            min_df=int(cfg["min_df"]),
            max_features=int(cfg["max_features"]),
            sublinear_tf=True,
            dtype=np.float32,
        ),
        LinearSVC(
            C=float(cfg["C"]),
            class_weight=cfg["class_weight"],
            max_iter=10000,
            random_state=42,
        ),
    )


def candidate_configs():
    base = {
        "kmer_min": 5,
        "kmer_max": 5,
        "class_weight": "balanced",
    }
    raw = [
        ("raw", 0),
        ("raw", 5000),
        ("segment_prefix", 0),
        ("segment_prefix", 5000),
        ("length_segment_prefix", 0),
        ("length_segment_prefix", 5000),
        ("length_segment_prefix", 6000),
        ("length_segment_prefix", 8000),
    ]
    for feature_mode, fragment_len in raw:
        for C in [0.5, 1.0, 1.5, 2.0, 3.0]:
            for min_df in [1, 2, 3]:
                for max_features in [60000, 100000]:
                    yield {
                        **base,
                        "feature_mode": feature_mode,
                        "fragment_len": fragment_len,
                        "C": C,
                        "min_df": min_df,
                        "max_features": max_features,
                    }
    # A few targeted alternatives; wider k-mers were usually worse, so keep this small.
    for kmer_min, kmer_max in [(4, 5), (5, 6)]:
        for fragment_len in [0, 5000]:
            yield {
                **base,
                "feature_mode": "length_segment_prefix",
                "fragment_len": fragment_len,
                "kmer_min": kmer_min,
                "kmer_max": kmer_max,
                "C": 1.0,
                "min_df": 2,
                "max_features": 100000,
            }


def config_name(cfg: dict) -> str:
    return (
        f"svc_{cfg['feature_mode']}_{cfg['fragment_len']}bp_"
        f"{cfg['kmer_min']}_{cfg['kmer_max']}_"
        f"C{cfg['C']}_df{cfg['min_df']}_mf{cfg['max_features']}"
    )


def row_subset(rows, idxs):
    return [rows[int(i)] for i in idxs]


def summarize(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["config_name"]), []).append(row)
    summary = {}
    for name, vals in grouped.items():
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


def evaluate(rows, cfg, seeds):
    labels = [r["genus"] for r in rows]
    groups = [r["species_group"] for r in rows]
    output = []
    for seed in seeds:
        train_idx, val_idx = choose_group_split(labels, groups, test_size=0.30, random_state=seed)
        train_rows = row_subset(rows, train_idx)
        val_rows = row_subset(rows, val_idx)
        model = build_model(cfg)
        model.fit([feature_text(r, cfg) for r in train_rows], [r["genus"] for r in train_rows])
        y_true = [r["genus"] for r in val_rows]
        y_pred = list(model.predict([feature_text(r, cfg) for r in val_rows]))
        output.append(
            {
                "config_name": config_name(cfg),
                "seed": seed,
                "n_train": len(train_rows),
                "n_val": len(val_rows),
                **cfg,
                **metric_dict(y_true, y_pred),
            }
        )
    return output


def main() -> None:
    rows = load_rows()
    cfgs = list(candidate_configs())
    print(f"[virus focused] rows={len(rows)} configs={len(cfgs)}")

    quick_rows = []
    quick_seeds = range(700, 705)
    for i, cfg in enumerate(cfgs, start=1):
        quick_rows.extend(evaluate(rows, cfg, quick_seeds))
        if i % 20 == 0 or i == len(cfgs):
            ranked = rank_summary(summarize(quick_rows))[:3]
            best = ranked[0]
            print(
                f"[virus focused] quick {i}/{len(cfgs)} best={best[0]} "
                f"acc={best[1]['accuracy']['mean']:.3f} "
                f"f1={best[1]['f1_macro']['mean']:.3f} "
                f"mcc={best[1]['mcc']['mean']:.3f}",
                flush=True,
            )

    quick_summary = summarize(quick_rows)
    ranked_quick = rank_summary(quick_summary)
    top_names = {name for name, _ in ranked_quick[:12]}
    cfg_by_name = {config_name(cfg): cfg for cfg in cfgs}

    robust_rows = []
    robust_seeds = range(700, 725)
    for i, name in enumerate(sorted(top_names), start=1):
        cfg = cfg_by_name[name]
        robust_rows.extend(evaluate(rows, cfg, robust_seeds))
        best = rank_summary(summarize(robust_rows))[0]
        print(
            f"[virus focused] robust {i}/{len(top_names)} current_best={best[0]} "
            f"acc={best[1]['accuracy']['mean']:.3f} "
            f"f1={best[1]['f1_macro']['mean']:.3f} "
            f"mcc={best[1]['mcc']['mean']:.3f}",
            flush=True,
        )

    write_outputs("quick", quick_rows)
    write_outputs("robust", robust_rows)

    ranked = rank_summary(summarize(robust_rows))
    print("[virus focused] robust top configs:")
    for name, vals in ranked[:10]:
        print(
            f"  {name}: acc={vals['accuracy']['mean']:.3f} "
            f"f1={vals['f1_macro']['mean']:.3f} mcc={vals['mcc']['mean']:.3f}"
        )


def rank_summary(summary: dict) -> list[tuple[str, dict]]:
    return sorted(
        summary.items(),
        key=lambda item: (
            item[1]["f1_macro"]["mean"],
            item[1]["mcc"]["mean"],
            item[1]["accuracy"]["mean"],
        ),
        reverse=True,
    )


def write_outputs(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    csv_path = OUT / f"virus_svc_focused_{name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    ranked = rank_summary(summary)
    (OUT / f"virus_svc_focused_{name}_summary.json").write_text(
        json.dumps({"ranked": ranked, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(f"[virus focused] wrote {csv_path}")


if __name__ == "__main__":
    main()
