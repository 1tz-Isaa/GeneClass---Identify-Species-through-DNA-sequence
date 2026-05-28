from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import os
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
OUT = BASE / "runs" / "marker_hierarchical_experiments"
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
REPEATS = int(os.environ.get("MARKER_EXPERIMENT_REPEATS", "25"))

VIRUS_FAMILY = {
    "Alphacoronavirus": "Coronaviridae",
    "Betacoronavirus": "Coronaviridae",
    "Enterovirus": "Picornaviridae",
    "Rhinovirus": "Picornaviridae",
    "Hantavirus": "Hantaviridae",
    "Orthobunyavirus": "Peribunyaviridae",
    "Arenavirus": "Arenaviridae",
    "Influenzavirus": "Orthomyxoviridae",
    "Metapneumovirus": "Pneumoviridae",
    "Orthopneumovirus": "Pneumoviridae",
    "Reovirus": "Reoviridae",
    "Respirovirus": "Paramyxoviridae",
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


def fungi_marker(header: str) -> str:
    h = header.lower()
    has_its = (
        any(x in h for x in ["internal transcribed spacer", "its1", "its2", "5.8s", " rdna "])
        or bool(re.search(r"\bits\b", h))
    )
    has_18 = any(x in h for x in ["18s", "small subunit", "ssu"])
    has_28 = any(x in h for x in ["28s", "large subunit", "lsu"])
    if has_its and (has_18 or has_28):
        return "mixed_its_rRNA"
    if has_its:
        return "ITS"
    if has_18 or has_28:
        return "18S_28S"
    return "unknown"


def load_rows(root: Path, target: str):
    rows = []
    seen = set()
    for fp in sorted(root.glob("*/*/*.fasta")):
        genus = fp.parts[-3]
        species = fp.parts[-2]
        group = f"{genus}/{species}"
        for header, raw in read_fasta(fp):
            if any(token in f" {header.lower()} " for token in BAD_HEADER_PATTERNS):
                continue
            seq = clean_sequence(raw)
            if len(seq) < 200:
                continue
            key = (genus, seq)
            if key in seen:
                continue
            seen.add(key)
            row = {
                "path": str(fp),
                "header": header,
                "seq": seq,
                "genus": genus,
                "species_group": group,
            }
            if target == "fungi":
                row["marker"] = fungi_marker(header)
            elif target == "virus":
                row["family"] = VIRUS_FAMILY[genus]
            rows.append(row)
    return rows


def choose_group_split(labels, groups, test_size: float, random_state: int, tries: int = 250):
    idx = np.arange(len(labels))
    total_classes = len(set(labels))
    target_n = round(len(labels) * test_size)
    best = None
    best_key = None
    for seed in range(random_state, random_state + 7):
        splitter = GroupShuffleSplit(n_splits=tries, test_size=test_size, random_state=seed)
        for train_idx, val_idx in splitter.split(idx, labels, groups=groups):
            val_classes = {labels[i] for i in val_idx}
            train_classes = {labels[i] for i in train_idx}
            balance = abs(len(val_idx) - target_n)
            min_val_class = min(Counter(labels[i] for i in val_idx).values()) if len(val_idx) else 0
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


def build_model(config):
    if config.get("estimator") == "linear_svc":
        classifier = LinearSVC(
            C=float(config.get("C", 1.0)),
            class_weight=config.get("class_weight", "balanced"),
            max_iter=5000,
            random_state=42,
        )
    else:
        classifier = OneVsRestClassifier(
            LogisticRegression(
                max_iter=1200,
                solver="liblinear",
                C=3.0,
                tol=0.001,
                class_weight=config.get("class_weight", "balanced"),
                random_state=42,
            )
        )
    return make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(int(config["kmer_min"]), int(config["kmer_max"])),
            lowercase=False,
            min_df=int(config.get("min_df", 2)),
            max_features=int(config["max_features"]),
            sublinear_tf=True,
            dtype=np.float32,
        ),
        classifier,
    )


def seq_value(row, marker_prefix=False):
    if marker_prefix:
        return f"MARKER_{row.get('marker', 'unknown')}_{row['seq']}"
    return row["seq"]


def configured_seq(row, config):
    seq = row["seq"]
    fragment_len = int(config.get("fragment_len", 0))
    if fragment_len > 0:
        return seq[:fragment_len]
    return seq


def row_subset(rows, idxs):
    return [rows[int(i)] for i in idxs]


def run_fungi_marker():
    rows = load_rows(BASE / "Database" / "fungi_genus", "fungi")
    labels = [r["genus"] for r in rows]
    groups = [r["species_group"] for r in rows]
    config = {"kmer_min": 5, "kmer_max": 8, "max_features": 25000, "class_weight": "balanced"}
    output_rows = []
    modes = ["global", "marker_prefix", "marker_specific"]

    for seed in range(100, 100 + REPEATS):
        train_idx, val_idx = choose_group_split(labels, groups, test_size=0.30, random_state=seed)
        train_rows = row_subset(rows, train_idx)
        val_rows = row_subset(rows, val_idx)
        y_true = [r["genus"] for r in val_rows]

        for mode in modes:
            if mode in {"global", "marker_prefix"}:
                marker_prefix = mode == "marker_prefix"
                model = build_model(config)
                model.fit([seq_value(r, marker_prefix) for r in train_rows], [r["genus"] for r in train_rows])
                pred = list(model.predict([seq_value(r, marker_prefix) for r in val_rows]))
            else:
                global_model = build_model(config)
                global_model.fit([r["seq"] for r in train_rows], [r["genus"] for r in train_rows])
                marker_models = {}
                for marker, marker_train in groupby_marker(train_rows).items():
                    if len({r["genus"] for r in marker_train}) >= 2 and len(marker_train) >= 8:
                        m = build_model(config)
                        m.fit([r["seq"] for r in marker_train], [r["genus"] for r in marker_train])
                        marker_models[marker] = m
                pred = []
                for r in val_rows:
                    model = marker_models.get(r["marker"], global_model)
                    pred.append(str(model.predict([r["seq"]])[0]))

            m = metric_dict(y_true, pred)
            output_rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "samples": len(rows),
                    "genera": len(set(labels)),
                    "validation_samples": len(val_rows),
                    "validation_genera": len(set(y_true)),
                    **m,
                }
            )

    path = OUT / "fungi_marker_experiment.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    summary = summarize_by_mode(output_rows, "mode")
    (OUT / "fungi_marker_experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[fungi marker] marker counts:", dict(Counter(r["marker"] for r in rows)))
    for mode, item in summary.items():
        print(
            f"[fungi marker] {mode}: acc={item['accuracy']['mean']:.3f} "
            f"f1={item['f1_macro']['mean']:.3f} mcc={item['mcc']['mean']:.3f}"
        )
    print(f"[fungi marker] wrote {path}")


def groupby_marker(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["marker"]].append(r)
    return out


def summarize_by_mode(rows, key_name):
    summary = {}
    for key in sorted({r[key_name] for r in rows}):
        subset = [r for r in rows if r[key_name] == key]
        summary[key] = {}
        for metric in ["accuracy", "f1_macro", "mcc"]:
            vals = [float(r[metric]) for r in subset]
            summary[key][metric] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    return summary


def run_virus_hierarchical():
    rows = load_rows(BASE / "Database" / "rna_genus", "virus")
    labels = [r["genus"] for r in rows]
    groups = [r["species_group"] for r in rows]
    configs = [
        {
            "name": "svc_1200bp_5_7_30k",
            "kmer_min": 5,
            "kmer_max": 7,
            "max_features": 30000,
            "class_weight": "balanced",
            "estimator": "linear_svc",
            "fragment_len": 1200,
        },
        {
            "name": "svc_1200bp_6_6_30k",
            "kmer_min": 6,
            "kmer_max": 6,
            "max_features": 30000,
            "class_weight": "balanced",
            "estimator": "linear_svc",
            "fragment_len": 1200,
        },
        {
            "name": "svc_2000bp_6_8_30k",
            "kmer_min": 6,
            "kmer_max": 8,
            "max_features": 30000,
            "class_weight": "balanced",
            "estimator": "linear_svc",
            "fragment_len": 2000,
        },
    ]
    output_rows = []

    for cfg in configs:
        for seed in range(100, 100 + REPEATS):
            train_idx, val_idx = choose_group_split(labels, groups, test_size=0.30, random_state=seed)
            train_rows = row_subset(rows, train_idx)
            val_rows = row_subset(rows, val_idx)

            family_model = build_model(cfg)
            family_model.fit([configured_seq(r, cfg) for r in train_rows], [r["family"] for r in train_rows])
            true_family = [r["family"] for r in val_rows]
            pred_family = list(family_model.predict([configured_seq(r, cfg) for r in val_rows]))

            family_genus_models = {}
            family_majority = {}
            for family in sorted({r["family"] for r in train_rows}):
                subset = [r for r in train_rows if r["family"] == family]
                family_majority[family] = Counter(r["genus"] for r in subset).most_common(1)[0][0]
                if len({r["genus"] for r in subset}) >= 2:
                    m = build_model(cfg)
                    m.fit([configured_seq(r, cfg) for r in subset], [r["genus"] for r in subset])
                    family_genus_models[family] = m

            pred_genus = []
            for r, pf in zip(val_rows, pred_family):
                if pf in family_genus_models:
                    pred_genus.append(str(family_genus_models[pf].predict([configured_seq(r, cfg)])[0]))
                else:
                    pred_genus.append(family_majority.get(pf, Counter(x["genus"] for x in train_rows).most_common(1)[0][0]))

            genus_metrics = metric_dict([r["genus"] for r in val_rows], pred_genus)
            family_metrics = metric_dict(true_family, pred_family)
            output_rows.append(
                {
                    "seed": seed,
                    "config": cfg["name"],
                    "samples": len(rows),
                    "genera": len(set(labels)),
                    "families": len(set(r["family"] for r in rows)),
                    "validation_samples": len(val_rows),
                    "validation_genera": len(set(r["genus"] for r in val_rows)),
                    "family_accuracy": family_metrics["accuracy"],
                    "family_f1_macro": family_metrics["f1_macro"],
                    "family_mcc": family_metrics["mcc"],
                    "genus_accuracy": genus_metrics["accuracy"],
                    "genus_f1_macro": genus_metrics["f1_macro"],
                    "genus_mcc": genus_metrics["mcc"],
                }
            )

    path = OUT / "virus_hierarchical_experiment.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {}
    for cfg_name in sorted({r["config"] for r in output_rows}):
        subset = [r for r in output_rows if r["config"] == cfg_name]
        summary[cfg_name] = {}
        for metric in ["family_accuracy", "family_f1_macro", "family_mcc", "genus_accuracy", "genus_f1_macro", "genus_mcc"]:
            vals = [float(r[metric]) for r in subset]
            summary[cfg_name][metric] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    (OUT / "virus_hierarchical_experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for cfg_name, item in summary.items():
        print(
            f"[virus hierarchical] {cfg_name}: family_acc={item['family_accuracy']['mean']:.3f} "
            f"family_f1={item['family_f1_macro']['mean']:.3f} genus_acc={item['genus_accuracy']['mean']:.3f} "
            f"genus_f1={item['genus_f1_macro']['mean']:.3f} genus_mcc={item['genus_mcc']['mean']:.3f}"
        )
    print(f"[virus hierarchical] wrote {path}")


def main():
    targets = {x.strip().lower() for x in os.environ.get("MARKER_EXPERIMENT_TARGETS", "fungi,virus").split(",")}
    if "fungi" in targets:
        run_fungi_marker()
    if "virus" in targets:
        run_virus_hierarchical()


if __name__ == "__main__":
    main()
