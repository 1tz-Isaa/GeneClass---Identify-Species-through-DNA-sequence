"""Train a kingdom router model (Bacteria/Fungi/Viruses) from existing FASTA dataset.

Run:
  python3 Train_Kingdom_Router.py

Outputs:
  - runs/saved_models/kingdom_router_kmer_lr.joblib
  - runs/training_logs/kingdom_router_<run_id>/summary.json
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from dataset_loader import load_dataset
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def clean_sequence(seq: str) -> str:
    return re.sub(r"[^A-Za-z]", "", seq).upper()


def build_model(
    kmin: int,
    kmax: int,
    min_df: int,
    max_features: int,
    c_value: float,
    lr_max_iter: int,
    lr_solver: str,
    lr_class_weight: str | None,
):
    return make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(kmin, kmax),
            lowercase=False,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
        ),
        LogisticRegression(
            max_iter=lr_max_iter,
            solver=lr_solver,
            C=c_value,
            class_weight=lr_class_weight,
            n_jobs=None,
        ),
    )


def main() -> None:
    started = perf_counter()
    random_state = int(os.getenv("RANDOM_STATE", "42"))
    test_size = float(os.getenv("TEST_SIZE", "0.25"))
    show_progress = os.getenv("SHOW_PROGRESS", "1") == "1"
    show_file_progress = os.getenv("SHOW_FILE_PROGRESS", "0") == "1"

    kmer_min = int(os.getenv("KMER_MIN", "4"))
    kmer_max = int(os.getenv("KMER_MAX", "6"))
    kmer_min_df = int(os.getenv("KMER_MIN_DF", "2"))
    kmer_max_features = int(os.getenv("KMER_MAX_FEATURES", "120000"))
    lr_c = float(os.getenv("LR_C", "4.0"))
    lr_max_iter = int(os.getenv("LR_MAX_ITER", "12000"))
    lr_solver = os.getenv("LR_SOLVER", "saga").strip().lower()
    lr_class_weight_env = os.getenv("LR_CLASS_WEIGHT", "balanced").strip()
    lr_class_weight = lr_class_weight_env if lr_class_weight_env else None

    runs_root = Path(os.getenv("RUNS_ROOT", "runs/training_logs"))
    model_root = Path(os.getenv("MODEL_STORE_ROOT", "runs/saved_models"))

    def log(msg: str) -> None:
        if show_progress:
            print(msg, flush=True)

    log(
        "[ROUTER] Start kingdom training "
        f"(k={kmer_min}-{kmer_max}, max_features={kmer_max_features}, "
        f"solver={lr_solver}, max_iter={lr_max_iter})"
    )

    rows = []
    per_root_counts = {}
    for root in ("DNA", "RNA"):
        before = len(rows)
        log(f"[ROUTER] Loading dataset root={root} ...")
        for item in load_dataset(root, show_progress=show_file_progress):
            seq = clean_sequence(item["sequence"])
            kingdom = item["kingdom"]
            if not seq or kingdom not in {"Bacteria", "Fungi", "Viruses"}:
                continue
            rows.append((seq, kingdom))
        per_root_counts[root] = len(rows) - before
        log(f"[ROUTER] Loaded root={root}: {per_root_counts[root]} usable records")

    # Dedup exact sequence + label.
    dedup = []
    seen = set()
    for seq, label in rows:
        key = (seq, label)
        if key in seen:
            continue
        seen.add(key)
        dedup.append((seq, label))

    rows = dedup
    log(f"[ROUTER] Dedup done: {len(rows)} records")
    if len(rows) < 30:
        raise ValueError("Not enough data to train kingdom router")

    x = [r[0] for r in rows]
    y = [r[1] for r in rows]

    counts = Counter(y)
    log(f"[ROUTER] Class counts: {dict(counts)}")
    if len(counts) < 3:
        raise ValueError(f"Need 3 kingdom classes, found: {sorted(counts)}")

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    log(f"[ROUTER] Split: train={len(x_train)} val={len(x_val)} test_size={test_size}")

    model = build_model(
        kmer_min,
        kmer_max,
        kmer_min_df,
        kmer_max_features,
        lr_c,
        lr_max_iter,
        lr_solver,
        lr_class_weight,
    )
    log("[ROUTER] Fitting model ...")
    model.fit(x_train, y_train)
    log("[ROUTER] Evaluating ...")

    y_pred = model.predict(x_val)
    acc = float(accuracy_score(y_val, y_pred))
    bal = float(balanced_accuracy_score(y_val, y_pred))
    rep = classification_report(y_val, y_pred, zero_division=0)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"kingdom_router_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "samples_total": len(rows),
        "class_counts": dict(counts),
        "n_train": len(x_train),
        "n_val": len(x_val),
        "metrics": {
            "accuracy": acc,
            "balanced_accuracy": bal,
        },
        "model": {
            "kmer_min": kmer_min,
            "kmer_max": kmer_max,
            "kmer_min_df": kmer_min_df,
            "kmer_max_features": kmer_max_features,
            "lr_c": lr_c,
            "lr_max_iter": lr_max_iter,
            "lr_solver": lr_solver,
            "lr_class_weight": lr_class_weight,
        },
        "duration_seconds": perf_counter() - started,
    }

    (run_dir / "classification_report.txt").write_text(rep, encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    model_root.mkdir(parents=True, exist_ok=True)
    model_path = model_root / "kingdom_router_kmer_lr.joblib"
    dump(model, model_path)

    print("Kingdom router trained.", flush=True)
    print(f"Accuracy: {acc:.4f}", flush=True)
    print(f"Balanced accuracy: {bal:.4f}", flush=True)
    print(f"Saved model: {model_path.resolve()}", flush=True)
    print(f"Run artifacts: {run_dir.resolve()}", flush=True)
    print(f"Duration: {summary['duration_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
