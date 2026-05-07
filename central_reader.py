"""Central kingdom router for shotgun/barcode sequences.

Flow:
1) Parse FASTA/raw sequence.
2) Route kingdom (Bacteria/Fungi/Viruses).
3) Choose the target genus model.
4) Predict on windowed chunks and aggregate by weighted voting.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from joblib import load


TARGETS = ("bacteria", "fungi", "rna")
TARGET_TO_KINGDOM = {
    "bacteria": "Bacteria",
    "fungi": "Fungi",
    "rna": "Viruses",
}
KINGDOM_TO_TARGET = {v: k for k, v in TARGET_TO_KINGDOM.items()}
DEFAULT_KINGDOM_ROUTER_MODEL = Path("runs/saved_models/kingdom_router_kmer_lr.joblib")
TARGET_ROOTS = {
    "bacteria": Path("DNA/Bacteria"),
    "fungi": Path("DNA/Fungi"),
    "rna": Path("RNA/Viruses"),
}
_TARGET_GENUS_VOCAB: dict[str, set[str]] | None = None
_ROUTER_MODELS_CACHE: dict[
    tuple[str, str], tuple[dict[str, "LoadedTargetModel"], dict[str, int]]
] = {}
_KINGDOM_ROUTER_CACHE: dict[str, object] = {
    "path": "",
    "mtime_ns": -1,
    "model": None,
}


@dataclass(frozen=True)
class LoadedTargetModel:
    target: str
    path: Path
    model: object


def _load_target_genus_vocab() -> dict[str, set[str]]:
    global _TARGET_GENUS_VOCAB
    if _TARGET_GENUS_VOCAB is not None:
        return _TARGET_GENUS_VOCAB

    vocab: dict[str, set[str]] = {target: set() for target in TARGETS}
    for target, root in TARGET_ROOTS.items():
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir():
                token = child.name.strip().lower()
                if token:
                    vocab[target].add(token)

    _TARGET_GENUS_VOCAB = vocab
    return vocab


def _header_hint_scores(router_records: list[tuple[str, str]]) -> dict[str, float]:
    vocab = _load_target_genus_vocab()
    hits = Counter()

    for header, _ in router_records:
        h = (header or "").lower()
        if not h:
            continue
        for target, genera in vocab.items():
            if any(genus in h for genus in genera):
                hits[target] += 1

    total = sum(hits.values())
    if total <= 0:
        return {target: 0.0 for target in TARGETS}

    return {target: float(hits.get(target, 0) / total) for target in TARGETS}


def clean_sequence(seq: str) -> str:
    return re.sub(r"[^A-Za-z]", "", seq).upper()


def parse_fasta_text(text: str) -> list[tuple[str, str]]:
    text = (text or "").strip()
    if not text:
        return []

    if not text.startswith(">"):
        seq = clean_sequence(text)
        return [("input_sequence", seq)] if seq else []

    items: list[tuple[str, str]] = []
    header = None
    seq_lines: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                seq = clean_sequence("".join(seq_lines))
                if seq:
                    items.append((header, seq))
            header = line[1:].strip() or "sequence"
            seq_lines = []
        else:
            seq_lines.append(line)

    if header is not None:
        seq = clean_sequence("".join(seq_lines))
        if seq:
            items.append((header, seq))

    return items


def parse_fasta_file(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return parse_fasta_text(text)


def _model_candidates(target: str, source: str) -> list[str]:
    if source == "latest":
        first = [
            f"latest_{target}_genus_kmer_lr_group_species_dedup1.joblib",
            f"latest_{target}_genus_kmer_lr.joblib",
        ]
        fallback = [
            f"best_{target}_genus_kmer_lr_group_species_dedup1.joblib",
            f"best_{target}_genus_kmer_lr.joblib",
        ]
        return first + fallback

    first = [
        f"best_{target}_genus_kmer_lr_group_species_dedup1.joblib",
        f"best_{target}_genus_kmer_lr.joblib",
    ]
    fallback = [
        f"latest_{target}_genus_kmer_lr_group_species_dedup1.joblib",
        f"latest_{target}_genus_kmer_lr.joblib",
    ]
    return first + fallback


def load_router_models(model_dir: Path = Path("runs/saved_models"), source: str = "best") -> dict[str, LoadedTargetModel]:
    if source not in {"best", "latest"}:
        raise ValueError("source must be 'best' or 'latest'")

    resolved_dir = str(model_dir.resolve())
    cache_key = (resolved_dir, source)
    cached = _ROUTER_MODELS_CACHE.get(cache_key)
    if cached is not None:
        cached_models, cached_mtimes = cached
        is_fresh = True
        for target, item in cached_models.items():
            p = item.path
            if not p.exists() or p.stat().st_mtime_ns != int(cached_mtimes.get(target, -1)):
                is_fresh = False
                break
        if is_fresh:
            return cached_models

    loaded: dict[str, LoadedTargetModel] = {}
    mtimes: dict[str, int] = {}
    for target in TARGETS:
        chosen = None
        for name in _model_candidates(target, source):
            p = model_dir / name
            if p.exists():
                chosen = p
                break

        if chosen is None:
            raise FileNotFoundError(
                f"No model found for target='{target}' in {model_dir}. "
                f"Expected one of: {_model_candidates(target, source)}"
            )

        loaded[target] = LoadedTargetModel(target=target, path=chosen, model=load(chosen))
        mtimes[target] = int(chosen.stat().st_mtime_ns)

    _ROUTER_MODELS_CACHE[cache_key] = (loaded, mtimes)
    return loaded


def load_kingdom_router_model(path: Path = DEFAULT_KINGDOM_ROUTER_MODEL):
    global _KINGDOM_ROUTER_CACHE

    if not path.exists():
        return None

    resolved = str(path.resolve())
    mtime = int(path.stat().st_mtime_ns)
    if (
        _KINGDOM_ROUTER_CACHE.get("path") == resolved
        and int(_KINGDOM_ROUTER_CACHE.get("mtime_ns", -1)) == mtime
    ):
        return _KINGDOM_ROUTER_CACHE.get("model")

    model = load(path)
    _KINGDOM_ROUTER_CACHE = {
        "path": resolved,
        "mtime_ns": mtime,
        "model": model,
    }
    return model


def _split_sequence_to_windows(
    seq: str,
    window_size: int,
    stride: int,
    min_len: int,
    max_windows: int,
) -> list[str]:
    seq = seq or ""
    n = len(seq)
    if n == 0:
        return []

    if window_size <= 0:
        window_size = n
    if stride <= 0:
        stride = window_size

    if n <= window_size:
        if n >= min_len:
            return [seq]
        return [seq] if n > 0 else []

    windows: list[str] = []
    start = 0
    while start < n and len(windows) < max_windows:
        end = start + window_size
        frag = seq[start:end]
        if len(frag) < min_len:
            break
        windows.append(frag)
        if end >= n:
            break
        start += stride

    if len(windows) < max_windows:
        tail = seq[-window_size:]
        if len(tail) >= min_len and (not windows or windows[-1] != tail):
            windows.append(tail)

    return windows


def _select_router_records(
    records: list[tuple[str, str]],
    min_len: int = 120,
    max_router_seqs: int = 120,
    router_window: int = 1200,
    router_stride: int = 600,
    max_windows_per_record: int = 25,
) -> list[tuple[str, str]]:
    filtered = [(h, s) for h, s in records if len(s) >= min_len]
    if not filtered:
        filtered = records[:]

    filtered.sort(key=lambda x: len(x[1]), reverse=True)

    selected: list[tuple[str, str]] = []
    for header, seq in filtered:
        windows = _split_sequence_to_windows(
            seq,
            window_size=router_window,
            stride=router_stride,
            min_len=min_len,
            max_windows=max_windows_per_record,
        )
        if not windows:
            continue

        total = len(windows)
        for idx, window in enumerate(windows, start=1):
            selected.append((f"{header} [win {idx}/{total}]", window))
            if len(selected) >= max_router_seqs:
                return selected

    if selected:
        return selected

    # Last fallback: keep any non-empty input sequence.
    return [(h, s) for h, s in records if s]


def _score_model(model, seqs: list[str], lens: list[int]) -> dict:
    if not seqs:
        return {
            "score": 0.0,
            "mean_conf": 0.0,
            "mean_margin": 0.0,
            "n_used": 0,
        }

    weights = np.sqrt(np.array(lens, dtype=float))

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(seqs)
        max_conf = proba.max(axis=1)

        if proba.shape[1] > 1:
            two = np.partition(proba, -2, axis=1)[:, -2:]
            margin = two[:, 1] - two[:, 0]
        else:
            margin = max_conf

        mean_conf = float(np.average(max_conf, weights=weights))
        mean_margin = float(np.average(margin, weights=weights))
        score = float(0.75 * mean_conf + 0.25 * mean_margin)

        return {
            "score": score,
            "mean_conf": mean_conf,
            "mean_margin": mean_margin,
            "n_used": len(seqs),
        }

    preds = model.predict(seqs)
    n_classes = len(set(preds))
    base = 1.0 / max(n_classes, 1)
    return {
        "score": float(base),
        "mean_conf": float(base),
        "mean_margin": 0.0,
        "n_used": len(seqs),
    }


def route_kingdom(
    records: list[tuple[str, str]],
    models: dict[str, LoadedTargetModel],
    min_len: int = 120,
    max_router_seqs: int = 120,
    router_window: int = 1200,
    router_stride: int = 600,
) -> dict:
    router_records = _select_router_records(
        records,
        min_len=min_len,
        max_router_seqs=max_router_seqs,
        router_window=router_window,
        router_stride=router_stride,
    )
    seqs = [s for _, s in router_records]
    lens = [len(s) for s in seqs]

    rows = []
    for target in TARGETS:
        stats = _score_model(models[target].model, seqs, lens)
        rows.append(
            {
                "target": target,
                "kingdom": TARGET_TO_KINGDOM[target],
                "model_path": str(models[target].path),
                **stats,
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    best = rows[0]

    return {
        "selected_target": best["target"],
        "selected_kingdom": best["kingdom"],
        "selected_model_path": best["model_path"],
        "routing_scores": rows,
        "router_sequences_used": len(router_records),
        "routing_method": "multi_model_confidence",
    }


def route_kingdom_by_router_model(
    records: list[tuple[str, str]],
    models: dict[str, LoadedTargetModel],
    kingdom_router_model,
    min_len: int = 120,
    max_router_seqs: int = 120,
    router_window: int = 1200,
    router_stride: int = 600,
    router_weight: float = 0.65,
    evidence_weight: float = 0.35,
    header_hint_weight: float = 0.20,
) -> dict:
    router_records = _select_router_records(
        records,
        min_len=min_len,
        max_router_seqs=max_router_seqs,
        router_window=router_window,
        router_stride=router_stride,
    )

    seqs = [s for _, s in router_records]
    lens = np.array([len(s) for s in seqs], dtype=float)
    weights = np.sqrt(lens) if len(lens) else np.array([], dtype=float)

    if hasattr(kingdom_router_model, "predict_proba"):
        proba = kingdom_router_model.predict_proba(seqs)
        classes = list(kingdom_router_model.classes_)
        score_by_class: dict[str, float] = {}
        for idx, cls in enumerate(classes):
            score_by_class[cls] = float(np.average(proba[:, idx], weights=weights))
    else:
        preds = kingdom_router_model.predict(seqs)
        score_by_class: dict[str, float] = {}
        for cls in set(preds):
            score_by_class[cls] = preds.count(cls) / max(len(preds), 1)

    evidence_by_target = {}
    for target in TARGETS:
        stats = _score_model(models[target].model, seqs, list(lens.astype(int)))
        evidence_by_target[target] = float(stats["score"])
    header_hint_by_target = _header_hint_scores(router_records)

    router_weight = max(0.0, float(router_weight))
    evidence_weight = max(0.0, float(evidence_weight))
    header_hint_weight = max(0.0, float(header_hint_weight))
    if router_weight + evidence_weight + header_hint_weight <= 0:
        router_weight, evidence_weight, header_hint_weight = 1.0, 0.0, 0.0

    norm = router_weight + evidence_weight + header_hint_weight
    router_weight /= norm
    evidence_weight /= norm
    header_hint_weight /= norm

    rows = []
    for kingdom, score in score_by_class.items():
        target = KINGDOM_TO_TARGET.get(kingdom)
        if not target or target not in models:
            continue
        model_evidence = evidence_by_target.get(target, 0.0)
        header_hint = header_hint_by_target.get(target, 0.0)
        combined = (
            router_weight * float(score)
            + evidence_weight * model_evidence
            + header_hint_weight * header_hint
        )
        rows.append(
            {
                "target": target,
                "kingdom": kingdom,
                "model_path": str(models[target].path),
                "score": float(combined),
                "router_score": float(score),
                "evidence_score": float(model_evidence),
                "header_hint_score": float(header_hint),
                "mean_conf": float(combined),
                "mean_margin": 0.0,
                "n_used": len(seqs),
            }
        )

    if not rows:
        return route_kingdom(
            records,
            models,
            min_len=min_len,
            max_router_seqs=max_router_seqs,
            router_window=router_window,
            router_stride=router_stride,
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    best = rows[0]
    routing_method = "kingdom_router_model"

    # Tie-break using header signal when combined scores are very close.
    header_target = max(header_hint_by_target, key=header_hint_by_target.get)
    header_strength = float(header_hint_by_target.get(header_target, 0.0))
    if header_strength >= 0.5:
        header_row = next((r for r in rows if r["target"] == header_target), None)
        if header_row is not None and best["target"] != header_target:
            if float(best["score"]) - float(header_row["score"]) < 0.12:
                best = header_row
                routing_method = "kingdom_router_model_header_tiebreak"

    return {
        "selected_target": best["target"],
        "selected_kingdom": best["kingdom"],
        "selected_model_path": best["model_path"],
        "routing_scores": rows,
        "router_sequences_used": len(router_records),
        "routing_method": routing_method,
    }


def predict_with_model(
    model,
    records: list[tuple[str, str]],
    top_k: int = 5,
    window_size: int = 1500,
    stride: int = 750,
    min_len: int = 120,
    max_windows_per_record: int = 40,
    reject_threshold: float = 0.0,
) -> list[dict]:
    has_proba = hasattr(model, "predict_proba")
    classes = list(model.classes_) if hasattr(model, "classes_") else []

    out = []
    for header, seq in records:
        windows = _split_sequence_to_windows(
            seq,
            window_size=window_size,
            stride=stride,
            min_len=min_len,
            max_windows=max_windows_per_record,
        )
        if not windows:
            continue

        row = {
            "header": header,
            "length": len(seq),
            "n_windows": len(windows),
            "prediction": None,
            "confidence": None,
            "top": [],
        }

        if has_proba and classes:
            proba = model.predict_proba(windows)
            weights = np.sqrt(np.array([len(w) for w in windows], dtype=float))
            avg = np.average(proba, axis=0, weights=weights)

            top_idx = np.argsort(avg)[::-1][:top_k]
            top_label = classes[top_idx[0]]
            top_conf = float(avg[top_idx[0]])
            row["prediction"] = top_label
            row["confidence"] = top_conf
            row["raw_prediction"] = top_label
            row["top"] = [{"label": classes[j], "score": float(avg[j])} for j in top_idx]
        else:
            preds = model.predict(windows)
            major, cnt = Counter(preds).most_common(1)[0]
            row["prediction"] = major
            row["confidence"] = float(cnt / len(preds))
            row["raw_prediction"] = major

        if row["confidence"] is not None and row["confidence"] < reject_threshold:
            row["prediction"] = "UNCERTAIN"

        out.append(row)

    return out


def _estimate_target_fit(
    model,
    records: list[tuple[str, str]],
    *,
    max_records: int,
    window_size: int,
    stride: int,
    min_len: int,
    max_windows_per_record: int,
) -> dict:
    sample = sorted(records, key=lambda x: len(x[1]), reverse=True)[:max_records]
    if not sample:
        return {"score": 0.0, "n_records": 0}

    preds = predict_with_model(
        model,
        sample,
        top_k=3,
        window_size=window_size,
        stride=stride,
        min_len=min_len,
        max_windows_per_record=max_windows_per_record,
        reject_threshold=0.0,
    )
    if not preds:
        return {"score": 0.0, "n_records": 0}

    confs = np.array([float(p.get("confidence") or 0.0) for p in preds], dtype=float)
    weights = np.sqrt(np.array([max(1, int(p.get("length", 1))) for p in preds], dtype=float))
    score = float(np.average(confs, weights=weights)) if len(confs) else 0.0
    return {"score": score, "n_records": len(preds)}


def run_central_reader(
    records: list[tuple[str, str]],
    model_source: str = "best",
    min_len: int = 120,
    max_router_seqs: int = 120,
    max_predict_seqs: int = 120,
    router_window: int = 1200,
    router_stride: int = 600,
    predict_window: int = 1500,
    predict_stride: int = 750,
    max_predict_windows_per_record: int = 40,
    reject_threshold: float = 0.0,
    force_target: str | None = None,
    router_weight: float = 0.65,
    evidence_weight: float = 0.35,
    header_hint_weight: float = 0.20,
    virus_fit_fallback: bool = True,
    fit_fallback_gap: float = 0.06,
    fit_fallback_min_virus_score: float = 0.50,
    fit_fallback_max_records: int = 20,
) -> dict:
    if not records:
        raise ValueError("No valid sequence records")

    models = load_router_models(source=model_source)

    kingdom_router = load_kingdom_router_model()

    if force_target is not None:
        force_target = force_target.strip().lower()
        if force_target not in TARGETS:
            raise ValueError(f"force_target must be one of {TARGETS}, got '{force_target}'")
        routed = {
            "selected_target": force_target,
            "selected_kingdom": TARGET_TO_KINGDOM[force_target],
            "selected_model_path": str(models[force_target].path),
            "routing_scores": [],
            "router_sequences_used": 0,
            "routing_method": "manual_override",
        }
    elif kingdom_router is not None:
        routed = route_kingdom_by_router_model(
            records,
            models,
            kingdom_router_model=kingdom_router,
            min_len=min_len,
            max_router_seqs=max_router_seqs,
            router_window=router_window,
            router_stride=router_stride,
            router_weight=router_weight,
            evidence_weight=evidence_weight,
            header_hint_weight=header_hint_weight,
        )
    else:
        routed = route_kingdom(
            records,
            models,
            min_len=min_len,
            max_router_seqs=max_router_seqs,
            router_window=router_window,
            router_stride=router_stride,
        )

    predict_records = sorted(records, key=lambda x: len(x[1]), reverse=True)[:max_predict_seqs]
    fit_scores: dict[str, dict] = {}

    selected_target = routed["selected_target"]
    if force_target is None and virus_fit_fallback and selected_target == "rna":
        for target in TARGETS:
            fit_scores[target] = _estimate_target_fit(
                models[target].model,
                predict_records,
                max_records=fit_fallback_max_records,
                window_size=predict_window,
                stride=predict_stride,
                min_len=min_len,
                max_windows_per_record=max_predict_windows_per_record,
            )

        virus_score = float(fit_scores.get("rna", {}).get("score", 0.0))
        best_target = max(TARGETS, key=lambda t: float(fit_scores.get(t, {}).get("score", 0.0)))
        best_score = float(fit_scores.get(best_target, {}).get("score", 0.0))

        if best_target != "rna":
            strong_alt = best_score >= max(0.55, virus_score + fit_fallback_gap)
            if virus_score < fit_fallback_min_virus_score and strong_alt:
                selected_target = best_target
                routed["selected_target"] = best_target
                routed["selected_kingdom"] = TARGET_TO_KINGDOM[best_target]
                routed["selected_model_path"] = str(models[best_target].path)
                routed["routing_method"] = f"{routed.get('routing_method', 'auto')}+fit_fallback"

    selected_model = models[selected_target].model
    predictions = predict_with_model(
        selected_model,
        predict_records,
        top_k=5,
        window_size=predict_window,
        stride=predict_stride,
        min_len=min_len,
        max_windows_per_record=max_predict_windows_per_record,
        reject_threshold=reject_threshold,
    )

    return {
        **routed,
        "model_source": model_source,
        "input_sequences": len(records),
        "predicted_sequences": len(predictions),
        "reject_threshold": reject_threshold,
        "fit_scores": fit_scores,
        "predictions": predictions,
    }
