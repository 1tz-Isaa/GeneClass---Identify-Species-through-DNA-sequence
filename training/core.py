import csv
import re
from dataclasses import replace
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from dataset_loader import load_dataset
import numpy as np
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

RNA_GENUS_TO_FAMILY = {
    "alphacoronavirus": "Coronaviridae",
    "betacoronavirus": "Coronaviridae",
    "enterovirus": "Picornaviridae",
    "rhinovirus": "Picornaviridae",
    "influenzavirus": "Orthomyxoviridae",
    "influenza": "Orthomyxoviridae",
    "hantavirus": "Hantaviridae",
    "orthobunyavirus": "Peribunyaviridae",
    "reovirus": "Reoviridae",
    "arenavirus": "Arenaviridae",
    "metapneumovirus": "Pneumoviridae",
    "orthopneumovirus": "Pneumoviridae",
    "respirovirus": "Paramyxoviridae",
}

RNA_GENUS_CANONICAL = {
    "alphacoronavirus": "Alphacoronavirus",
    "betacoronavirus": "Betacoronavirus",
    "enterovirus": "Enterovirus",
    "rhinovirus": "Rhinovirus",
    "influenzavirus": "Influenzavirus",
    "influenza": "Influenzavirus",
    "hantavirus": "Hantavirus",
    "orthobunyavirus": "Orthobunyavirus",
    "reovirus": "Reovirus",
    "arenavirus": "Arenavirus",
    "metapneumovirus": "Metapneumovirus",
    "orthopneumovirus": "Orthopneumovirus",
    "respirovirus": "Respirovirus",
}

RNA_FAMILY_FRAGMENT_LEN = {
    "Orthomyxoviridae": 1000,
    "Coronaviridae": 1400,
    "Picornaviridae": 1000,
    "Hantaviridae": 1200,
    "Peribunyaviridae": 1200,
    "Reoviridae": 1000,
    "Arenaviridae": 1200,
    "Pneumoviridae": 1200,
    "Paramyxoviridae": 1400,
}


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def clean_sequence(seq: str) -> str:
    return re.sub(r"[^A-Za-z]", "", seq).upper()


def normalize_train_fragment(seq: str, fragment_len: int) -> str:
    if fragment_len <= 0:
        return seq
    n = len(seq)
    if n == fragment_len:
        return seq
    if n > fragment_len:
        start = (n - fragment_len) // 2
        return seq[start : start + fragment_len]
    return seq + ("N" * (fragment_len - n))


def is_bad_header(header: str) -> bool:
    h = (header or "").lower()
    return any(token in h for token in BAD_HEADER_PATTERNS)


def _normalize_taxon_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").strip().lower())


def rna_genus_to_family(label: str) -> str:
    raw = str(label or "").strip()
    token = _normalize_taxon_token(raw)
    if not token:
        return "Unknown"
    return RNA_GENUS_TO_FAMILY.get(token, raw)


def canonicalize_rna_label(label: str, genus: str, label_level: str, collapse_nested_species: bool) -> str:
    raw_label = str(label or "").strip()
    if not raw_label:
        return raw_label

    genus_token = _normalize_taxon_token(genus)

    if label_level == "genus":
        token = _normalize_taxon_token(raw_label)
        return RNA_GENUS_CANONICAL.get(token, raw_label)

    if not collapse_nested_species:
        return raw_label

    if genus_token in {"influenzavirus", "influenza"}:
        collapsed = re.sub(r"_H\d+N\d+$", "", raw_label, flags=re.IGNORECASE)
        collapsed = re.sub(r"\s+H\d+N\d+$", "", collapsed, flags=re.IGNORECASE)
        return collapsed

    return raw_label


def resolve_rna_fragment_len(genus: str, cfg: TrainConfig) -> int:
    base_len = int(cfg.train_fragment_len)
    if not cfg.rna_use_family_fragment_len:
        return base_len
    family = rna_genus_to_family(genus)
    return int(RNA_FAMILY_FRAGMENT_LEN.get(family, base_len))


def _resolve_n_jobs(cpu_jobs: int) -> int:
    if cpu_jobs < 0:
        return cpu_jobs
    if cpu_jobs == 0:
        cores = os.cpu_count() or 1
        # Keep one core free so local UI/apps stay responsive.
        return max(1, cores - 1)
    return max(1, cpu_jobs)


def build_model(cfg: TrainConfig):
    return make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(cfg.kmer_min, cfg.kmer_max),
            lowercase=False,
            min_df=cfg.kmer_min_df,
            max_features=cfg.kmer_max_features,
            sublinear_tf=True,
            dtype=np.float32,
        ),
        LogisticRegression(
            max_iter=cfg.lr_max_iter,
            solver=cfg.lr_solver,
            C=cfg.lr_c,
            tol=cfg.lr_tol,
            class_weight=cfg.lr_class_weight,
            random_state=cfg.random_state,
        ),
    )


class HierarchicalRnaFamilyModel:
    """Two-stage RNA classifier: family first, then family-specific genus model."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.global_model = None
        self.family_model = None
        self.family_models: dict[str, object] = {}
        self.family_single_label: dict[str, str] = {}
        self.family_default_label: dict[str, str] = {}
        self.family_to_labels: dict[str, list[str]] = {}
        self.classes_ = np.array([], dtype=object)
        # Keep global genus evidence to recover from wrong family routing.
        self.hierarchical_weight = float(cfg.rna_hierarchical_weight)
        self.family_top_k = max(0, int(cfg.rna_family_top_k))

    def fit(self, sequences, labels):
        labels = [str(x) for x in labels]
        families = [rna_genus_to_family(x) for x in labels]

        self.classes_ = np.array(sorted(set(labels)), dtype=object)
        self.global_model = build_model(self.cfg)
        self.global_model.fit(sequences, labels)

        family_cfg = replace(
            self.cfg,
            kmer_max_features=min(self.cfg.kmer_max_features, 18000),
            lr_max_iter=min(self.cfg.lr_max_iter, 2500),
            lr_tol=max(self.cfg.lr_tol, 0.0015),
        )
        self.family_model = build_model(family_cfg)
        self.family_model.fit(sequences, families)

        fam_to_idx: dict[str, list[int]] = defaultdict(list)
        for idx, fam in enumerate(families):
            fam_to_idx[fam].append(idx)

        sub_cfg = replace(
            self.cfg,
            kmer_max_features=min(self.cfg.kmer_max_features, 25000),
            lr_max_iter=min(self.cfg.lr_max_iter, 3000),
            lr_tol=max(self.cfg.lr_tol, 0.001),
        )

        for fam, idxs in fam_to_idx.items():
            fam_labels = [labels[i] for i in idxs]
            label_counts = Counter(fam_labels)
            self.family_default_label[fam] = label_counts.most_common(1)[0][0]
            uniq_labels = sorted(set(fam_labels))
            self.family_to_labels[fam] = uniq_labels

            if len(uniq_labels) == 1:
                self.family_single_label[fam] = uniq_labels[0]
                continue

            sub_x = [sequences[i] for i in idxs]
            sub_model = build_model(sub_cfg)
            sub_model.fit(sub_x, fam_labels)
            self.family_models[fam] = sub_model

        return self

    def predict(self, sequences):
        if self.family_model is None:
            raise RuntimeError("Model is not fitted")
        probs = self.predict_proba(sequences)
        if len(probs) == 0 or len(self.classes_) == 0:
            return np.array([], dtype=object)
        return np.array([self.classes_[idx] for idx in np.argmax(probs, axis=1)], dtype=object)

    def predict_proba(self, sequences):
        if self.family_model is None:
            raise RuntimeError("Model is not fitted")

        n = len(sequences)
        out = np.zeros((n, len(self.classes_)), dtype=float)
        if n == 0 or len(self.classes_) == 0:
            return out

        class_to_idx = {str(label): idx for idx, label in enumerate(self.classes_)}

        if hasattr(self.family_model, "predict_proba"):
            fam_classes = [str(x) for x in self.family_model.classes_]
            fam_proba = self.family_model.predict_proba(sequences)
        else:
            fam_pred = [str(x) for x in self.family_model.predict(sequences)]
            fam_classes = sorted(set(fam_pred))
            fam_proba = np.zeros((n, len(fam_classes)), dtype=float)
            fam_to_idx = {fam: idx for idx, fam in enumerate(fam_classes)}
            for i, fam in enumerate(fam_pred):
                fam_proba[i, fam_to_idx[fam]] = 1.0

        for i, seq in enumerate(sequences):
            if self.family_top_k > 0 and self.family_top_k < len(fam_classes):
                top_k = int(self.family_top_k)
                fam_row = fam_proba[i]
                top_idx = np.argpartition(fam_row, -top_k)[-top_k:]
                top_mass = float(np.sum(fam_row[top_idx]))
                if top_mass > 0:
                    family_iter = [
                        (int(j), float(fam_row[j] / top_mass))
                        for j in top_idx.tolist()
                        if float(fam_row[j]) > 0
                    ]
                else:
                    family_iter = []
            else:
                family_iter = [
                    (fam_idx, float(fam_proba[i, fam_idx]))
                    for fam_idx in range(len(fam_classes))
                    if float(fam_proba[i, fam_idx]) > 0
                ]

            for fam_idx, p_fam in family_iter:
                fam = fam_classes[fam_idx]

                labels_in_family = self.family_to_labels.get(fam, [])
                if not labels_in_family:
                    continue

                if fam in self.family_single_label:
                    lbl = self.family_single_label[fam]
                    idx = class_to_idx.get(str(lbl))
                    if idx is not None:
                        out[i, idx] += p_fam
                    continue

                sub_model = self.family_models.get(fam)
                if sub_model is not None and hasattr(sub_model, "predict_proba"):
                    sub_proba = sub_model.predict_proba([seq])[0]
                    for lbl, p_sub in zip(sub_model.classes_, sub_proba):
                        idx = class_to_idx.get(str(lbl))
                        if idx is not None:
                            out[i, idx] += p_fam * float(p_sub)
                    continue

                if sub_model is not None:
                    lbl = str(sub_model.predict([seq])[0])
                    idx = class_to_idx.get(lbl)
                    if idx is not None:
                        out[i, idx] += p_fam
                    continue

                share = p_fam / len(labels_in_family)
                for lbl in labels_in_family:
                    idx = class_to_idx.get(str(lbl))
                    if idx is not None:
                        out[i, idx] += share

            row_sum = float(out[i].sum())
            if row_sum > 0:
                out[i] /= row_sum
            else:
                out[i] = 1.0 / len(self.classes_)

        if self.global_model is not None:
            global_out = np.zeros_like(out)
            if hasattr(self.global_model, "predict_proba"):
                global_proba = self.global_model.predict_proba(sequences)
                for j, lbl in enumerate(self.global_model.classes_):
                    idx = class_to_idx.get(str(lbl))
                    if idx is not None:
                        global_out[:, idx] = global_proba[:, j]
            else:
                for i, lbl in enumerate(self.global_model.predict(sequences)):
                    idx = class_to_idx.get(str(lbl))
                    if idx is not None:
                        global_out[i, idx] = 1.0

            alpha = float(min(max(self.hierarchical_weight, 0.0), 1.0))
            out = alpha * out + (1.0 - alpha) * global_out
            row_sums = out.sum(axis=1, keepdims=True)
            nonzero = row_sums[:, 0] > 0
            out[nonzero] /= row_sums[nonzero]
            if np.any(~nonzero):
                out[~nonzero] = 1.0 / len(self.classes_)

        return out


def prepare_dataset(cfg: TrainConfig):
    if cfg.train_target not in TARGET_CONFIG:
        raise ValueError(
            f"Invalid TRAIN_TARGET input='{cfg.train_target_input}' resolved='{cfg.train_target}'.\n"
            f"{format_target_table()}"
        )

    root_folder = TARGET_CONFIG[cfg.train_target]["root"]
    kingdom_name = TARGET_CONFIG[cfg.train_target]["kingdom"]

    data = load_dataset(
        root_folder,
        show_progress=cfg.show_file_progress,
        kingdom_filter=kingdom_name,
    )

    rows = []
    dropped_labels_min_unique: dict[str, int] = {}
    dropped_samples_min_unique = 0
    dropped_labels_min_samples: dict[str, int] = {}
    dropped_samples_min_samples = 0
    for item in data:
        if item["kingdom"] != kingdom_name:
            continue

        raw_seq = clean_sequence(item["sequence"])
        if not raw_seq or len(raw_seq) < cfg.min_seq_len:
            continue
        genus = item["genus"]
        species = item["species"]

        fragment_len = int(cfg.train_fragment_len)
        if cfg.train_target == "rna":
            fragment_len = resolve_rna_fragment_len(genus, cfg)

        seq = normalize_train_fragment(raw_seq, fragment_len)
        if cfg.max_seq_len > 0 and len(seq) > cfg.max_seq_len:
            seq = seq[: cfg.max_seq_len]

        header = item.get("header", "")
        if cfg.filter_bad_headers and is_bad_header(header):
            continue

        label = item[cfg.label_level]
        if cfg.train_target == "rna":
            label = canonicalize_rna_label(
                label=label,
                genus=genus,
                label_level=cfg.label_level,
                collapse_nested_species=cfg.rna_collapse_nested_species,
            )
        group = f"{genus}/{species}"
        rows.append((seq, label, genus, group, raw_seq))

    if cfg.dedup_exact:
        deduped = []
        seen = set()
        for seq, label, genus, group, raw_seq in rows:
            key = (raw_seq, label)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((seq, label, genus, group, raw_seq))
        rows = deduped

    if cfg.train_target == "rna" and cfg.rna_min_unique_genomes_per_label > 0:
        label_to_genomes: dict[str, set[str]] = defaultdict(set)
        for _, label, _, _, raw_seq in rows:
            label_to_genomes[str(label)].add(raw_seq)

        min_unique = int(cfg.rna_min_unique_genomes_per_label)
        dropped_labels_min_unique = {
            label: len(genomes)
            for label, genomes in label_to_genomes.items()
            if len(genomes) < min_unique
        }
        if dropped_labels_min_unique:
            before = len(rows)
            blocked = set(dropped_labels_min_unique)
            rows = [row for row in rows if row[1] not in blocked]
            dropped_samples_min_unique = before - len(rows)

    if cfg.train_target == "rna" and cfg.rna_min_samples_per_label > 0:
        label_counts = Counter(str(row[1]) for row in rows)
        min_samples = int(cfg.rna_min_samples_per_label)
        dropped_labels_min_samples = {
            label: int(count)
            for label, count in label_counts.items()
            if int(count) < min_samples
        }
        if dropped_labels_min_samples:
            before = len(rows)
            blocked = set(dropped_labels_min_samples)
            rows = [row for row in rows if str(row[1]) not in blocked]
            dropped_samples_min_samples = before - len(rows)

    if cfg.max_samples_per_label > 0:
        by_label: dict[str, list[tuple[str, str, str, str, str]]] = defaultdict(list)
        for row in rows:
            by_label[row[1]].append(row)

        sampled_rows = []
        rng = np.random.default_rng(cfg.random_state)
        limit = int(cfg.max_samples_per_label)
        for label in sorted(by_label):
            items = by_label[label]
            if len(items) <= limit:
                sampled_rows.extend(items)
                continue
            pick_idx = np.sort(rng.choice(len(items), size=limit, replace=False))
            sampled_rows.extend(items[i] for i in pick_idx.tolist())
        rows = sampled_rows

    if cfg.max_samples_total > 0 and len(rows) > cfg.max_samples_total:
        idx = list(range(len(rows)))
        labels_all = [rows[i][1] for i in idx]
        use_stratify = min(Counter(labels_all).values()) >= 2
        keep_idx, _ = train_test_split(
            idx,
            train_size=cfg.max_samples_total,
            random_state=cfg.random_state,
            stratify=labels_all if use_stratify else None,
        )
        keep_set = set(keep_idx)
        rows = [rows[i] for i in idx if i in keep_set]

    if not rows:
        raise ValueError(f"No data found for target={cfg.train_target} in {root_folder}/{kingdom_name}")

    labels = [row[1] for row in rows]
    if len(set(labels)) < 2:
        raise ValueError(f"Need at least 2 classes to train, found: {sorted(set(labels))}")

    label_unique_genomes: dict[str, int] = {}
    uniq_tracker: dict[str, set[str]] = defaultdict(set)
    for _, label, _, _, raw_seq in rows:
        uniq_tracker[str(label)].add(raw_seq)
    for label, genomes in uniq_tracker.items():
        label_unique_genomes[label] = len(genomes)

    return {
        "root_folder": root_folder,
        "kingdom": kingdom_name,
        "sequences": [row[0] for row in rows],
        "labels": labels,
        "genera": [row[2] for row in rows],
        "groups": [row[3] for row in rows],
        "label_unique_genomes": label_unique_genomes,
        "dropped_labels_min_unique_genomes": dropped_labels_min_unique,
        "dropped_samples_min_unique_genomes": dropped_samples_min_unique,
        "dropped_labels_min_samples_per_label": dropped_labels_min_samples,
        "dropped_samples_min_samples_per_label": dropped_samples_min_samples,
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


def _build_lr_c_candidates(lr_c: float) -> list[float]:
    base = max(float(lr_c), 1e-6)
    out = sorted(
        {
            round(max(0.05, base * 0.5), 6),
            round(max(0.05, base), 6),
            round(max(0.05, base * 2.0), 6),
        }
    )
    return out


def _dedupe_preserve(values):
    out = []
    seen = set()
    for v in values:
        key = str(v)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _split_fit_eval_indices(labels, groups, cfg: TrainConfig, holdout_size: float):
    train_sub_idx, val_sub_idx = [], []
    try:
        if cfg.split_mode == "group_species" and len(set(groups)) >= 2:
            split = GroupShuffleSplit(
                n_splits=1,
                test_size=holdout_size,
                random_state=cfg.random_state,
            )
            train_sub_idx, val_sub_idx = next(split.split(range(len(labels)), labels, groups=groups))
            train_sub_idx = list(train_sub_idx)
            val_sub_idx = list(val_sub_idx)
        else:
            use_stratify = min(Counter(labels).values()) >= 2
            train_sub_idx, val_sub_idx = train_test_split(
                list(range(len(labels))),
                test_size=holdout_size,
                random_state=cfg.random_state,
                stratify=labels if use_stratify else None,
            )
            train_sub_idx = list(train_sub_idx)
            val_sub_idx = list(val_sub_idx)
    except Exception:
        train_sub_idx, val_sub_idx = [], []
    return train_sub_idx, val_sub_idx


def fit_model(sequences, labels, groups, cfg: TrainConfig):
    if cfg.train_target == "rna" and cfg.label_level == "genus":
        selected_weight = float(cfg.rna_hierarchical_weight)
        selected_top_k = int(cfg.rna_family_top_k)
        holdout_size = min(0.35, max(0.10, float(cfg.auto_tune_holdout_size)))
        hierarchy_candidates = []
        hierarchy_tune_used = False
        hierarchy_auto_tune_enabled = os.getenv("RNA_AUTO_TUNE_BLEND", "0") == "1"

        can_tune_hierarchy = (
            hierarchy_auto_tune_enabled
            and len(set(labels)) >= 2
            and len(labels) >= 80
        )
        if can_tune_hierarchy:
            sample_idx = list(range(len(labels)))
            if len(sample_idx) > cfg.auto_tune_max_samples > 0:
                sample_idx, _ = train_test_split(
                    sample_idx,
                    train_size=cfg.auto_tune_max_samples,
                    random_state=cfg.random_state,
                    stratify=labels if min(Counter(labels).values()) >= 2 else None,
                )
                sample_idx = list(sample_idx)

            sx = [sequences[i] for i in sample_idx]
            sy = [labels[i] for i in sample_idx]
            sg = [groups[i] for i in sample_idx]
            train_sub_idx, val_sub_idx = _split_fit_eval_indices(sy, sg, cfg, holdout_size)

            if train_sub_idx and val_sub_idx:
                x_fit = [sx[i] for i in train_sub_idx]
                y_fit = [sy[i] for i in train_sub_idx]
                x_eval = [sx[i] for i in val_sub_idx]
                y_eval = [sy[i] for i in val_sub_idx]

                tune_model = HierarchicalRnaFamilyModel(cfg).fit(x_fit, y_fit)
                top_k_candidates = _dedupe_preserve([selected_top_k, 0, 1, 2, 3])
                weight_candidates = _dedupe_preserve(
                    [0.0, 0.15, 0.25, selected_weight, 0.35, 0.50, 0.70, 1.0]
                )

                best_score = float("-inf")
                for top_k in top_k_candidates:
                    tune_model.family_top_k = max(0, int(top_k))
                    for weight in weight_candidates:
                        tune_model.hierarchical_weight = float(min(max(weight, 0.0), 1.0))
                        try:
                            pred = tune_model.predict(x_eval)
                            metrics = calc_metrics(y_eval, pred)
                            score = float(0.7 * metrics["balanced_accuracy"] + 0.3 * metrics["f1_weighted"])
                            row = {
                                "family_top_k": int(tune_model.family_top_k),
                                "hierarchical_weight": float(tune_model.hierarchical_weight),
                                "accuracy": float(metrics["accuracy"]),
                                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                                "f1_weighted": float(metrics["f1_weighted"]),
                                "score": score,
                            }
                            hierarchy_candidates.append(row)
                            if score > best_score:
                                best_score = score
                                selected_top_k = int(tune_model.family_top_k)
                                selected_weight = float(tune_model.hierarchical_weight)
                        except Exception:
                            continue

                hierarchy_tune_used = bool(hierarchy_candidates)

        tuned_cfg = replace(
            cfg,
            rna_family_top_k=int(selected_top_k),
            rna_hierarchical_weight=float(selected_weight),
        )
        model = HierarchicalRnaFamilyModel(tuned_cfg).fit(sequences, labels)
        return model, {
            "hierarchical_family_mode": True,
            "family_top_k": int(selected_top_k),
            "hierarchical_weight": float(selected_weight),
            "hierarchy_auto_tune_enabled": bool(hierarchy_auto_tune_enabled),
            "hierarchy_auto_tune_used": bool(hierarchy_tune_used),
            "hierarchy_candidates": hierarchy_candidates,
            "auto_tune_lr_c_enabled": bool(cfg.auto_tune_lr_c),
            "auto_tune_lr_c_used": False,
            "selected_lr_c": float(cfg.lr_c),
            "sample_size": int(len(labels)),
            "holdout_size": float(cfg.auto_tune_holdout_size),
            "family_count": int(len(model.family_to_labels)),
        }

    selected_lr_c = float(cfg.lr_c)
    tuning_rows = []

    can_tune = (
        cfg.auto_tune_lr_c
        and len(set(labels)) >= 2
        and len(labels) >= 200
    )

    sample_idx = list(range(len(labels)))
    if can_tune and len(sample_idx) > cfg.auto_tune_max_samples > 0:
        sample_idx, _ = train_test_split(
            sample_idx,
            train_size=cfg.auto_tune_max_samples,
            random_state=cfg.random_state,
            stratify=labels if min(Counter(labels).values()) >= 2 else None,
        )
        sample_idx = list(sample_idx)

    holdout_size = min(0.35, max(0.10, float(cfg.auto_tune_holdout_size)))
    tuning_used = False

    if can_tune and len(sample_idx) >= 80:
        sx = [sequences[i] for i in sample_idx]
        sy = [labels[i] for i in sample_idx]
        sg = [groups[i] for i in sample_idx]

        try:
            if cfg.split_mode == "group_species" and len(set(sg)) >= 2:
                split = GroupShuffleSplit(
                    n_splits=1,
                    test_size=holdout_size,
                    random_state=cfg.random_state,
                )
                train_sub_idx, val_sub_idx = next(split.split(range(len(sy)), sy, groups=sg))
                train_sub_idx = list(train_sub_idx)
                val_sub_idx = list(val_sub_idx)
            else:
                use_stratify = min(Counter(sy).values()) >= 2
                train_sub_idx, val_sub_idx = train_test_split(
                    list(range(len(sy))),
                    test_size=holdout_size,
                    random_state=cfg.random_state,
                    stratify=sy if use_stratify else None,
                )
                train_sub_idx = list(train_sub_idx)
                val_sub_idx = list(val_sub_idx)
        except Exception:
            train_sub_idx, val_sub_idx = [], []

        if train_sub_idx and val_sub_idx:
            x_fit = [sx[i] for i in train_sub_idx]
            y_fit = [sy[i] for i in train_sub_idx]
            x_eval = [sx[i] for i in val_sub_idx]
            y_eval = [sy[i] for i in val_sub_idx]

            best_score = float("-inf")
            for candidate_c in _build_lr_c_candidates(cfg.lr_c):
                # Keep tuning cheap: fewer features + lower max_iter for candidate scan.
                candidate_cfg = replace(
                    cfg,
                    lr_c=float(candidate_c),
                    kmer_max_features=min(cfg.kmer_max_features, 30000),
                    lr_max_iter=min(cfg.lr_max_iter, 1200),
                    lr_tol=max(cfg.lr_tol, 0.001),
                )
                candidate_model = build_model(candidate_cfg)
                try:
                    candidate_model.fit(x_fit, y_fit)
                    pred = candidate_model.predict(x_eval)
                    metrics = calc_metrics(y_eval, pred)
                    score = float(0.7 * metrics["balanced_accuracy"] + 0.3 * metrics["f1_weighted"])
                    tuning_rows.append(
                        {
                            "lr_c": float(candidate_c),
                            "balanced_accuracy": float(metrics["balanced_accuracy"]),
                            "f1_weighted": float(metrics["f1_weighted"]),
                            "score": score,
                        }
                    )
                    if score > best_score:
                        best_score = score
                        selected_lr_c = float(candidate_c)
                except Exception:
                    continue

            tuning_used = bool(tuning_rows)

    final_cfg = replace(cfg, lr_c=selected_lr_c)
    model = build_model(final_cfg)
    model.fit(sequences, labels)

    return model, {
        "auto_tune_lr_c_enabled": bool(cfg.auto_tune_lr_c),
        "auto_tune_lr_c_used": bool(tuning_used),
        "selected_lr_c": float(selected_lr_c),
        "sample_size": int(len(sample_idx)),
        "holdout_size": float(holdout_size),
        "candidates": tuning_rows,
    }


def run_cv(sequences, labels, groups, cfg: TrainConfig):
    if not cfg.enable_cv:
        return {"enabled": False, "reason": "Disabled by config (ENABLE_CV=0)"}

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
        n_jobs=min(2, _resolve_n_jobs(cfg.cpu_jobs)),
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
