from dataclasses import dataclass
from pathlib import Path
import os


TARGET_CONFIG = {
    "bacteria": {"root": "DNA", "kingdom": "Bacteria"},
    "fungi": {"root": "DNA", "kingdom": "Fungi"},
    "rna": {"root": "RNA", "kingdom": "Viruses"},
}

TARGET_ALIASES = {
    "1": "bacteria",
    "2": "fungi",
    "3": "rna",
    "b": "bacteria",
    "f": "fungi",
    "v": "rna",
    "virus": "rna",
    "viruses": "rna",
    "bacteria": "bacteria",
    "fungi": "fungi",
    "fungus": "fungi",
    "rna": "rna",
}


TRAIN_PRESET_DEFAULTS = {
    "turbo": {
        "test_size": 0.30,
        "cv_folds": 2,
        "enable_cv": False,
        "group_split_tries": 8,
        "kmer_min": 4,
        "kmer_max": 5,
        "kmer_min_df": 2,
        "kmer_max_features": 12000,
        "cpu_jobs": 0,
        "lr_c": 3.0,
        "lr_max_iter": 900,
        "lr_tol": 0.002,
        "lr_class_weight": "",
        "auto_tune_lr_c": False,
        "auto_tune_max_samples": 1200,
        "auto_tune_holdout_size": 0.15,
        "train_fragment_len": 1200,
        "max_seq_len": 1800,
        "max_samples_total": 450,
        "max_samples_per_label": 45,
    },
    "fast": {
        "test_size": 0.30,
        "cv_folds": 2,
        "enable_cv": False,
        "group_split_tries": 10,
        "kmer_min": 4,
        "kmer_max": 6,
        "kmer_min_df": 2,
        "kmer_max_features": 25000,
        "cpu_jobs": 0,
        "lr_c": 3.0,
        "lr_max_iter": 1800,
        "lr_tol": 0.001,
        "lr_class_weight": "",
        "auto_tune_lr_c": False,
        "auto_tune_max_samples": 2000,
        "auto_tune_holdout_size": 0.15,
        "train_fragment_len": 1400,
        "max_seq_len": 2500,
        "max_samples_total": 700,
        "max_samples_per_label": 90,
    },
    "balanced": {
        "test_size": 0.30,
        "cv_folds": 3,
        "enable_cv": False,
        "group_split_tries": 18,
        "kmer_min": 4,
        "kmer_max": 6,
        "kmer_min_df": 2,
        "kmer_max_features": 60000,
        "cpu_jobs": 0,
        "lr_c": 4.0,
        "lr_max_iter": 4500,
        "lr_tol": 0.0005,
        "lr_class_weight": "",
        "auto_tune_lr_c": False,
        "auto_tune_max_samples": 5000,
        "auto_tune_holdout_size": 0.16,
        "train_fragment_len": 1600,
        "max_seq_len": 6000,
        "max_samples_total": 0,
        "max_samples_per_label": 0,
    },
    "accuracy": {
        "test_size": 0.30,
        "cv_folds": 3,
        "enable_cv": False,
        "group_split_tries": 90,
        "kmer_min": 5,
        "kmer_max": 5,
        "kmer_min_df": 2,
        "kmer_max_features": 50000,
        "cpu_jobs": 0,
        "lr_c": 2.0,
        "lr_max_iter": 5000,
        "lr_tol": 0.001,
        "lr_class_weight": "balanced",
        "auto_tune_lr_c": False,
        "auto_tune_max_samples": 4000,
        "auto_tune_holdout_size": 0.16,
        "train_fragment_len": 1200,
        "max_seq_len": 0,
        "max_samples_total": 0,
        "max_samples_per_label": 0,
    },
}


@dataclass(frozen=True)
class EmailConfig:
    host: str | None
    port: int
    user: str | None
    password: str | None
    to: str | None


@dataclass(frozen=True)
class TrainConfig:
    train_target_input: str
    train_target: str
    train_preset: str
    label_level: str
    test_size: float
    random_state: int
    cv_folds: int
    enable_cv: bool
    group_split_tries: int
    split_mode: str
    dedup_exact: bool
    filter_bad_headers: bool
    min_seq_len: int
    train_fragment_len: int
    max_seq_len: int
    max_samples_total: int
    max_samples_per_label: int
    rna_min_unique_genomes_per_label: int
    rna_min_samples_per_label: int
    rna_use_family_fragment_len: bool
    rna_family_top_k: int
    rna_hierarchical_weight: float
    rna_collapse_nested_species: bool
    show_target_table: bool
    show_file_progress: bool
    show_check_progress: bool
    kmer_min: int
    kmer_max: int
    kmer_min_df: int
    kmer_max_features: int
    cpu_jobs: int
    lr_c: float
    lr_max_iter: int
    lr_tol: float
    lr_solver: str
    lr_class_weight: str | None
    auto_tune_lr_c: bool
    auto_tune_max_samples: int
    auto_tune_holdout_size: float
    runs_root: Path
    save_model: bool
    model_store_root: Path
    email: EmailConfig


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) == "1"


def _resolve_preset() -> tuple[str, dict]:
    preset = os.getenv("TRAIN_PRESET", "fast").strip().lower()
    if preset not in TRAIN_PRESET_DEFAULTS:
        preset = "fast"
    return preset, TRAIN_PRESET_DEFAULTS[preset]


def format_target_table() -> str:
    return "\n".join(
        [
            "Target Table",
            "key | target   | data folder",
            "----+----------+------------",
            "1   | bacteria | DNA/Bacteria",
            "2   | fungi    | DNA/Fungi",
            "3   | rna      | RNA/Viruses",
        ]
    )


def load_config() -> TrainConfig:
    train_preset, preset_defaults = _resolve_preset()
    train_target_input = os.getenv("TRAIN_TARGET", "bacteria").strip().lower()
    train_target = TARGET_ALIASES.get(train_target_input, train_target_input)

    label_level_input = os.getenv("LABEL_LEVEL", "genus").strip().lower()
    label_level = "species" if label_level_input == "species" else "genus"
    rna_mode = train_target == "rna"

    return TrainConfig(
        train_target_input=train_target_input,
        train_target=train_target,
        train_preset=train_preset,
        label_level=label_level,
        test_size=float(os.getenv("TEST_SIZE", str(preset_defaults["test_size"]))),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        cv_folds=int(os.getenv("CV_FOLDS", str(preset_defaults["cv_folds"]))),
        enable_cv=_env_flag("ENABLE_CV", "1" if preset_defaults["enable_cv"] else "0"),
        group_split_tries=int(os.getenv("GROUP_SPLIT_TRIES", str(preset_defaults["group_split_tries"]))),
        split_mode=os.getenv("SPLIT_MODE", "group_species").strip().lower(),
        dedup_exact=_env_flag("DEDUP_EXACT", "1"),
        filter_bad_headers=_env_flag("FILTER_BAD_HEADERS", "1"),
        min_seq_len=int(os.getenv("MIN_SEQ_LEN", "200")),
        train_fragment_len=int(os.getenv("TRAIN_FRAGMENT_LEN", str(preset_defaults["train_fragment_len"]))),
        max_seq_len=int(os.getenv("MAX_SEQ_LEN", str(preset_defaults["max_seq_len"]))),
        max_samples_total=int(os.getenv("MAX_SAMPLES_TOTAL", str(preset_defaults["max_samples_total"]))),
        max_samples_per_label=int(
            os.getenv("MAX_SAMPLES_PER_LABEL", str(preset_defaults["max_samples_per_label"]))
        ),
        rna_min_unique_genomes_per_label=int(
            os.getenv("RNA_MIN_UNIQUE_GENOMES_PER_LABEL", "5" if rna_mode else "0")
        ),
        rna_min_samples_per_label=int(os.getenv("RNA_MIN_SAMPLES_PER_LABEL", "0")),
        rna_use_family_fragment_len=_env_flag("RNA_USE_FAMILY_FRAGMENT_LEN", "0"),
        rna_family_top_k=int(os.getenv("RNA_FAMILY_TOP_K", "2" if rna_mode else "0")),
        rna_hierarchical_weight=float(os.getenv("RNA_HIERARCHICAL_WEIGHT", "0.0")),
        rna_collapse_nested_species=_env_flag("RNA_COLLAPSE_NESTED_SPECIES", "1" if rna_mode else "0"),
        show_target_table=_env_flag("SHOW_TARGET_TABLE", "0"),
        show_file_progress=_env_flag("SHOW_FILE_PROGRESS", "0"),
        show_check_progress=_env_flag("SHOW_CHECK_PROGRESS", "0"),
        kmer_min=int(os.getenv("KMER_MIN", str(preset_defaults["kmer_min"]))),
        kmer_max=int(os.getenv("KMER_MAX", str(preset_defaults["kmer_max"]))),
        kmer_min_df=int(os.getenv("KMER_MIN_DF", str(preset_defaults["kmer_min_df"]))),
        kmer_max_features=int(os.getenv("KMER_MAX_FEATURES", str(preset_defaults["kmer_max_features"]))),
        cpu_jobs=int(os.getenv("CPU_JOBS", str(preset_defaults["cpu_jobs"]))),
        lr_c=float(os.getenv("LR_C", str(preset_defaults["lr_c"]))),
        lr_max_iter=int(os.getenv("LR_MAX_ITER", str(preset_defaults["lr_max_iter"]))),
        lr_tol=float(os.getenv("LR_TOL", str(preset_defaults["lr_tol"]))),
        lr_solver=os.getenv("LR_SOLVER", "saga").strip().lower(),
        lr_class_weight=(
            os.getenv("LR_CLASS_WEIGHT", str(preset_defaults["lr_class_weight"])).strip()
            if os.getenv("LR_CLASS_WEIGHT", str(preset_defaults["lr_class_weight"])).strip()
            else None
        ),
        auto_tune_lr_c=_env_flag("AUTO_TUNE_LR_C", "1" if preset_defaults["auto_tune_lr_c"] else "0"),
        auto_tune_max_samples=int(
            os.getenv("AUTO_TUNE_MAX_SAMPLES", str(preset_defaults["auto_tune_max_samples"]))
        ),
        auto_tune_holdout_size=float(
            os.getenv("AUTO_TUNE_HOLDOUT_SIZE", str(preset_defaults["auto_tune_holdout_size"]))
        ),
        runs_root=Path(os.getenv("RUNS_ROOT", "runs/training_logs")),
        save_model=_env_flag("SAVE_MODEL", "1"),
        model_store_root=Path(os.getenv("MODEL_STORE_ROOT", "runs/saved_models")),
        email=EmailConfig(
            host=os.getenv("SMTP_HOST"),
            port=int(os.getenv("SMTP_PORT", "587")),
            user=os.getenv("SMTP_USER"),
            password=os.getenv("SMTP_PASSWORD"),
            to=os.getenv("EMAIL_TO"),
        ),
    )
