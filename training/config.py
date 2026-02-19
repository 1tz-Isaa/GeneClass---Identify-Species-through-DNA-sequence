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
    label_level: str
    test_size: float
    random_state: int
    cv_folds: int
    group_split_tries: int
    split_mode: str
    dedup_exact: bool
    filter_bad_headers: bool
    min_seq_len: int
    show_target_table: bool
    show_file_progress: bool
    show_check_progress: bool
    kmer_min: int
    kmer_max: int
    kmer_min_df: int
    kmer_max_features: int
    lr_c: float
    lr_max_iter: int
    lr_solver: str
    lr_class_weight: str | None
    runs_root: Path
    save_model: bool
    model_store_root: Path
    email: EmailConfig


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) == "1"


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
    train_target_input = os.getenv("TRAIN_TARGET", "bacteria").strip().lower()
    train_target = TARGET_ALIASES.get(train_target_input, train_target_input)

    label_level_input = os.getenv("LABEL_LEVEL", "genus").strip().lower()
    label_level = "species" if label_level_input == "species" else "genus"

    return TrainConfig(
        train_target_input=train_target_input,
        train_target=train_target,
        label_level=label_level,
        test_size=float(os.getenv("TEST_SIZE", "0.3")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        cv_folds=int(os.getenv("CV_FOLDS", "5")),
        group_split_tries=int(os.getenv("GROUP_SPLIT_TRIES", "40")),
        split_mode=os.getenv("SPLIT_MODE", "group_species").strip().lower(),
        dedup_exact=_env_flag("DEDUP_EXACT", "1"),
        filter_bad_headers=_env_flag("FILTER_BAD_HEADERS", "1"),
        min_seq_len=int(os.getenv("MIN_SEQ_LEN", "200")),
        show_target_table=_env_flag("SHOW_TARGET_TABLE", "1"),
        show_file_progress=_env_flag("SHOW_FILE_PROGRESS", "1"),
        show_check_progress=_env_flag("SHOW_CHECK_PROGRESS", "1"),
        kmer_min=int(os.getenv("KMER_MIN", "4")),
        kmer_max=int(os.getenv("KMER_MAX", "6")),
        kmer_min_df=int(os.getenv("KMER_MIN_DF", "2")),
        kmer_max_features=int(os.getenv("KMER_MAX_FEATURES", "120000")),
        lr_c=float(os.getenv("LR_C", "4.0")),
        lr_max_iter=int(os.getenv("LR_MAX_ITER", "12000")),
        lr_solver=os.getenv("LR_SOLVER", "saga").strip().lower(),
        lr_class_weight=(
            os.getenv("LR_CLASS_WEIGHT", "balanced").strip()
            if os.getenv("LR_CLASS_WEIGHT", "balanced").strip()
            else None
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
