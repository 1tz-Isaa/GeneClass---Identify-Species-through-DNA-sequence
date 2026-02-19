import csv
import json
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from shutil import copy2

from joblib import dump
from sklearn.metrics import classification_report, confusion_matrix

from training.config import TrainConfig
from training.core import safe_name


def send_email_report(report_text: str, cfg: TrainConfig) -> None:
    email_cfg = cfg.email
    required = [email_cfg.host, email_cfg.user, email_cfg.password, email_cfg.to]
    if not all(required):
        print("Email not sent: set SMTP_HOST, SMTP_USER, SMTP_PASSWORD, EMAIL_TO.")
        return

    msg = EmailMessage()
    msg["Subject"] = "Geneclass Training Report"
    msg["From"] = email_cfg.user
    msg["To"] = email_cfg.to
    msg.set_content(report_text)

    with smtplib.SMTP(email_cfg.host, email_cfg.port, timeout=30) as server:
        server.starttls()
        server.login(email_cfg.user, email_cfg.password)
        server.send_message(msg)

    print(f"Email report sent to {email_cfg.to}")


def write_confusion_matrix_csv(path: Path, y_true, y_pred, labels_sorted):
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *labels_sorted])
        for idx, label in enumerate(labels_sorted):
            writer.writerow([label, *cm[idx].tolist()])


def write_classification_report_csv(path: Path, y_true, y_pred, labels_sorted):
    report = classification_report(y_true, y_pred, labels=labels_sorted, output_dict=True, zero_division=0)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "precision", "recall", "f1-score", "support"])
        for key, value in report.items():
            if isinstance(value, dict):
                writer.writerow([
                    key,
                    value.get("precision", ""),
                    value.get("recall", ""),
                    value.get("f1-score", ""),
                    value.get("support", ""),
                ])


def save_model_bundle(model, run_dir: Path, run_id: str, val_accuracy: float, cfg: TrainConfig):
    out = {
        "run_model": "",
        "latest_model": "",
        "best_model": "",
        "best_val_accuracy": "",
        "is_new_best": False,
    }

    if not cfg.save_model:
        return out

    cfg.model_store_root.mkdir(parents=True, exist_ok=True)

    model_slug = safe_name(
        f"{cfg.train_target}_{cfg.label_level}_kmer_lr_{cfg.split_mode}_dedup{int(cfg.dedup_exact)}"
    )
    registry_path = cfg.model_store_root / "best_registry.json"

    registry = {}
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception:
            registry = {}

    run_model_path = run_dir / "model_pipeline.joblib"
    dump(model, run_model_path)
    out["run_model"] = str(run_model_path.resolve())

    latest_model_path = cfg.model_store_root / f"latest_{model_slug}.joblib"
    copy2(run_model_path, latest_model_path)
    out["latest_model"] = str(latest_model_path.resolve())

    key = f"{cfg.train_target}::{cfg.label_level}::kmer_lr::{cfg.split_mode}::dedup{int(cfg.dedup_exact)}"
    prev_best = float(registry.get(key, {}).get("val_accuracy", -1.0))

    best_model_path = cfg.model_store_root / f"best_{model_slug}.joblib"
    if val_accuracy > prev_best:
        copy2(run_model_path, best_model_path)
        registry[key] = {
            "run_id": run_id,
            "val_accuracy": float(val_accuracy),
            "model_path": str(best_model_path.resolve()),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        out["is_new_best"] = True

    if key in registry:
        out["best_model"] = registry[key].get("model_path", str(best_model_path.resolve()))
        out["best_val_accuracy"] = registry[key].get("val_accuracy", "")
    else:
        out["best_model"] = str(best_model_path.resolve())

    return out


def append_history_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    fields = [
        "run_id",
        "timestamp_utc",
        "started_utc",
        "duration_seconds",
        "target",
        "label_level",
        "split_mode",
        "dedup_exact",
        "test_size",
        "kmer_min",
        "kmer_max",
        "kmer_min_df",
        "kmer_max_features",
        "lr_c",
        "lr_max_iter",
        "lr_solver",
        "lr_class_weight",
        "n_samples_total",
        "n_classes_total",
        "n_validation_classes_present",
        "n_validation_classes_missing",
        "n_validation_genera",
        "n_train",
        "n_validation",
        "n_groups_total",
        "n_groups_train",
        "n_groups_validation",
        "min_class_count_total",
        "train_accuracy",
        "train_balanced_accuracy",
        "train_precision_macro",
        "train_recall_macro",
        "train_f1_macro",
        "train_precision_weighted",
        "train_recall_weighted",
        "train_f1_weighted",
        "train_mcc",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_precision_macro",
        "val_recall_macro",
        "val_f1_macro",
        "val_precision_weighted",
        "val_recall_weighted",
        "val_f1_weighted",
        "val_mcc",
        "val_log_loss",
        "cv_enabled",
        "cv_splits",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_balanced_accuracy_mean",
        "cv_balanced_accuracy_std",
        "cv_f1_macro_mean",
        "cv_f1_macro_std",
        "cv_f1_weighted_mean",
        "cv_f1_weighted_std",
        "run_model_path",
        "latest_model_path",
        "best_model_path",
        "is_new_best",
        "run_dir",
    ]

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def build_report(summary: dict) -> str:
    val = summary["metrics"]["validation"]
    cv = summary["cv"]
    art = summary["artifacts"]

    lines = [
        "Training finished.",
        f"Run ID: {summary['run_id']}",
        f"Started (UTC): {summary.get('started_utc', '')}",
        f"Duration (s): {summary.get('duration_seconds', 0.0):.2f}",
        f"Target: {summary['config']['target']}",
        f"Label level: {summary['config']['label_level']}",
        f"Split mode: {summary['config']['split_mode']}",
        f"Dedup exact: {summary['config']['dedup_exact']}",
        f"Samples: {summary['data']['n_samples_total']}",
        f"Classes: {summary['data']['n_classes_total']}",
        (
            "Validation classes: "
            f"{summary['data']['n_validation_classes_present']}/{summary['data']['n_classes_total']} "
            f"(missing {summary['data']['n_validation_classes_missing']})"
        ),
        f"Validation genera checked: {summary['data']['n_validation_genera']}",
        f"Validation Accuracy: {val['accuracy'] * 100:.2f}%",
        f"Validation Balanced Acc: {val['balanced_accuracy'] * 100:.2f}%",
        f"Validation F1 Macro: {val['f1_macro']:.4f}",
        f"Validation F1 Weighted: {val['f1_weighted']:.4f}",
        f"Validation MCC: {val['mcc']:.4f}",
        f"Validation LogLoss: {val.get('log_loss', float('nan')):.6f}",
    ]

    if cv.get("enabled"):
        lines.append(
            f"CV ({cv['n_splits']} folds) Acc={cv['accuracy_mean']:.4f}±{cv['accuracy_std']:.4f}, "
            f"F1m={cv['f1_macro_mean']:.4f}±{cv['f1_macro_std']:.4f}"
        )
    else:
        lines.append(f"CV: disabled ({cv.get('reason', 'n/a')})")

    lines.append(f"Artifacts: {art['run_dir']}")
    lines.append(f"Accuracy timeline: {art['accuracy_timeline_csv']}")

    if art.get("run_model"):
        lines.append(f"Run model: {art['run_model']}")
    if art.get("latest_model"):
        lines.append(f"Latest model: {art['latest_model']}")
    if art.get("best_model"):
        lines.append(f"Best model: {art['best_model']} (acc={art.get('best_val_accuracy', '')})")
    if art.get("is_new_best"):
        lines.append("Best model updated: YES")

    return "\n".join(lines) + "\n"
