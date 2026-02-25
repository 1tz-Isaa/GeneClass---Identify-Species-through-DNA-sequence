"""Training dashboard for experiment tracking.

Run:
  python3 -m streamlit run /Users/isaacluu/Downloads/Geneclass25/Training_Dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


HISTORY_DEFAULT = Path("runs/training_logs/history.csv")

PERCENT_COLUMNS = [
    "train_accuracy",
    "train_balanced_accuracy",
    "train_precision_macro",
    "train_recall_macro",
    "train_f1_macro",
    "train_precision_weighted",
    "train_recall_weighted",
    "train_f1_weighted",
    "val_accuracy",
    "val_balanced_accuracy",
    "val_precision_macro",
    "val_recall_macro",
    "val_f1_macro",
    "val_precision_weighted",
    "val_recall_weighted",
    "val_f1_weighted",
    "cv_accuracy_mean",
    "cv_balanced_accuracy_mean",
    "cv_f1_macro_mean",
    "cv_f1_weighted_mean",
]

NUMERIC_COLUMNS = [
    *PERCENT_COLUMNS,
    "train_mcc",
    "val_mcc",
    "val_log_loss",
    "duration_seconds",
    "kmer_min",
    "kmer_max",
    "kmer_min_df",
    "kmer_max_features",
    "lr_c",
    "lr_max_iter",
    "n_samples_total",
    "n_classes_total",
    "n_train",
    "n_validation",
    "n_groups_total",
    "n_groups_train",
    "n_groups_validation",
    "min_class_count_total",
]

DEFAULT_TABLE_COLUMNS = [
    "run_id",
    "timestamp_utc",
    "target",
    "label_level",
    "val_accuracy",
    "val_balanced_accuracy",
    "val_f1_macro",
    "val_f1_weighted",
    "val_mcc",
    "duration_seconds",
    "n_samples_total",
    "n_classes_total",
    "kmer_min",
    "kmer_max",
    "kmer_max_features",
    "lr_c",
    "lr_max_iter",
    "lr_solver",
    "split_mode",
]


def _is_streamlit_context() -> bool:
    try:
        from streamlit.runtime import exists

        return bool(exists())
    except Exception:
        return False


def _format_percent(value) -> str:
    try:
        if value != value:  # NaN
            return "-"
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "-"


def _format_float(value, digits: int = 4) -> str:
    try:
        if value != value:  # NaN
            return "-"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _safe_columns(df, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def _coerce_run_id(run_dir: Path, summary: dict) -> str:
    run_id = str(summary.get("run_id") or "").strip()
    if run_id:
        return run_id
    name = run_dir.name
    if name.startswith("run_"):
        tail = name[len("run_") :]
        if "_" in tail:
            return tail.split("_", 1)[0]
        return tail
    return name


def _load_history_from_summaries(logs_dir: Path):
    import pandas as pd

    if not logs_dir.exists():
        return pd.DataFrame()

    rows = []
    for summary_path in sorted(logs_dir.glob("run_*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        run_dir = summary_path.parent
        cfg = summary.get("config", {}) or {}
        model = cfg.get("model", {}) or {}
        data = summary.get("data", {}) or {}
        metrics = summary.get("metrics", {}) or {}
        train = metrics.get("train", {}) or {}
        val = metrics.get("validation", {}) or {}
        cv = summary.get("cv", {}) or {}

        rows.append(
            {
                "run_id": _coerce_run_id(run_dir, summary),
                "timestamp_utc": summary.get("timestamp_utc"),
                "started_utc": summary.get("started_utc"),
                "duration_seconds": summary.get("duration_seconds"),
                "target": cfg.get("target"),
                "train_profile": cfg.get("train_profile"),
                "train_preset": cfg.get("train_preset"),
                "label_level": cfg.get("label_level"),
                "split_mode": cfg.get("split_mode"),
                "dedup_exact": cfg.get("dedup_exact"),
                "test_size": cfg.get("test_size"),
                "kmer_min": model.get("kmer_min"),
                "kmer_max": model.get("kmer_max"),
                "kmer_min_df": model.get("kmer_min_df"),
                "kmer_max_features": model.get("kmer_max_features"),
                "lr_c": model.get("lr_c"),
                "lr_max_iter": model.get("lr_max_iter"),
                "lr_solver": model.get("lr_solver"),
                "lr_class_weight": model.get("lr_class_weight"),
                "n_samples_total": data.get("n_samples_total"),
                "n_classes_total": data.get("n_classes_total"),
                "n_validation_classes_present": data.get("n_validation_classes_present"),
                "n_validation_classes_missing": data.get("n_validation_classes_missing"),
                "n_validation_genera": data.get("n_validation_genera"),
                "n_train": data.get("n_train"),
                "n_validation": data.get("n_validation"),
                "n_groups_total": data.get("n_groups_total"),
                "n_groups_train": data.get("n_groups_train"),
                "n_groups_validation": data.get("n_groups_validation"),
                "min_class_count_total": data.get("min_class_count_total"),
                "train_accuracy": train.get("accuracy"),
                "train_balanced_accuracy": train.get("balanced_accuracy"),
                "train_precision_macro": train.get("precision_macro"),
                "train_recall_macro": train.get("recall_macro"),
                "train_f1_macro": train.get("f1_macro"),
                "train_precision_weighted": train.get("precision_weighted"),
                "train_recall_weighted": train.get("recall_weighted"),
                "train_f1_weighted": train.get("f1_weighted"),
                "train_mcc": train.get("mcc"),
                "val_accuracy": val.get("accuracy"),
                "val_balanced_accuracy": val.get("balanced_accuracy"),
                "val_precision_macro": val.get("precision_macro"),
                "val_recall_macro": val.get("recall_macro"),
                "val_f1_macro": val.get("f1_macro"),
                "val_precision_weighted": val.get("precision_weighted"),
                "val_recall_weighted": val.get("recall_weighted"),
                "val_f1_weighted": val.get("f1_weighted"),
                "val_mcc": val.get("mcc"),
                "val_log_loss": val.get("log_loss"),
                "cv_enabled": cv.get("enabled"),
                "cv_splits": cv.get("n_splits"),
                "cv_accuracy_mean": cv.get("accuracy_mean"),
                "cv_accuracy_std": cv.get("accuracy_std"),
                "cv_balanced_accuracy_mean": cv.get("balanced_accuracy_mean"),
                "cv_balanced_accuracy_std": cv.get("balanced_accuracy_std"),
                "cv_f1_macro_mean": cv.get("f1_macro_mean"),
                "cv_f1_macro_std": cv.get("f1_macro_std"),
                "cv_f1_weighted_mean": cv.get("f1_weighted_mean"),
                "cv_f1_weighted_std": cv.get("f1_weighted_std"),
                "run_dir": str(run_dir.resolve()),
            }
        )

    return pd.DataFrame(rows)


def _postprocess_history_df(df):
    import pandas as pd

    if df.empty:
        return df

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
        df = df.sort_values("timestamp_utc").reset_index(drop=True)

    if "started_utc" in df.columns:
        df["started_utc"] = pd.to_datetime(df["started_utc"], errors="coerce", utc=True)

    if "run_id" not in df.columns and "run_dir" in df.columns:
        df["run_id"] = df["run_dir"].astype(str).apply(lambda x: Path(x).name.replace("run_", ""))

    if "run_dir" in df.columns:
        df = df[df["run_dir"].astype(str).str.len() > 0]

    if "run_id" in df.columns:
        df = df.dropna(subset=["run_id"])
        df = df[df["run_id"].astype(str).str.len() > 0]
        if "timestamp_utc" in df.columns:
            df = df.sort_values(["timestamp_utc", "run_id"]).drop_duplicates(
                subset=["run_id"], keep="last"
            )
        else:
            df = df.drop_duplicates(subset=["run_id"], keep="last")
        df = df.reset_index(drop=True)

    df["run_index"] = range(1, len(df) + 1)
    return df


def load_history_df(path: Path):
    import pandas as pd

    summary_df = _postprocess_history_df(_load_history_from_summaries(path.parent))

    if not path.exists():
        return summary_df

    df = pd.DataFrame()
    try:
        # Fast path when CSV schema is consistent.
        df = pd.read_csv(path)
    except Exception:
        try:
            # Tolerate malformed mixed-schema rows.
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            df = pd.DataFrame()

    # Mixed old/new history formats can still decode but become semantically invalid.
    looks_valid = (
        (not df.empty)
        and ("run_id" in df.columns)
        and ("timestamp_utc" in df.columns)
        and ("target" in df.columns)
    )
    if not looks_valid:
        return summary_df

    csv_df = _postprocess_history_df(df)

    # Prefer summary-derived history when mixed-schema CSV drops many rows.
    if not summary_df.empty and len(summary_df) > len(csv_df):
        return summary_df

    return csv_df


def render_dashboard() -> None:
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="Geneclass Training Dashboard", page_icon="CHART", layout="wide")
    st.title("Geneclass Training Dashboard")
    st.caption("Track every training run with tables and graphs.")

    path_input = st.sidebar.text_input("History CSV", str(HISTORY_DEFAULT))
    history_path = Path(path_input)
    df = load_history_df(history_path)

    if df.empty:
        st.error(f"No history data found at: {history_path.resolve()}")
        st.info("Run training first with Main_Train.py to generate history.csv")
        return

    target_options = sorted(df["target"].dropna().astype(str).unique().tolist()) if "target" in df.columns else []
    level_options = (
        sorted(df["label_level"].dropna().astype(str).unique().tolist()) if "label_level" in df.columns else []
    )

    selected_targets = st.sidebar.multiselect("Target filter", target_options, default=target_options)
    selected_levels = st.sidebar.multiselect("Label filter", level_options, default=level_options)

    dff = df.copy()
    if selected_targets and "target" in dff.columns:
        dff = dff[dff["target"].isin(selected_targets)]
    if selected_levels and "label_level" in dff.columns:
        dff = dff[dff["label_level"].isin(selected_levels)]

    if dff.empty:
        st.warning("No rows after filter.")
        return

    # Top summary table by target.
    st.subheader("Best And Latest Per Target")
    if "target" in dff.columns:
        latest_rows = (
            dff.sort_values("timestamp_utc")
            .groupby("target", as_index=False)
            .tail(1)
            .sort_values("target")
        )
        best_rows = (
            dff.sort_values("val_accuracy", ascending=False)
            .groupby("target", as_index=False)
            .head(1)
            .sort_values("target")
        )
        merge_cols = ["target", "run_id", "val_accuracy", "val_f1_macro", "val_mcc", "duration_seconds"]
        latest_small = latest_rows[_safe_columns(latest_rows, merge_cols)].rename(
            columns={
                "run_id": "latest_run",
                "val_accuracy": "latest_val_accuracy",
                "val_f1_macro": "latest_val_f1_macro",
                "val_mcc": "latest_val_mcc",
                "duration_seconds": "latest_duration_sec",
            }
        )
        best_small = best_rows[_safe_columns(best_rows, merge_cols)].rename(
            columns={
                "run_id": "best_run",
                "val_accuracy": "best_val_accuracy",
                "val_f1_macro": "best_val_f1_macro",
                "val_mcc": "best_val_mcc",
                "duration_seconds": "best_duration_sec",
            }
        )
        summary_table = latest_small.merge(best_small, on="target", how="left")

        for col in ["latest_val_accuracy", "latest_val_f1_macro", "best_val_accuracy", "best_val_f1_macro"]:
            if col in summary_table.columns:
                summary_table[col] = summary_table[col].apply(_format_percent)
        for col in ["latest_val_mcc", "best_val_mcc"]:
            if col in summary_table.columns:
                summary_table[col] = summary_table[col].apply(_format_float)
        for col in ["latest_duration_sec", "best_duration_sec"]:
            if col in summary_table.columns:
                summary_table[col] = summary_table[col].apply(lambda x: _format_float(x, digits=2))

        st.dataframe(summary_table, use_container_width=True)

    # Full runs table.
    st.subheader("All Runs Table")
    available_cols = _safe_columns(dff, DEFAULT_TABLE_COLUMNS)
    selected_cols = st.multiselect("Columns", options=dff.columns.tolist(), default=available_cols)
    table_df = dff[selected_cols].copy()

    # User-friendly format.
    for col in selected_cols:
        if col in PERCENT_COLUMNS:
            table_df[col] = table_df[col].apply(_format_percent)
        elif col in ["train_mcc", "val_mcc", "val_log_loss", "duration_seconds", "lr_c"]:
            digits = 2 if col == "duration_seconds" else 4
            table_df[col] = table_df[col].apply(lambda x, d=digits: _format_float(x, d))
        elif "timestamp" in col:
            table_df[col] = table_df[col].astype(str)

    st.dataframe(table_df, use_container_width=True, height=360)
    st.download_button(
        label="Download filtered runs CSV",
        data=dff.to_csv(index=False).encode("utf-8"),
        file_name="training_runs_filtered.csv",
        mime="text/csv",
    )

    # Graphs.
    st.subheader("Graphs")
    x_axis = st.radio("X-axis", ["run_index", "timestamp_utc"], index=0, horizontal=True)
    metrics = st.multiselect(
        "Metrics to plot",
        options=["val_accuracy", "val_f1_macro", "val_mcc", "val_balanced_accuracy", "train_accuracy"],
        default=["val_accuracy", "val_f1_macro", "val_mcc"],
    )

    for metric in metrics:
        if metric not in dff.columns:
            continue

        chart_df = dff[_safe_columns(dff, [x_axis, "target", metric])].dropna()
        if chart_df.empty:
            continue

        if x_axis == "timestamp_utc":
            chart_df = chart_df.sort_values("timestamp_utc")
        else:
            chart_df = chart_df.sort_values("run_index")

        pivot = chart_df.pivot_table(index=x_axis, columns="target", values=metric, aggfunc="last")
        st.write(f"{metric} by target")
        st.line_chart(pivot, use_container_width=True)

    # Best scores by target.
    if "target" in dff.columns:
        st.subheader("Best Score By Target")
        for metric in ["val_accuracy", "val_f1_macro", "val_mcc"]:
            if metric not in dff.columns:
                continue
            best_by_target = dff.groupby("target", as_index=True)[metric].max().sort_values(ascending=False)
            st.write(metric)
            st.bar_chart(best_by_target, use_container_width=True)

    # Per-run inspector.
    st.subheader("Run Inspector")
    run_ids = dff["run_id"].dropna().astype(str).tolist() if "run_id" in dff.columns else []
    if not run_ids:
        st.info("No run_id available.")
        return

    selected_run = st.selectbox("Select run_id", run_ids, index=max(0, len(run_ids) - 1))
    selected_row = dff[dff["run_id"].astype(str) == selected_run].tail(1)
    if selected_row.empty:
        return

    run_dir_value = str(selected_row.iloc[0].get("run_dir", ""))
    if not run_dir_value:
        st.info("No run_dir path in history for this run.")
        return

    run_dir = Path(run_dir_value)
    st.code(str(run_dir), language="text")

    summary_path = run_dir / "summary.json"
    timeline_path = run_dir / "accuracy_timeline_by_genus.csv"

    if summary_path.exists():
        import json

        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        with st.expander("summary.json", expanded=False):
            st.json(summary_data)

    if timeline_path.exists():
        tl = pd.read_csv(timeline_path)
        if not tl.empty:
            cols = _safe_columns(tl, ["step", "cumulative_accuracy", "genus_accuracy"])
            if "step" in cols:
                st.write("Per-genus checking timeline")
                plot_df = tl[cols].set_index("step")
                st.line_chart(plot_df, use_container_width=True)


if __name__ == "__main__":
    if _is_streamlit_context():
        render_dashboard()
    else:
        app_path = Path(__file__).resolve()
        print("[INFO] This is a Streamlit app.")
        print(f"[INFO] Run with: streamlit run {app_path}")
