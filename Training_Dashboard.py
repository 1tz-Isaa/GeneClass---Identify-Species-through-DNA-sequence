"""Training dashboard for experiment tracking.

Run:
  streamlit run /Users/isaacluu/Downloads/Geneclass25/Training_Dashboard.py
"""

from __future__ import annotations

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


def load_history_df(path: Path):
    import pandas as pd

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
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

    df["run_index"] = range(1, len(df) + 1)
    return df


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
