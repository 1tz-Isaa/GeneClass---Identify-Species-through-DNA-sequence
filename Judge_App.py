"""Simple judge app: load a trained model and predict from barcode/FASTA.

Run:
  python3 -m streamlit run /Users/isaacluu/Downloads/Geneclass25/Judge_App.py
"""

from __future__ import annotations

import random
from pathlib import Path

from joblib import load
from clinical_mapping import infer_closest_disease, rank_disease_groups
from central_reader import (
    parse_fasta_text,
    predict_with_model,
    run_central_reader,
)

def list_models() -> list[Path]:
    model_dir = Path("runs/saved_models")
    if not model_dir.exists():
        return []

    candidates = sorted(model_dir.glob("best_*.joblib")) + sorted(model_dir.glob("latest_*.joblib"))
    # Remove duplicates while preserving order.
    unique = []
    seen = set()
    for p in candidates:
        if p.name in seen:
            continue
        seen.add(p.name)
        unique.append(p)
    return unique


def pick_random_dataset_fasta() -> Path | None:
    roots = [Path("DNA"), Path("RNA")]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(root.rglob("*.fasta"))
    if not files:
        return None
    return random.choice(files)


def fasta_ground_truth(file_path: Path) -> str:
    # Expected structure: Domain/Kingdom/Genus/Species/file.fasta
    parts = file_path.parts
    try:
        idx = parts.index("DNA")
    except ValueError:
        try:
            idx = parts.index("RNA")
        except ValueError:
            return "unknown"

    tail = parts[idx:]
    if len(tail) >= 4:
        return f"{tail[2]} / {tail[3]}"
    if len(tail) >= 3:
        return f"{tail[2]}"
    return "unknown"


def main() -> None:
    import streamlit as st

    def pct(value) -> str:
        return f"{float(value) * 100:.2f}%"

    st.set_page_config(page_title="Geneclass Judge App", page_icon="DNA", layout="wide")
    st.title("Geneclass Judge App")
    st.caption("Upload FASTA/barcode and predict genus labels. Supports manual model or auto kingdom routing.")
    st.caption("Research support only. Clinical mapping is suggestive and not a medical diagnosis.")

    model_paths = list_models()
    if not model_paths:
        st.error("No model found in runs/saved_models. Train first with Main_Train.py")
        return

    model_map = {p.name: p for p in model_paths}
    mode = st.sidebar.radio("Mode", ["Auto kingdom (central)", "Manual model"], index=0)
    model_source = st.sidebar.selectbox("Model source (auto mode)", ["best", "latest"], index=0)
    force_kingdom = st.sidebar.selectbox(
        "Force kingdom (auto mode)",
        ["Auto", "Bacteria", "Fungi", "Viruses"],
        index=0,
    )
    reject_threshold = st.sidebar.slider("Reject threshold", min_value=0.0, max_value=0.95, value=0.35, step=0.01)

    selected_name = st.sidebar.selectbox("Model (manual mode)", list(model_map.keys()), index=0)
    model_path = model_map[selected_name]

    with st.sidebar:
        if mode == "Manual model":
            st.write("Model path")
            st.code(str(model_path), language="text")
        else:
            st.write("Auto mode")
            st.code("Route kingdom first, then pick best model automatically.", language="text")

    tab_upload, tab_paste, tab_random = st.tabs(["Upload FASTA", "Paste barcode", "Random dataset test"])

    records: list[tuple[str, str]] = []
    source_note = ""

    with tab_upload:
        up = st.file_uploader("Upload .fasta/.fa/.txt", type=["fasta", "fa", "txt"])
        if up is not None:
            text = up.read().decode("utf-8", errors="ignore")
            records = parse_fasta_text(text)
            source_note = f"Uploaded file: {up.name}"

    with tab_paste:
        pasted = st.text_area("Paste FASTA or raw sequence", height=180)
        if pasted.strip():
            records = parse_fasta_text(pasted)
            source_note = "Pasted input"

    with tab_random:
        if st.button("Pick random FASTA from dataset"):
            chosen = pick_random_dataset_fasta()
            if chosen is None:
                st.warning("No FASTA file found under DNA/ or RNA/")
            else:
                text = chosen.read_text(encoding="utf-8", errors="ignore")
                records = parse_fasta_text(text)
                gt = fasta_ground_truth(chosen)
                source_note = f"Random file: {chosen} | expected path label: {gt}"

    if st.button("Predict", type="primary"):
        if not records:
            st.warning("No valid sequence found.")
            return

        if mode == "Manual model":
            model = load(model_path)
            raw = predict_with_model(model, records, top_k=5, reject_threshold=reject_threshold)
            preds = [
                {
                    "header": row["header"],
                    "length": row["length"],
                    "prediction": row["prediction"],
                    "confidence": row["confidence"],
                    "top5": row["top"],
                }
                for row in raw
            ]
        else:
            force_target = None
            if force_kingdom == "Bacteria":
                force_target = "bacteria"
            elif force_kingdom == "Fungi":
                force_target = "fungi"
            elif force_kingdom == "Viruses":
                force_target = "rna"

            routed = run_central_reader(
                records,
                model_source=model_source,
                reject_threshold=reject_threshold,
                force_target=force_target,
            )
            st.success(
                "Detected kingdom: "
                f"{routed['selected_kingdom']} (target={routed['selected_target']})"
            )
            st.caption(f"Routing method: {routed.get('routing_method', 'unknown')}")
            st.caption(f"Selected model: {routed['selected_model_path']}")
            st.subheader("Routing scores")
            if routed["routing_scores"]:
                routing_view = []
                for row in routed["routing_scores"]:
                    view = {
                        "target": row.get("target"),
                        "kingdom": row.get("kingdom"),
                        "score": pct(row.get("score", 0.0)),
                    }
                    if "router_score" in row:
                        view["router_score"] = pct(row.get("router_score", 0.0))
                    if "evidence_score" in row:
                        view["evidence_score"] = pct(row.get("evidence_score", 0.0))
                    if "header_hint_score" in row:
                        view["header_hint_score"] = pct(row.get("header_hint_score", 0.0))
                    routing_view.append(view)
                st.table(routing_view)
            else:
                st.info("Routing scores skipped due to manual override.")
            if routed.get("fit_scores"):
                fit_view = []
                for target, item in routed["fit_scores"].items():
                    fit_view.append(
                        {
                            "target": target,
                            "fit_score": pct(item.get("score", 0.0)),
                            "records_used": int(item.get("n_records", 0)),
                        }
                    )
                st.subheader("Cross-model fit")
                st.table(fit_view)
            preds = [
                {
                    "header": row["header"],
                    "length": row["length"],
                    "prediction": row["prediction"],
                    "confidence": row["confidence"],
                    "top5": row["top"],
                }
                for row in routed["predictions"]
            ]

        st.success(f"Predicted {len(preds)} sequence(s)")
        if source_note:
            st.caption(source_note)

        for idx, row in enumerate(preds, start=1):
            st.markdown(f"### Result {idx}")
            st.write(f"Header: `{row['header']}`")
            st.write(f"Length: `{row['length']}`")
            st.write(f"Prediction: `{row['prediction']}`")
            if row["confidence"] is not None:
                st.write(f"Top-1 confidence: `{pct(row['confidence'])}`")

            clinical = infer_closest_disease(row["prediction"], row.get("top5") or [])
            if clinical.get("matched") == "1":
                st.write(f"Closest disease group: `{clinical['disease_group']}`")
                st.write(f"Pathogen panel: `{clinical['pathogen_panel']}`")
                st.write(f"Clinical hint: `{clinical['syndrome_hint']}`")
            else:
                st.write("Closest disease group: `Unknown / not mapped`")

            ranked_groups = rank_disease_groups(row.get("top5") or [], top_n=3)
            if ranked_groups:
                group_view = [
                    {
                        "pathogen_panel": x["pathogen_panel"],
                        "disease_group": x["disease_group"],
                        "score": pct(x["score"]),
                        "supporting_genera": x["supporting_genera"],
                    }
                    for x in ranked_groups
                ]
                st.write("Closest disease groups from top predictions")
                st.table(group_view)

            if row["top5"]:
                top_view = [{"label": item["label"], "score": pct(item["score"])} for item in row["top5"]]
                st.table(top_view)


def _is_streamlit_context() -> bool:
    try:
        from streamlit.runtime import exists

        return bool(exists())
    except Exception:
        return False


if __name__ == "__main__":
    if _is_streamlit_context():
        main()
    else:
        app_path = Path(__file__).resolve()
        print("[INFO] This is a Streamlit app.")
        print(f"[INFO] Run with: streamlit run {app_path}")
