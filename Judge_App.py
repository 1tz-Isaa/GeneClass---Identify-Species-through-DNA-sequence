"""Simple judge app: load a trained model and predict from barcode/FASTA.

Run:
  python3 -m streamlit run /Users/isaacluu/Downloads/Geneclass25/Judge_App.py
"""

from __future__ import annotations

import random
from pathlib import Path

from clinical_mapping import infer_closest_disease, rank_disease_groups
from central_reader import (
    parse_fasta_text,
    run_central_reader,
)


DATASET_ROOTS = [
    Path("Database/bacteria_genus"),
    Path("Database/fungi_genus"),
    Path("Database/rna_genus"),
    Path("DNA"),
    Path("RNA"),
]

KINGDOM_OPTIONS = {
    "Auto": None,
    "Bacteria": "bacteria",
    "Fungi": "fungi",
    "Viruses": "rna",
}

TARGET_LABELS = {
    "bacteria": "Bacteria",
    "fungi": "Fungi",
    "rna": "Viruses",
}


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


def app_model_summary() -> list[dict]:
    rows = []
    model_dir = Path("runs/saved_models")
    for target, label in TARGET_LABELS.items():
        p = model_dir / f"best_{target}_genus_kmer_lr_app.joblib"
        summary_path = model_dir / f"{target}_app_model_summary.json"
        genera = "unknown"
        source_sequences = "unknown"
        if summary_path.exists():
            try:
                import json

                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                genera = summary.get("genera", "unknown")
                source_sequences = summary.get("source_sequences", "unknown")
            except Exception:
                pass
        rows.append(
            {
                "Kingdom": label,
                "Model": p.name if p.exists() else "missing",
                "Trained classes": genera,
                "Source sequences": source_sequences,
            }
        )
    family_path = model_dir / "best_rna_family_kmer_lr_app.joblib"
    family_summary_path = model_dir / "rna_family_app_model_summary.json"
    families = "unknown"
    source_sequences = "unknown"
    if family_summary_path.exists():
        try:
            import json

            summary = json.loads(family_summary_path.read_text(encoding="utf-8"))
            families = summary.get("families", summary.get("classes", "unknown"))
            source_sequences = summary.get("source_sequences", "unknown")
        except Exception:
            pass
    rows.append(
        {
            "Kingdom": "Viruses family route",
            "Model": family_path.name if family_path.exists() else "missing",
            "Trained classes": families,
            "Source sequences": source_sequences,
        }
    )
    return rows


def pick_random_dataset_fasta() -> Path | None:
    files: list[Path] = []
    for root in DATASET_ROOTS:
        if root.exists():
            files.extend(root.rglob("*.fasta"))
    if not files:
        return None
    return random.choice(files)


def fasta_ground_truth(file_path: Path) -> str:
    # Expected structures:
    #   Database/<target>_genus/Genus/Species/file.fasta
    #   DNA|RNA/Kingdom/Genus/Species/file.fasta
    parts = file_path.parts
    try:
        idx = parts.index("Database")
        tail = parts[idx:]
        if len(tail) >= 5:
            return f"{tail[2]} / {tail[3]}"
        if len(tail) >= 4:
            return f"{tail[2]}"
    except ValueError:
        pass

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


def pct(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def text_value(value) -> str:
    if value is None:
        return "n/a"
    return str(value)


def result_level(row: dict) -> str:
    score = float(row.get("confidence") or 0.0)
    margin = float(row.get("margin") or 0.0)
    status = row.get("status") or ""
    if status == "uncertain_unsupported_header":
        return "Unsupported input"
    if status == "family_predicted_genus_uncertain":
        return "Family match"
    if row.get("prediction") == "UNCERTAIN":
        return "Possible match" if row.get("raw_prediction") else "Uncertain"
    if score >= 0.45 or margin >= 0.20:
        return "Strong match"
    if score >= 0.30 or margin >= 0.12:
        return "Possible match"
    return "Weak match"


def suggestion_for(row: dict) -> str:
    level = result_level(row)
    closest = row.get("raw_prediction") or row.get("prediction") or "unknown"
    status = row.get("status") or ""
    if status == "uncertain_unsupported_header":
        return "Use a cleaner FASTA record; this input looks like patent/vector/synthetic or otherwise unsupported metadata."
    if status == "family_predicted_genus_uncertain":
        family = row.get("family_prediction") or "unknown family"
        closest_genus = row.get("raw_prediction") or "unknown genus"
        return f"Report the virus family as {family}; keep genus uncertain, with {closest_genus} only as the closest trained genus."
    if level == "Strong match":
        return f"Report {closest} as the selected genus match for this research demo."
    if level == "Possible match":
        return f"Treat {closest} as the closest trained genus, but keep the result labeled as a possible match."
    if level == "Weak match":
        return f"Do not overstate the genus; report {closest} only as a weak closest match."
    return "Report the kingdom result only and test a longer or cleaner sequence before naming a genus."


def build_conclusion_rows(result: dict) -> list[dict]:
    rows = []
    for idx, row in enumerate(result.get("predictions", []), start=1):
        closest = text_value(row.get("raw_prediction") or row.get("prediction"))
        reported = text_value(row.get("prediction"))
        rows.append(
            {
                "#": idx,
                "Detected kingdom": text_value(result.get("selected_kingdom")),
                "Model group": text_value(TARGET_LABELS.get(result.get("selected_target"), result.get("selected_target"))),
                "Result": result_level(row),
                "Virus family": text_value(row.get("family_prediction")),
                "Reported genus": reported,
                "Closest trained genus": closest,
                "Family score": pct(row.get("family_confidence")),
                "Top match score": pct(row.get("confidence")),
                "Separation margin": pct(row.get("margin")),
                "Status": text_value(row.get("status", "predicted")),
                "Length": row.get("length", "n/a"),
            }
        )
    return rows


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Geneclass Judge App", page_icon="DNA", layout="wide")
    st.title("Geneclass Judge App")
    st.caption("Unknown sequence routing for Bacteria, Fungi, and Viruses using the best app models.")
    st.caption("Top match score is a relative model match score, not the probability that the prediction is correct.")

    model_summary = app_model_summary()
    if any(row["Model"] == "missing" for row in model_summary):
        st.error("One or more app models are missing. Run train_app_models.py first.")
        return

    st.sidebar.header("Prediction")
    route_choice = st.sidebar.selectbox(
        "Kingdom routing",
        list(KINGDOM_OPTIONS.keys()),
        index=0,
    )
    reject_threshold = 0.35
    with st.sidebar.expander("Quality threshold"):
        reject_threshold = st.slider(
            "Minimum reporting score",
            min_value=0.0,
            max_value=0.95,
            value=0.35,
            step=0.01,
        )
        st.caption("Low-score matches are reported as possible or uncertain instead of confirmed genus calls.")

    st.sidebar.subheader("Best models")
    st.sidebar.table(model_summary)

    tab_upload, tab_paste, tab_random = st.tabs(["Upload FASTA", "Paste Sequence", "Random Test"])

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
                st.warning("No FASTA file found under Database/, DNA/, or RNA/")
            else:
                text = chosen.read_text(encoding="utf-8", errors="ignore")
                records = parse_fasta_text(text)
                gt = fasta_ground_truth(chosen)
                source_note = f"Random file: {chosen} | expected path label: {gt}"

    if st.button("Predict", type="primary", use_container_width=True):
        if not records:
            st.warning("No valid sequence found.")
            return

        routed = run_central_reader(
            records,
            model_source="best",
            reject_threshold=reject_threshold,
            force_target=KINGDOM_OPTIONS[route_choice],
        )
        preds = routed["predictions"]

        if source_note:
            st.caption(source_note)

        st.subheader("Conclusion")
        conclusion_rows = build_conclusion_rows(routed)
        if conclusion_rows:
            st.dataframe(conclusion_rows, hide_index=True, use_container_width=True)
        else:
            st.warning("No prediction rows were generated.")
            return

        suggestions = [{"#": idx, "Recommendation": suggestion_for(row)} for idx, row in enumerate(preds, start=1)]
        st.subheader("Suggestion")
        st.dataframe(suggestions, hide_index=True, use_container_width=True)

        with st.expander("Model routing details"):
            st.write(f"Routing mode: `{route_choice}`")
            st.write(f"Routing method: `{routed.get('routing_method', 'unknown')}`")
            st.write(f"Selected model: `{routed['selected_model_path']}`")
            if routed.get("selected_family_model_path"):
                st.write(f"Virus family model: `{routed['selected_family_model_path']}`")
            st.write(f"Trained genera in selected model: `{routed.get('selected_model_class_count', 0)}`")
            if routed.get("selected_family_model_class_count"):
                st.write(f"Virus families in family model: `{routed.get('selected_family_model_class_count', 0)}`")
            if routed.get("routing_scores"):
                routing_view = []
                for row in sorted(
                    routed["routing_scores"],
                    key=lambda x: float(x.get("score", 0.0) or 0.0),
                    reverse=True,
                ):
                    routing_view.append(
                        {
                            "Kingdom": row.get("kingdom"),
                            "Combined score": pct(row.get("score", 0.0)),
                            "Router score": pct(row.get("router_score")) if "router_score" in row else "n/a",
                            "Model evidence": pct(row.get("evidence_score")) if "evidence_score" in row else "n/a",
                            "Header hint": pct(row.get("header_hint_score")) if "header_hint_score" in row else "n/a",
                        }
                    )
                st.dataframe(routing_view, hide_index=True, use_container_width=True)
            if routed.get("fit_scores"):
                fit_view = []
                for target, item in sorted(
                    routed["fit_scores"].items(),
                    key=lambda kv: float(((kv[1] or {}).get("score", 0.0)) or 0.0),
                    reverse=True,
                ):
                    fit_view.append(
                        {
                            "Model group": TARGET_LABELS.get(target, target),
                            "Fit score": pct(item.get("score", 0.0)),
                            "Records used": int(item.get("n_records", 0)),
                        }
                    )
                st.dataframe(fit_view, hide_index=True, use_container_width=True)

        for idx, row in enumerate(preds, start=1):
            with st.expander(f"Sequence {idx} details"):
                st.write(f"Header: `{row['header']}`")
                st.write(f"Length: `{row['length']}`")
                top_items = sorted(
                    row.get("top") or [],
                    key=lambda x: float(x.get("score", 0.0) or 0.0),
                    reverse=True,
                )
                if top_items:
                    top_view = [{"Genus": item["label"], "Top match score": pct(item["score"])} for item in top_items]
                    st.write("Top trained-genus matches")
                    st.dataframe(top_view, hide_index=True, use_container_width=True)

                family_items = sorted(
                    row.get("family_top") or [],
                    key=lambda x: float(x.get("score", 0.0) or 0.0),
                    reverse=True,
                )
                if family_items:
                    family_view = [
                        {"Family": item["label"], "Family match score": pct(item["score"])}
                        for item in family_items
                    ]
                    st.write("Top virus-family matches")
                    st.dataframe(family_view, hide_index=True, use_container_width=True)

                clinical = infer_closest_disease(row.get("raw_prediction") or row.get("prediction"), top_items)
                if clinical.get("matched") == "1":
                    st.write("Reference mapping")
                    st.dataframe(
                        [
                            {
                                "Pathogen panel": clinical["pathogen_panel"],
                                "Disease group": clinical["disease_group"],
                                "Hint": clinical["syndrome_hint"],
                            }
                        ],
                        hide_index=True,
                        use_container_width=True,
                    )

                ranked_groups = rank_disease_groups(top_items, top_n=3)
                if ranked_groups:
                    group_view = [
                        {
                            "Pathogen panel": x["pathogen_panel"],
                            "Disease group": x["disease_group"],
                            "Score": pct(x["score"]),
                            "Supporting genera": x["supporting_genera"],
                        }
                        for x in sorted(
                            ranked_groups,
                            key=lambda x: float(x.get("score", 0.0) or 0.0),
                            reverse=True,
                        )
                    ]
                    st.write("Closest reference groups")
                    st.dataframe(group_view, hide_index=True, use_container_width=True)


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
