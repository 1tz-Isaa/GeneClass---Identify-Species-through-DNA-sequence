from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import importlib.util
import io
import json
import re
import ssl
import time
import urllib.parse
import urllib.request

import numpy as np
from Bio import SeqIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from central_reader import run_central_reader
from run_virus_svc_focused_search import feature_text, load_rows


BASE = Path(__file__).resolve().parent
OUT = BASE / "runs" / "external_unknown_test"
OUT.mkdir(parents=True, exist_ok=True)
HELPER_PATH = BASE / "import file.py"
ENA_PORTAL_API = "https://www.ebi.ac.uk/ena/portal/api/search"
ENA_FASTA_API = "https://www.ebi.ac.uk/ena/browser/api/fasta"

TARGET_PER_GENUS = 3
MAX_SEARCH_ROWS = 250
MAX_ATTEMPTS_PER_SPECIES = 80

BEST_SVC_CONFIG = {
    "feature_mode": "length_segment_prefix",
    "fragment_len": 0,
    "kmer_min": 5,
    "kmer_max": 6,
    "C": 1.0,
    "min_df": 2,
    "max_features": 100000,
    "class_weight": "balanced",
}


def load_helpers():
    spec = importlib.util.spec_from_file_location("pathogen_import_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load_helpers()


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


SSL_CONTEXT = ssl_context()


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Geneclass25-external-test/1.0"})
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as handle:
                return handle.read().decode("utf-8", errors="ignore")
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(attempt * 1.5)
    raise RuntimeError(f"Request failed after retries: {url}") from last_error


def local_accessions_and_hashes() -> tuple[set[str], set[str]]:
    accessions: set[str] = set()
    hashes: set[str] = set()
    for fp in (BASE / "Database" / "rna_genus").glob("*/*/*.fasta"):
        rec = helpers.parse_local_fasta(fp)
        if rec is None:
            continue
        accessions.add(helpers.normalize_token(helpers.extract_accession(rec.description)))
        seq_key = helpers.sequence_hash_key(rec.seq)
        if seq_key:
            hashes.add(seq_key)
    return accessions, hashes


def ena_search(query: str) -> list[dict[str, str]]:
    params = {
        "result": "sequence",
        "query": query,
        "fields": "accession,scientific_name,description,base_count,tax_id,mol_type",
        "format": "tsv",
        "limit": str(MAX_SEARCH_ROWS),
    }
    try:
        raw = fetch_text(ENA_PORTAL_API + "?" + urllib.parse.urlencode(params))
    except Exception as exc:
        print(f"[external] skip ENA query after error: {query} ({exc})")
        return []
    return [row for row in csv.DictReader(io.StringIO(raw), delimiter="\t") if row.get("accession")]


def ena_queries(genus: str, species: str) -> list[str]:
    names = helpers.expand_organism_names(helpers.build_organism_name(genus, species, False), True)
    out: list[str] = []
    seen = set()
    for name in names:
        escaped = name.replace('"', '\\"')
        for clause in [f'scientific_name="{escaped}"', f'description="{escaped}"']:
            q = f'{clause} AND mol_type="genomic RNA" AND base_count >= 500 AND base_count <= 40000'
            if q in seen:
                continue
            seen.add(q)
            out.append(q)
    return out


def fetch_fasta(accession: str):
    raw = fetch_text(f"{ENA_FASTA_API}/{urllib.parse.quote(accession)}?download=false")
    return next(SeqIO.parse(io.StringIO(raw), "fasta"), None)


def train_svc_model(rows: list[dict]):
    model = make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(BEST_SVC_CONFIG["kmer_min"], BEST_SVC_CONFIG["kmer_max"]),
            lowercase=False,
            min_df=BEST_SVC_CONFIG["min_df"],
            max_features=BEST_SVC_CONFIG["max_features"],
            sublinear_tf=True,
            dtype=np.float32,
        ),
        LinearSVC(
            C=BEST_SVC_CONFIG["C"],
            class_weight=BEST_SVC_CONFIG["class_weight"],
            max_iter=10000,
            random_state=42,
        ),
    )
    model.fit([feature_text(r, BEST_SVC_CONFIG) for r in rows], [r["genus"] for r in rows])
    return model


def record_to_row(genus: str, species: str, rec) -> dict:
    seq = helpers.clean_sequence_text(rec.seq)
    return {
        "genus": genus,
        "species_group": f"{genus}/{species}",
        "seq": seq,
        "segment": segment_tag(rec.description),
        "length_tag": length_tag(seq),
        "header": rec.description,
    }


def segment_tag(header: str) -> str:
    h = header.lower()
    m = re.search(r"\bsegment\s*[:=]?\s*([0-9]+|[a-z])\b", h)
    if m:
        return f"segment_{m.group(1).lower()}"
    for token in ["pb2", "pb1", "pa", "ha", "np", "na", "m1", "m2", "ns1", "ns2"]:
        if re.search(rf"\b{re.escape(token)}\b", h):
            return f"gene_{token}"
    for token in ["nucleoprotein", "polymerase", "glycoprotein", "matrix", "hemagglutinin", "neuraminidase"]:
        if token in h:
            return "gene_" + token.replace(" ", "_")
    return "segment_unknown"


def length_tag(seq: str) -> str:
    n = len(seq)
    if n < 1000:
        return "len_short"
    if n < 3000:
        return "len_segment"
    if n < 10000:
        return "len_mid"
    return "len_genome"


def collect_external_records() -> list[dict]:
    local_accessions, local_hashes = local_accessions_and_hashes()
    selected: list[dict] = []
    selected_hashes: set[str] = set()
    per_genus: Counter[str] = Counter()

    for genus, species_list in helpers.RNA_GENUS_TO_SPECIES.items():
        for species in species_list:
            if per_genus[genus] >= TARGET_PER_GENUS:
                break
            attempts = 0
            for query in ena_queries(genus, species):
                if per_genus[genus] >= TARGET_PER_GENUS:
                    break
                for meta in ena_search(query):
                    if attempts >= MAX_ATTEMPTS_PER_SPECIES or per_genus[genus] >= TARGET_PER_GENUS:
                        break
                    attempts += 1
                    accession = meta.get("accession", "").strip()
                    if not accession:
                        continue
                    if helpers.normalize_token(accession) in local_accessions:
                        continue
                    header_hint = f"{accession} {meta.get('scientific_name', '')} {meta.get('description', '')}"
                    if helpers.is_bad_import_header(header_hint):
                        continue
                    if helpers.violates_species_header_exclusion(header_hint, species):
                        continue
                    try:
                        rec = fetch_fasta(accession)
                    except Exception:
                        continue
                    if rec is None:
                        continue
                    seq = helpers.clean_sequence_text(rec.seq)
                    if not helpers.sequence_quality_ok(seq, min_len=500, max_len=40000):
                        continue
                    seq_key = helpers.sequence_hash_key(seq)
                    if seq_key in local_hashes or seq_key in selected_hashes:
                        continue
                    if helpers.is_bad_import_header(rec.description):
                        continue
                    if helpers.violates_species_header_exclusion(rec.description, species):
                        continue

                    selected_hashes.add(seq_key)
                    per_genus[genus] += 1
                    selected.append(
                        {
                            "expected_genus": genus,
                            "species": species,
                            "accession": accession,
                            "header": rec.description,
                            "sequence": seq,
                            "row": record_to_row(genus, species, rec),
                        }
                    )
    return selected


def metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    _, _, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f_macro),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "n": len(y_true),
    }


def main() -> None:
    train_rows = load_rows()
    print(f"[external] training SVC on local rows={len(train_rows)}")
    model = train_svc_model(train_rows)

    external = collect_external_records()
    print(f"[external] collected external records={len(external)}")
    if not external:
        raise RuntimeError("No external records collected")

    y_true = []
    y_svc = []
    y_app_raw = []
    y_app_reported = []
    rows = []
    for item in external:
        expected = item["expected_genus"]
        row = item["row"]
        svc_pred = str(model.predict([feature_text(row, BEST_SVC_CONFIG)])[0])
        app_result = run_central_reader([(item["header"], item["sequence"])], force_target="rna")
        app_pred = app_result["predictions"][0]
        app_raw = str(app_pred.get("raw_prediction") or app_pred.get("prediction"))
        app_reported = str(app_pred.get("prediction"))

        y_true.append(expected)
        y_svc.append(svc_pred)
        y_app_raw.append(app_raw)
        y_app_reported.append(app_reported)
        rows.append(
            {
                "expected_genus": expected,
                "species": item["species"],
                "accession": item["accession"],
                "svc_pred": svc_pred,
                "svc_correct": svc_pred == expected,
                "app_raw_pred": app_raw,
                "app_reported_pred": app_reported,
                "app_raw_correct": app_raw == expected,
                "app_reported_correct": app_reported == expected,
                "app_family": app_pred.get("family_prediction"),
                "app_report_level": app_pred.get("report_level"),
                "app_status": app_pred.get("status"),
                "app_confidence": app_pred.get("confidence"),
                "header": item["header"],
            }
        )

    out_csv = OUT / "external_virus_test_predictions.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "selection": "ENA records not present in local dataset by accession or exact sequence",
        "target_per_genus": TARGET_PER_GENUS,
        "svc_config": BEST_SVC_CONFIG,
        "svc_metrics": metrics(y_true, y_svc),
        "app_raw_metrics": metrics(y_true, y_app_raw),
        "app_reported_metrics": metrics(y_true, y_app_reported),
        "per_genus_counts": dict(Counter(y_true)),
    }
    out_json = OUT / "external_virus_test_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[external] wrote {out_csv}")
    print(f"[external] wrote {out_json}")


if __name__ == "__main__":
    main()
