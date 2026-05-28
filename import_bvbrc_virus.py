"""Refill virus FASTA data from BV-BRC genome_sequence API.

BV-BRC overlaps with NCBI/ENA, so this script treats it as a supplemental source:
it only appends records that pass the same quality/header filters and are not
exact-sequence duplicates of anything already in the local virus collection.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

csv.field_size_limit(10_000_000)


BASE = Path(__file__).resolve().parent
HELPER_PATH = BASE / "import file.py"
RUN_DIR = BASE / "runs" / "external_sources"
RUN_DIR.mkdir(parents=True, exist_ok=True)

BV_API = "https://www.bv-brc.org/api/genome_sequence/"
TARGET_ROOT = BASE / "Database" / "rna_genus"
TARGET_COUNT = max(1, int(os.getenv("BVBRC_RNA_SAMPLES_PER_SPECIES", "20")))
SEARCH_LIMIT = max(1, int(os.getenv("BVBRC_SEARCH_LIMIT", "800")))
REQUEST_SLEEP_SEC = float(os.getenv("BVBRC_REQUEST_SLEEP_SEC", "0.15"))
RETRIES = max(1, int(os.getenv("BVBRC_RETRIES", "3")))
MIN_LEN = max(1, int(os.getenv("BVBRC_MIN_LEN", "500")))
MAX_LEN = max(MIN_LEN, int(os.getenv("BVBRC_MAX_LEN", "40000")))
SKIP_COMPLETED = os.getenv("BVBRC_SKIP_COMPLETED", "1") == "1"
REFILL_ONLY_MISSING = os.getenv("BVBRC_REFILL_ONLY_MISSING", "1") == "1"


def load_helpers():
    spec = importlib.util.spec_from_file_location("pathogen_import_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load_helpers()

RNA_PROFILE = dict(helpers.PROFILE_RNA)
RNA_PROFILE["min_len"] = MIN_LEN
RNA_PROFILE["max_len"] = MAX_LEN
RNA_PROFILE["require_complete_genome"] = False

BVBRC_QUERY_ALIASES = {
    "influenza a virus h3n2": ["Influenza A virus H3N2", "Influenza A virus"],
    "avian metapneumovirus": [
        "Avian metapneumovirus",
        "Avian pneumovirus",
        "Turkey rhinotracheitis virus",
    ],
    "bovine orthopneumovirus": ["Bovine orthopneumovirus", "Bovine respiratory syncytial virus"],
    "murine orthopneumovirus": ["Murine orthopneumovirus", "Pneumonia virus of mice"],
    "bovine respirovirus 3": ["Bovine respirovirus 3", "Bovine parainfluenza virus 3"],
    "porcine respirovirus 1": ["Porcine respirovirus 1", "Porcine parainfluenza virus 1"],
    "murine respirovirus": ["Murine respirovirus", "Sendai virus"],
}


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


SSL_CONTEXT = ssl_context()


def fetch_text(url: str) -> str:
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Geneclass25-BVBRC-import/1.0"})
            with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as handle:
                return handle.read().decode("utf-8", errors="ignore")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < RETRIES:
                time.sleep(attempt * 1.5)
    raise RuntimeError(f"BV-BRC request failed after {RETRIES} tries: {url}") from last_error


def bvbrc_query(genome_name: str) -> list[dict[str, str]]:
    safe_name = urllib.parse.quote(genome_name, safe="")
    rql = (
        f"eq(genome_name,{safe_name})"
        "&select(genome_id,genome_name,accession,description,sequence)"
        f"&limit({SEARCH_LIMIT})"
    )
    url = BV_API + "?" + rql
    raw = fetch_text(url)
    if REQUEST_SLEEP_SEC > 0:
        time.sleep(REQUEST_SLEEP_SEC)
    reader = csv.DictReader(io.StringIO(raw))
    return [row for row in reader if row.get("sequence")]


def query_names_for_species(genus: str, species: str) -> list[str]:
    names = helpers.expand_organism_names(helpers.build_organism_name(genus, species, False), True)
    names.extend(BVBRC_QUERY_ALIASES.get(helpers.normalize_key(species), []))
    out: list[str] = []
    seen = set()
    for name in names:
        key = helpers.normalize_key(name)
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def rows_to_records(rows: Iterable[dict[str, str]], species: str) -> list[SeqRecord]:
    records: list[SeqRecord] = []
    for row in rows:
        accession = (row.get("accession") or row.get("genome_id") or "unknown").strip()
        genome_id = (row.get("genome_id") or "").strip()
        genome_name = (row.get("genome_name") or "").strip()
        description = (row.get("description") or "").strip()
        header = f"BVBRC|{accession}|{genome_id} {genome_name} {description}".strip()

        if helpers.is_bad_import_header(header):
            continue
        if helpers.violates_species_header_exclusion(header, species):
            continue

        seq = helpers.clean_sequence_text(row.get("sequence", ""))
        if not seq:
            continue
        rec = SeqRecord(Seq(seq), id=f"BVBRC|{accession}|{genome_id}", description=header)
        records.append(rec)
    return records


def search_records(genus: str, species: str) -> tuple[list[SeqRecord], int]:
    seen_headers: set[str] = set()
    all_records: list[SeqRecord] = []
    total_rows = 0

    for idx, name in enumerate(query_names_for_species(genus, species), start=1):
        rows = bvbrc_query(name)
        total_rows += len(rows)
        print(f"[BV-BRC] {genus}/{species} query{idx} {name!r}: {len(rows)} rows")
        for rec in rows_to_records(rows, species):
            key = rec.description
            if key in seen_headers:
                continue
            seen_headers.add(key)
            all_records.append(rec)
    return all_records, total_rows


def import_species(genus: str, species: str, collection_sequence_hashes: set[str]) -> dict[str, object]:
    existing = helpers.existing_species_state(TARGET_ROOT, genus, species)
    existing_count = int(existing["count"])

    if SKIP_COMPLETED and existing_count >= TARGET_COUNT:
        print(f"[BV-BRC] SKIP {genus}/{species}: already has {existing_count}")
        return {
            "genus": genus,
            "species": species,
            "existing": existing_count,
            "added": 0,
            "final": existing_count,
            "rows_found": 0,
            "status": "skipped_existing",
        }

    append_mode = REFILL_ONLY_MISSING
    needed = max(0, TARGET_COUNT - existing_count) if append_mode else TARGET_COUNT
    if needed == 0:
        return {
            "genus": genus,
            "species": species,
            "existing": existing_count,
            "added": 0,
            "final": existing_count,
            "rows_found": 0,
            "status": "no_refill_needed",
        }

    print(f"\n[BV-BRC] Searching {genus}/{species} existing={existing_count}, need={needed}")
    records, rows_found = search_records(genus, species)
    print(f"[BV-BRC] {genus}/{species}: {len(records)} candidate records from {rows_found} rows")
    if not records:
        return {
            "genus": genus,
            "species": species,
            "existing": existing_count,
            "added": 0,
            "final": existing_count,
            "rows_found": rows_found,
            "status": "no_records",
        }

    selected = helpers.choose_unique_strain_samples(
        records=records,
        genus=genus,
        species=species,
        profile=RNA_PROFILE,
        target_count=needed,
        pre_used_accessions=set(existing["accessions"]) if append_mode else None,
        pre_used_strains=set(existing["strains"]) if append_mode else None,
        pre_used_sequence_hashes=collection_sequence_hashes,
    )

    helpers.write_species_samples(
        root=TARGET_ROOT,
        genus=genus,
        species=species,
        selected=selected,
        append=append_mode,
        start_index=int(existing["next_index"]) if append_mode else 1,
    )

    for rec, _, _ in selected:
        seq_key = helpers.sequence_hash_key(rec.seq)
        if seq_key:
            collection_sequence_hashes.add(seq_key)

    final_count = existing_count + len(selected) if append_mode else len(selected)
    status = "ok" if final_count >= TARGET_COUNT else "not_enough_unique"
    if not selected and records:
        status = "query_hit_but_filtered_or_duplicate"

    print(f"[BV-BRC] SAVE {genus}/{species}: +{len(selected)} -> {final_count}/{TARGET_COUNT} [{status}]")
    return {
        "genus": genus,
        "species": species,
        "existing": existing_count,
        "added": len(selected),
        "final": final_count,
        "rows_found": rows_found,
        "status": status,
    }


def main() -> None:
    print(
        "[BV-BRC CONFIG] "
        f"target_root={TARGET_ROOT} target_count={TARGET_COUNT} "
        f"search_limit={SEARCH_LIMIT} min_len={MIN_LEN} max_len={MAX_LEN}"
    )
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    collection_sequence_hashes = helpers.collect_existing_sequence_hashes(TARGET_ROOT)
    rows: list[dict[str, object]] = []

    for genus, species_list in helpers.RNA_GENUS_TO_SPECIES.items():
        for species in species_list:
            rows.append(import_species(genus, species, collection_sequence_hashes))

    summary = {
        "source": "BV-BRC",
        "target_root": str(TARGET_ROOT),
        "target_count": TARGET_COUNT,
        "total_added": sum(int(row["added"]) for row in rows),
        "rows": rows,
    }
    summary_path = RUN_DIR / "bvbrc_virus_import_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tsv_path = RUN_DIR / "bvbrc_virus_import_summary.tsv"
    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["genus", "species", "existing", "added", "final", "rows_found", "status"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[BV-BRC SUMMARY] added={summary['total_added']} wrote {summary_path}")
    print(f"[BV-BRC SUMMARY] wrote {tsv_path}")


if __name__ == "__main__":
    main()
