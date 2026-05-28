"""Refill virus FASTA data from ENA/EBI with strict deduplication.

This importer is intentionally separate from ``import file.py`` because ENA uses
different APIs from NCBI Entrez. It reuses the same local target list, cleaning,
quality, and duplicate filters so externally sourced sequences do not weaken the
dataset.
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

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


BASE = Path(__file__).resolve().parent
HELPER_PATH = BASE / "import file.py"
RUN_DIR = BASE / "runs" / "external_sources"
RUN_DIR.mkdir(parents=True, exist_ok=True)

ENA_PORTAL_API = "https://www.ebi.ac.uk/ena/portal/api/search"
ENA_FASTA_API = "https://www.ebi.ac.uk/ena/browser/api/fasta"

TARGET_ROOT = BASE / "Database" / "rna_genus"
TARGET_COUNT = max(1, int(os.getenv("ENA_RNA_SAMPLES_PER_SPECIES", "20")))
SEARCH_LIMIT = max(1, int(os.getenv("ENA_SEARCH_LIMIT", "700")))
FETCH_BATCH_SIZE = max(1, int(os.getenv("ENA_FETCH_BATCH_SIZE", "20")))
REQUEST_SLEEP_SEC = float(os.getenv("ENA_REQUEST_SLEEP_SEC", "0.15"))
RETRIES = max(1, int(os.getenv("ENA_RETRIES", "3")))
MIN_LEN = max(1, int(os.getenv("ENA_MIN_LEN", "500")))
MAX_LEN = max(MIN_LEN, int(os.getenv("ENA_MAX_LEN", "40000")))
SKIP_COMPLETED = os.getenv("ENA_SKIP_COMPLETED", "1") == "1"
REFILL_ONLY_MISSING = os.getenv("ENA_REFILL_ONLY_MISSING", "1") == "1"


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

ENA_QUERY_ALIASES = {
    # ENA usually stores H3N2 as Influenza A virus plus subtype in description.
    "influenza a virus h3n2": [
        'scientific_name="Influenza A virus" AND description="H3N2"',
        'scientific_name="Influenza A virus" AND description="H3"',
    ],
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
            req = urllib.request.Request(url, headers={"User-Agent": "Geneclass25-ENA-import/1.0"})
            with urllib.request.urlopen(req, timeout=45, context=SSL_CONTEXT) as handle:
                return handle.read().decode("utf-8", errors="ignore")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < RETRIES:
                time.sleep(attempt * 1.5)
    raise RuntimeError(f"ENA request failed after {RETRIES} tries: {url}") from last_error


def ena_search(query: str) -> list[dict[str, str]]:
    fields = "accession,scientific_name,description,base_count,tax_id,mol_type"
    params = {
        "result": "sequence",
        "query": query,
        "fields": fields,
        "format": "tsv",
        "limit": str(SEARCH_LIMIT),
    }
    url = ENA_PORTAL_API + "?" + urllib.parse.urlencode(params)
    raw = fetch_text(url)
    if REQUEST_SLEEP_SEC > 0:
        time.sleep(REQUEST_SLEEP_SEC)
    reader = csv.DictReader(io.StringIO(raw), delimiter="\t")
    return [row for row in reader if row.get("accession")]


def build_ena_queries(genus: str, species: str) -> list[str]:
    base_name = helpers.build_organism_name(genus, species, False)
    names = helpers.expand_organism_names(base_name, True)
    species_key = helpers.normalize_key(species)
    queries: list[str] = []

    queries.extend(ENA_QUERY_ALIASES.get(species_key, []))
    for name in names:
        escaped = name.replace('"', '\\"')
        queries.append(f'scientific_name="{escaped}"')
        queries.append(f'description="{escaped}"')

    length_filter = f"base_count >= {MIN_LEN} AND base_count <= {MAX_LEN}"
    mol_filter = 'mol_type="genomic RNA"'

    out: list[str] = []
    seen = set()
    for query in queries:
        full = f"{query} AND {mol_filter} AND {length_filter}"
        if full in seen:
            continue
        seen.add(full)
        out.append(full)
    return out


def search_accessions(genus: str, species: str) -> list[str]:
    seen: set[str] = set()
    accessions: list[str] = []
    for idx, query in enumerate(build_ena_queries(genus, species), start=1):
        rows = ena_search(query)
        print(f"[ENA] {genus}/{species} query{idx}: {len(rows)} rows")
        for row in rows:
            accession = row.get("accession", "").strip()
            description = row.get("description", "")
            scientific_name = row.get("scientific_name", "")
            header = f"{accession} {scientific_name} {description}"
            if helpers.is_bad_import_header(header):
                continue
            if helpers.violates_species_header_exclusion(header, species):
                continue
            if accession and accession not in seen:
                seen.add(accession)
                accessions.append(accession)
        if len(accessions) >= max(TARGET_COUNT * 15, 120):
            break
    return accessions


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_fasta_records(accessions: list[str]) -> list[SeqRecord]:
    records: list[SeqRecord] = []
    for batch in chunks(accessions, FETCH_BATCH_SIZE):
        encoded = ",".join(urllib.parse.quote(acc) for acc in batch)
        url = f"{ENA_FASTA_API}/{encoded}?download=false"
        raw = fetch_text(url)
        parsed = list(SeqIO.parse(io.StringIO(raw), "fasta"))
        records.extend(parsed)
        if REQUEST_SLEEP_SEC > 0:
            time.sleep(REQUEST_SLEEP_SEC)
    return records


def import_species(genus: str, species: str, collection_sequence_hashes: set[str]) -> dict[str, object]:
    existing = helpers.existing_species_state(TARGET_ROOT, genus, species)
    existing_count = int(existing["count"])

    if SKIP_COMPLETED and existing_count >= TARGET_COUNT:
        print(f"[ENA] SKIP {genus}/{species}: already has {existing_count}")
        return {
            "genus": genus,
            "species": species,
            "existing": existing_count,
            "added": 0,
            "final": existing_count,
            "accessions_found": 0,
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
            "accessions_found": 0,
            "status": "no_refill_needed",
        }

    print(f"\n[ENA] Searching {genus}/{species} existing={existing_count}, need={needed}")
    accessions = search_accessions(genus, species)
    print(f"[ENA] {genus}/{species}: {len(accessions)} candidate accessions")
    if not accessions:
        return {
            "genus": genus,
            "species": species,
            "existing": existing_count,
            "added": 0,
            "final": existing_count,
            "accessions_found": 0,
            "status": "no_records",
        }

    selected: list[tuple[SeqRecord, str, str]] = []
    fetch_cursor = 0
    while len(selected) < needed and fetch_cursor < len(accessions):
        batch_accessions = accessions[fetch_cursor : fetch_cursor + FETCH_BATCH_SIZE]
        fetch_cursor += FETCH_BATCH_SIZE
        records = fetch_fasta_records(batch_accessions)
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
    if not selected and accessions:
        status = "query_hit_but_filtered_or_duplicate"

    print(f"[ENA] SAVE {genus}/{species}: +{len(selected)} -> {final_count}/{TARGET_COUNT} [{status}]")
    return {
        "genus": genus,
        "species": species,
        "existing": existing_count,
        "added": len(selected),
        "final": final_count,
        "accessions_found": len(accessions),
        "status": status,
    }


def main() -> None:
    print(
        "[ENA CONFIG] "
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
        "source": "ENA/EBI",
        "target_root": str(TARGET_ROOT),
        "target_count": TARGET_COUNT,
        "total_added": sum(int(row["added"]) for row in rows),
        "rows": rows,
    }
    summary_path = RUN_DIR / "ena_virus_import_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tsv_path = RUN_DIR / "ena_virus_import_summary.tsv"
    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["genus", "species", "existing", "added", "final", "accessions_found", "status"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[ENA SUMMARY] added={summary['total_added']} wrote {summary_path}")
    print(f"[ENA SUMMARY] wrote {tsv_path}")


if __name__ == "__main__":
    main()
