import csv
import gzip
import importlib.util
import json
import os
import re
import ssl
import tarfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


UNITE_DOI = "10.15156/BIO/3301230"
UNITE_DOI_API = "https://api.plutof.ut.ee/v1/public/dois/"
DEFAULT_ARCHIVE_PATH = Path("runs/external_sources/sh_general_release_s_19.02.2025.tgz")
DEFAULT_REPORT_PATH = Path("runs/external_imports/unite_fungi_import_report.tsv")
FUNGI_ROOT = Path(os.getenv("UNITE_FUNGI_ROOT", "Database/fungi_genus"))
TARGET_SAMPLES_PER_SPECIES = max(
    1,
    int(
        os.getenv(
            "UNITE_SAMPLES_PER_SPECIES",
            os.getenv("DNA_SAMPLES_PER_SPECIES", os.getenv("SAMPLES_PER_SPECIES", "8")),
        )
    ),
)
MAX_AMBIGUOUS_FRACTION = float(os.getenv("UNITE_MAX_AMBIGUOUS_FRACTION", "0.20"))
MIN_UNIQUE_BASES = max(1, int(os.getenv("UNITE_MIN_UNIQUE_BASES", "4")))
MIN_SEQUENCE_LENGTH = max(1, int(os.getenv("UNITE_MIN_SEQUENCE_LENGTH", "250")))
MAX_SEQUENCE_LENGTH = max(MIN_SEQUENCE_LENGTH, int(os.getenv("UNITE_MAX_SEQUENCE_LENGTH", "3500")))
UNITE_USER_AGENT = os.getenv("UNITE_USER_AGENT", "Geneclass25 UNITE importer")


BAD_HEADER_TERMS = (
    "patent",
    "synthetic construct",
    "cloning vector",
    "expression vector",
    "plasmid",
    "artificial sequence",
    "environmental sample",
    "uncultured",
    "unverified",
)


@dataclass
class FastaRecord:
    header: str
    seq: str

    @property
    def description(self) -> str:
        return self.header


@dataclass
class UniteTaxonomy:
    genus: str
    species: str
    accession: str
    sh_code: str


def load_ncbi_importer():
    path = Path("import file.py")
    spec = importlib.util.spec_from_file_location("geneclass_ncbi_importer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ssl_context():
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UNITE_USER_AGENT})
    with urllib.request.urlopen(req, timeout=60, context=ssl_context()) as handle:
        return json.load(handle)


def resolve_unite_archive_url() -> Tuple[str, str]:
    params = urllib.parse.urlencode({"identifier": UNITE_DOI})
    metadata = fetch_json(f"{UNITE_DOI_API}?{params}")
    rows = metadata.get("data", [])
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one UNITE DOI result, got {len(rows)}")

    media = rows[0].get("attributes", {}).get("media", [])
    gzip_media = [m for m in media if "gzip" in str(m.get("content_type", ""))]
    selected = gzip_media[0] if gzip_media else (media[0] if media else None)
    if not selected or not selected.get("url"):
        raise RuntimeError("UNITE DOI metadata did not expose a downloadable media URL")
    return str(selected["url"]), str(selected.get("name") or DEFAULT_ARCHIVE_PATH.name)


def download_archive(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target

    url, name = resolve_unite_archive_url()
    if target == DEFAULT_ARCHIVE_PATH and name:
        target = target.with_name(name)

    print(f"[UNITE] Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": UNITE_USER_AGENT})
    with urllib.request.urlopen(req, timeout=180, context=ssl_context()) as response, target.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    print(f"[UNITE] Saved archive: {target} ({target.stat().st_size} bytes)")
    return target


def clean_sequence_text(seq: object) -> str:
    return re.sub(r"[^ACGTUNacgtun]", "", str(seq)).upper()


def sequence_hash_key(seq: object) -> str:
    return clean_sequence_text(seq).replace("U", "T")


def sequence_quality_ok(seq: object) -> bool:
    clean = clean_sequence_text(seq)
    if len(clean) < MIN_SEQUENCE_LENGTH or len(clean) > MAX_SEQUENCE_LENGTH:
        return False
    canonical = clean.replace("U", "T")
    ambiguous_fraction = canonical.count("N") / len(canonical)
    if ambiguous_fraction > MAX_AMBIGUOUS_FRACTION:
        return False
    unique_bases = {base for base in canonical if base in {"A", "C", "G", "T"}}
    return len(unique_bases) >= MIN_UNIQUE_BASES


def normalize_token(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def safe_path_name(name: str) -> str:
    name = name.strip().replace("/", "-")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_.()\-]", "", name)
    return name or "unknown"


def species_prefix(genus: str, species: str) -> str:
    g = next((c for c in genus if c.isalnum()), "X")
    s = next((c for c in species if c.isalnum()), "X")
    return f"{g}{s}".upper()


def species_output_dir(root: Path, genus: str, species: str) -> Path:
    return root / safe_path_name(genus) / safe_path_name(species)


def parse_fasta_stream(lines: Iterable[bytes]) -> Iterator[FastaRecord]:
    header: Optional[str] = None
    seq_parts: List[str] = []

    for raw in lines:
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield FastaRecord(header=header, seq="".join(seq_parts))
            header = line[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(line)

    if header is not None:
        yield FastaRecord(header=header, seq="".join(seq_parts))


def iter_fasta_records(source_path: Path) -> Iterator[FastaRecord]:
    suffixes = "".join(source_path.suffixes).lower()
    if suffixes.endswith(".tgz") or suffixes.endswith(".tar.gz"):
        with tarfile.open(source_path, "r:gz") as archive:
            members = [m for m in archive.getmembers() if m.isfile() and m.name.lower().endswith((".fasta", ".fa"))]
            preferred = [m for m in members if "_dev" not in m.name.lower()]
            member = preferred[0] if preferred else members[0]
            print(f"[UNITE] Reading FASTA from archive member: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"Could not read archive member {member.name}")
            yield from parse_fasta_stream(handle)
        return

    if suffixes.endswith(".gz"):
        with gzip.open(source_path, "rb") as handle:
            yield from parse_fasta_stream(handle)
        return

    with source_path.open("rb") as handle:
        yield from parse_fasta_stream(handle)


def parse_unite_taxonomy(header: str) -> Optional[UniteTaxonomy]:
    parts = header.split("|")
    accession = parts[1].strip() if len(parts) > 1 else parts[0].strip()
    sh_code = parts[2].strip() if len(parts) > 2 else ""
    taxonomy = parts[-1] if parts else header

    genus = ""
    species = ""
    for field in taxonomy.split(";"):
        field = field.strip()
        if field.startswith("g__"):
            genus = field[3:].replace("_", " ").strip()
        elif field.startswith("s__"):
            species = field[3:].replace("_", " ").strip()

    if not genus or not species:
        first = parts[0].replace("_", " ").strip() if parts else ""
        words = first.split()
        if len(words) >= 2:
            genus = words[0]
            species = " ".join(words[:2])

    if not genus or not species:
        return None
    if genus.endswith(" gen Incertae sedis") or species.endswith(" sp"):
        return None

    return UniteTaxonomy(
        genus=genus,
        species=species,
        accession=accession,
        sh_code=sh_code,
    )


def build_target_lookup(genus_to_species: Dict[str, List[str]]) -> Dict[Tuple[str, str], Tuple[str, str]]:
    lookup: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for genus, species_list in genus_to_species.items():
        for species in species_list:
            if species.strip().lower().startswith("sp."):
                continue
            target_species_phrase = f"{genus} {species}".lower()
            lookup[(genus.lower(), target_species_phrase)] = (genus, species)
    return lookup


def parse_local_fasta(path: Path) -> Optional[FastaRecord]:
    try:
        with path.open("rb") as handle:
            return next(parse_fasta_stream(handle), None)
    except OSError:
        return None


def collect_existing_state(root: Path, genus_to_species: Dict[str, List[str]]) -> Dict[Tuple[str, str], dict]:
    state: Dict[Tuple[str, str], dict] = {}
    collection_hashes = set()
    for genus, species_list in genus_to_species.items():
        for species in species_list:
            out_dir = species_output_dir(root, genus, species)
            prefix = species_prefix(genus, species)
            files = sorted(out_dir.glob(f"{prefix}_sample*.fasta")) if out_dir.exists() else []
            accessions = set()
            seq_hashes = set()
            max_idx = 0
            valid_count = 0

            for path in files:
                match = re.search(r"_sample(\d+)\.fasta$", path.name, flags=re.IGNORECASE)
                if match:
                    max_idx = max(max_idx, int(match.group(1)))
                record = parse_local_fasta(path)
                if record is None:
                    continue
                valid_count += 1
                header_accession = record.header.split()[0].strip("|>")
                accessions.add(normalize_token(header_accession))
                seq_hash = sequence_hash_key(record.seq)
                if seq_hash:
                    seq_hashes.add(seq_hash)
                    collection_hashes.add(seq_hash)

            state[(genus, species)] = {
                "count": valid_count,
                "accessions": accessions,
                "seq_hashes": seq_hashes,
                "next_index": max_idx + 1 if max_idx else 1,
            }
    state[("__collection__", "__hashes__")] = {"seq_hashes": collection_hashes}
    return state


def format_unite_header(record: FastaRecord, taxonomy: UniteTaxonomy) -> str:
    return (
        f"UNITE|{taxonomy.accession}|{taxonomy.sh_code}|source_doi={UNITE_DOI}|"
        f"{record.header}"
    )


def write_record(root: Path, genus: str, species: str, index: int, record: FastaRecord, taxonomy: UniteTaxonomy) -> Path:
    out_dir = species_output_dir(root, genus, species)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = species_prefix(genus, species)
    path = out_dir / f"{prefix}_sample{index}.fasta"
    while path.exists():
        index += 1
        path = out_dir / f"{prefix}_sample{index}.fasta"

    seq = clean_sequence_text(record.seq)
    wrapped = "\n".join(seq[i : i + 80] for i in range(0, len(seq), 80))
    path.write_text(f">{format_unite_header(record, taxonomy)}\n{wrapped}\n", encoding="utf-8")
    return path


def import_unite_records(source_path: Path, root: Path, genus_to_species: Dict[str, List[str]]) -> List[dict]:
    target_lookup = build_target_lookup(genus_to_species)
    state = collect_existing_state(root, genus_to_species)
    collection_hashes = set(state[("__collection__", "__hashes__")]["seq_hashes"])

    rows: Dict[Tuple[str, str], dict] = {}
    for genus, species_list in genus_to_species.items():
        for species in species_list:
            existing = int(state[(genus, species)]["count"])
            rows[(genus, species)] = {
                "source": "UNITE",
                "doi": UNITE_DOI,
                "genus": genus,
                "species": species,
                "existing_before": existing,
                "target": TARGET_SAMPLES_PER_SPECIES,
                "matched_unite_records": 0,
                "added": 0,
                "skipped_completed": int(existing >= TARGET_SAMPLES_PER_SPECIES),
                "skipped_bad_header": 0,
                "skipped_quality": 0,
                "skipped_duplicate_accession": 0,
                "skipped_duplicate_sequence": 0,
                "final_count": existing,
            }

    for record in iter_fasta_records(source_path):
        taxonomy = parse_unite_taxonomy(record.header)
        if taxonomy is None:
            continue

        key = (taxonomy.genus.lower(), taxonomy.species.lower())
        target = target_lookup.get(key)
        if target is None:
            continue

        genus, species = target
        row = rows[(genus, species)]
        row["matched_unite_records"] += 1

        existing = state[(genus, species)]
        if int(existing["count"]) >= TARGET_SAMPLES_PER_SPECIES:
            row["skipped_completed"] = 1
            continue

        header_lower = record.header.lower()
        if any(term in header_lower for term in BAD_HEADER_TERMS):
            row["skipped_bad_header"] += 1
            continue

        if not sequence_quality_ok(record.seq):
            row["skipped_quality"] += 1
            continue

        accession_key = normalize_token(taxonomy.accession)
        if accession_key in existing["accessions"]:
            row["skipped_duplicate_accession"] += 1
            continue

        seq_key = sequence_hash_key(record.seq)
        if seq_key in collection_hashes:
            row["skipped_duplicate_sequence"] += 1
            continue

        path = write_record(
            root=root,
            genus=genus,
            species=species,
            index=int(existing["next_index"]),
            record=record,
            taxonomy=taxonomy,
        )
        existing["count"] = int(existing["count"]) + 1
        existing["next_index"] = int(existing["next_index"]) + 1
        existing["accessions"].add(accession_key)
        existing["seq_hashes"].add(seq_key)
        collection_hashes.add(seq_key)

        row["added"] += 1
        row["final_count"] = int(existing["count"])
        print(f"[UNITE] added {genus}/{species}: {path}")

    for (genus, species), row in rows.items():
        row["final_count"] = int(state[(genus, species)]["count"])
    return [rows[key] for key in sorted(rows)]


def write_report(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "doi",
        "genus",
        "species",
        "existing_before",
        "target",
        "matched_unite_records",
        "added",
        "skipped_completed",
        "skipped_bad_header",
        "skipped_quality",
        "skipped_duplicate_accession",
        "skipped_duplicate_sequence",
        "final_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[UNITE] Report saved: {path}")


def main() -> None:
    importer = load_ncbi_importer()
    genus_to_species = importer.FUNGI_GENUS_TO_SPECIES

    source_override = os.getenv("UNITE_FASTA_PATH")
    if source_override:
        source_path = Path(source_override)
    else:
        archive_path = Path(os.getenv("UNITE_ARCHIVE_PATH", str(DEFAULT_ARCHIVE_PATH)))
        source_path = download_archive(archive_path)

    print(
        "[UNITE] config "
        f"root={FUNGI_ROOT} target_samples={TARGET_SAMPLES_PER_SPECIES} "
        f"max_ambiguous_fraction={MAX_AMBIGUOUS_FRACTION} min_unique_bases={MIN_UNIQUE_BASES}"
    )
    rows = import_unite_records(source_path, FUNGI_ROOT, genus_to_species)
    report_path = Path(os.getenv("UNITE_REPORT_PATH", str(DEFAULT_REPORT_PATH)))
    write_report(rows, report_path)

    added = sum(int(r["added"]) for r in rows)
    still_short = [
        r for r in rows if int(r["final_count"]) < int(r["target"]) and not str(r["species"]).startswith("sp.")
    ]
    print(f"[UNITE] Total added: {added}")
    print(f"[UNITE] Species still below target: {len(still_short)}")
    for row in still_short:
        print(
            f"- {row['genus']}/{row['species']}: "
            f"{row['final_count']}/{row['target']} "
            f"(matched_unite_records={row['matched_unite_records']}, added={row['added']})"
        )


if __name__ == "__main__":
    main()
