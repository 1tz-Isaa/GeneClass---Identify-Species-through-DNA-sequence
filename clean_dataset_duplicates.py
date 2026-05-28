from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASE = Path(__file__).resolve().parent
DEFAULT_ROOTS = {
    "fungi": BASE / "Database" / "fungi_genus",
    "virus": BASE / "Database" / "rna_genus",
}
REPORT_DIR = BASE / "runs" / "dataset_audit"
BACKUP_ROOT = BASE / "legacy_dataset_backup" / "exact_duplicates_removed"
BAD_HEADER_BACKUP_ROOT = BASE / "legacy_dataset_backup" / "bad_headers_removed"
LOW_QUALITY_BACKUP_ROOT = BASE / "legacy_dataset_backup" / "low_quality_removed"
LABEL_MISMATCH_BACKUP_ROOT = BASE / "legacy_dataset_backup" / "label_mismatch_removed"
MIN_SEQUENCE_LENGTH = 200
MAX_AMBIGUOUS_FRACTION = 0.20
BAD_HEADER_TERMS = (
    "patent",
    " jp ",
    " kr ",
    "synthetic construct",
    "cloning vector",
    "expression vector",
    "plasmid",
    "artificial sequence",
    "environmental sample",
    "uncultured",
    "unverified",
    "partial genome",
    "partial sequence",
    "partial cds",
    "rna construct",
    "composition",
    "oligonucleotide",
    "extracellular vesicle",
    "vaccine against",
    "circular rna",
)


@dataclass(frozen=True)
class FastaFile:
    dataset: str
    path: Path
    genus: str
    species: str
    header: str
    sequence: str


def read_first_fasta(path: Path) -> tuple[str, str] | None:
    header = ""
    seq_parts: list[str] = []
    try:
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header:
                        break
                    header = line[1:].strip()
                    continue
                if header:
                    seq_parts.append(line)
    except OSError:
        return None

    if not header:
        return None
    return header, "".join(seq_parts)


def clean_sequence(seq: str) -> str:
    return re.sub(r"[^ACGTUNacgtun]", "", seq).upper()


def sequence_key(seq: str) -> str:
    # DNA/RNA exact duplicate detection treats T and U as equivalent.
    return clean_sequence(seq).replace("U", "T")


def iter_fasta_files(dataset: str, root: Path) -> Iterable[FastaFile]:
    for path in sorted(root.glob("*/*/*.fasta")):
        parsed = read_first_fasta(path)
        if parsed is None:
            continue
        header, sequence = parsed
        yield FastaFile(
            dataset=dataset,
            path=path,
            genus=path.parts[-3],
            species=path.parts[-2],
            header=header,
            sequence=sequence,
        )


def backup_path_for(record: FastaFile) -> Path:
    relative = record.path.relative_to(BASE)
    return BACKUP_ROOT / record.dataset / relative


def contains_bad_header(header: str) -> list[str]:
    lower = f" {header.lower()} "
    return [term for term in BAD_HEADER_TERMS if term in lower]


def normalized_words(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def virus_label_mismatch_reasons(record: FastaFile) -> list[str]:
    if record.dataset != "virus":
        return []

    h = normalized_words(record.header)
    species = record.species
    reasons: list[str] = []

    # NCBI searches for older SARS-CoV frequently return SARS-CoV-2 because the
    # organism names share a parent taxonomy. Keep these folders label-clean.
    if species == "SARS-CoV" and (
        "sars cov 2" in h or "coronavirus 2" in h or "sars coronavirus 2" in h
    ):
        reasons.append("sars_cov_folder_contains_sars_cov_2")

    # Patents sometimes avoid the literal word "patent" but still carry JP/KR
    # publication identifiers and invention-style descriptions.
    if re.search(r"\b(jp|kr)\s+\d{8,}", h):
        reasons.append("patent_publication_identifier")

    return reasons


def find_duplicates(records: list[FastaFile]) -> list[dict]:
    keepers: dict[str, FastaFile] = {}
    duplicates: list[dict] = []

    for record in records:
        key = sequence_key(record.sequence)
        if not key:
            continue
        keeper = keepers.get(key)
        if keeper is None:
            keepers[key] = record
            continue

        duplicates.append(
            {
                "dataset": record.dataset,
                "duplicate_path": str(record.path.relative_to(BASE)),
                "keeper_path": str(keeper.path.relative_to(BASE)),
                "duplicate_genus": record.genus,
                "duplicate_species": record.species,
                "keeper_genus": keeper.genus,
                "keeper_species": keeper.species,
                "same_label": int(record.genus == keeper.genus),
                "same_species_group": int(
                    record.genus == keeper.genus and record.species == keeper.species
                ),
                "sequence_length": len(key),
                "duplicate_header": record.header,
                "keeper_header": keeper.header,
            }
        )

    return duplicates


def find_bad_headers(records: list[FastaFile]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        matched = contains_bad_header(record.header)
        if not matched:
            continue
        rows.append(
            {
                "dataset": record.dataset,
                "path": str(record.path.relative_to(BASE)),
                "genus": record.genus,
                "species": record.species,
                "matched_terms": ",".join(matched),
                "sequence_length": len(sequence_key(record.sequence)),
                "header": record.header,
            }
        )
    return rows


def find_label_mismatches(records: list[FastaFile]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        reasons = virus_label_mismatch_reasons(record)
        if not reasons:
            continue
        rows.append(
            {
                "dataset": record.dataset,
                "path": str(record.path.relative_to(BASE)),
                "genus": record.genus,
                "species": record.species,
                "reasons": ",".join(reasons),
                "sequence_length": len(sequence_key(record.sequence)),
                "header": record.header,
            }
        )
    return rows


def low_quality_reasons(sequence: str) -> list[str]:
    clean = clean_sequence(sequence)
    if not clean:
        return ["empty_sequence"]
    reasons: list[str] = []
    if len(clean) < MIN_SEQUENCE_LENGTH:
        reasons.append(f"short_lt_{MIN_SEQUENCE_LENGTH}")
    canonical = clean.replace("U", "T")
    if canonical and canonical.count("N") / len(canonical) > MAX_AMBIGUOUS_FRACTION:
        reasons.append(f"ambiguous_gt_{MAX_AMBIGUOUS_FRACTION}")
    unique_bases = {base for base in canonical if base in {"A", "C", "G", "T"}}
    if len(unique_bases) < 4:
        reasons.append("low_base_diversity")
    return reasons


def find_low_quality(records: list[FastaFile]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        reasons = low_quality_reasons(record.sequence)
        if not reasons:
            continue
        rows.append(
            {
                "dataset": record.dataset,
                "path": str(record.path.relative_to(BASE)),
                "genus": record.genus,
                "species": record.species,
                "reasons": ",".join(reasons),
                "sequence_length": len(sequence_key(record.sequence)),
                "header": record.header,
            }
        )
    return rows


def move_duplicates(duplicates: list[dict]) -> None:
    for row in duplicates:
        source = BASE / row["duplicate_path"]
        if not source.exists():
            continue
        target = BACKUP_ROOT / row["dataset"] / row["duplicate_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            suffix = 1
            while True:
                candidate = target.with_name(f"{target.stem}.dup{suffix}{target.suffix}")
                if not candidate.exists():
                    target = candidate
                    break
                suffix += 1
        shutil.move(str(source), str(target))


def move_bad_headers(rows: list[dict]) -> None:
    for row in rows:
        source = BASE / row["path"]
        if not source.exists():
            continue
        target = BAD_HEADER_BACKUP_ROOT / row["dataset"] / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            suffix = 1
            while True:
                candidate = target.with_name(f"{target.stem}.bad{suffix}{target.suffix}")
                if not candidate.exists():
                    target = candidate
                    break
                suffix += 1
        shutil.move(str(source), str(target))


def move_low_quality(rows: list[dict]) -> None:
    for row in rows:
        source = BASE / row["path"]
        if not source.exists():
            continue
        target = LOW_QUALITY_BACKUP_ROOT / row["dataset"] / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            suffix = 1
            while True:
                candidate = target.with_name(f"{target.stem}.lowq{suffix}{target.suffix}")
                if not candidate.exists():
                    target = candidate
                    break
                suffix += 1
        shutil.move(str(source), str(target))


def move_label_mismatches(rows: list[dict]) -> None:
    for row in rows:
        source = BASE / row["path"]
        if not source.exists():
            continue
        target = LABEL_MISMATCH_BACKUP_ROOT / row["dataset"] / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            suffix = 1
            while True:
                candidate = target.with_name(f"{target.stem}.mismatch{suffix}{target.suffix}")
                if not candidate.exists():
                    target = candidate
                    break
                suffix += 1
        shutil.move(str(source), str(target))


def write_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "duplicate_path",
        "keeper_path",
        "duplicate_genus",
        "duplicate_species",
        "keeper_genus",
        "keeper_species",
        "same_label",
        "same_species_group",
        "sequence_length",
        "duplicate_header",
        "keeper_header",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_bad_header_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "path",
        "genus",
        "species",
        "matched_terms",
        "sequence_length",
        "header",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_low_quality_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "path",
        "genus",
        "species",
        "reasons",
        "sequence_length",
        "header",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_label_mismatch_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "path",
        "genus",
        "species",
        "reasons",
        "sequence_length",
        "header",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and optionally remove exact duplicate FASTA records.")
    parser.add_argument("--apply", action="store_true", help="Move duplicate FASTA files out of Database.")
    parser.add_argument(
        "--datasets",
        default="fungi,virus",
        help="Comma-separated dataset keys to audit: fungi,virus.",
    )
    args = parser.parse_args()

    selected = [item.strip() for item in args.datasets.split(",") if item.strip()]
    all_duplicate_rows: list[dict] = []
    all_bad_header_rows: list[dict] = []
    all_low_quality_rows: list[dict] = []
    all_label_mismatch_rows: list[dict] = []

    for dataset in selected:
        root = DEFAULT_ROOTS.get(dataset)
        if root is None:
            raise SystemExit(f"Unknown dataset {dataset!r}; use one of {sorted(DEFAULT_ROOTS)}")
        records = list(iter_fasta_files(dataset, root))
        duplicates = find_duplicates(records)
        bad_headers = find_bad_headers(records)
        low_quality = find_low_quality(records)
        label_mismatches = find_label_mismatches(records)
        all_duplicate_rows.extend(duplicates)
        all_bad_header_rows.extend(bad_headers)
        all_low_quality_rows.extend(low_quality)
        all_label_mismatch_rows.extend(label_mismatches)
        print(
            f"[{dataset}] records={len(records)} exact_duplicates={len(duplicates)} "
            f"cross_genus={sum(1 for row in duplicates if not int(row['same_label']))} "
            f"bad_headers={len(bad_headers)} low_quality={len(low_quality)} "
            f"label_mismatches={len(label_mismatches)}"
        )

    report_path = REPORT_DIR / "exact_duplicate_audit.tsv"
    write_report(report_path, all_duplicate_rows)
    print(f"[audit] wrote {report_path}")
    bad_header_report_path = REPORT_DIR / "bad_header_audit.tsv"
    write_bad_header_report(bad_header_report_path, all_bad_header_rows)
    print(f"[audit] wrote {bad_header_report_path}")
    low_quality_report_path = REPORT_DIR / "low_quality_audit.tsv"
    write_low_quality_report(low_quality_report_path, all_low_quality_rows)
    print(f"[audit] wrote {low_quality_report_path}")
    label_mismatch_report_path = REPORT_DIR / "label_mismatch_audit.tsv"
    write_label_mismatch_report(label_mismatch_report_path, all_label_mismatch_rows)
    print(f"[audit] wrote {label_mismatch_report_path}")

    if args.apply:
        move_duplicates(all_duplicate_rows)
        move_bad_headers(all_bad_header_rows)
        move_low_quality(all_low_quality_rows)
        move_label_mismatches(all_label_mismatch_rows)
        print(f"[clean] moved {len(all_duplicate_rows)} duplicate files to {BACKUP_ROOT}")
        print(f"[clean] moved {len(all_bad_header_rows)} bad-header files to {BAD_HEADER_BACKUP_ROOT}")
        print(f"[clean] moved {len(all_low_quality_rows)} low-quality files to {LOW_QUALITY_BACKUP_ROOT}")
        print(
            f"[clean] moved {len(all_label_mismatch_rows)} label-mismatch files to "
            f"{LABEL_MISMATCH_BACKUP_ROOT}"
        )
    else:
        print("[audit] dry run only; pass --apply to move duplicates.")


if __name__ == "__main__":
    main()
