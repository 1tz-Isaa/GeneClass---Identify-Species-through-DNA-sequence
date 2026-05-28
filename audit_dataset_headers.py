"""Audit suspicious FASTA headers that often hurt model accuracy.

Run:
  python3 audit_dataset_headers.py
"""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

from dataset_loader import load_dataset

PATTERNS = {
    "patent": "patent",
    "jp_publication": " jp ",
    "kr_publication": " kr ",
    "synthetic_construct": "synthetic construct",
    "vector": "vector",
    "plasmid": "plasmid",
    "cloning_vector": "cloning vector",
    "unverified": "unverified",
    "partial_genome": "partial genome",
    "partial_sequence": "partial sequence",
    "partial_cds": "partial cds",
    "rna_construct": "rna construct",
    "composition": "composition",
    "oligonucleotide": "oligonucleotide",
    "extracellular_vesicle": "extracellular vesicle",
    "vaccine_against": "vaccine against",
    "circular_rna": "circular rna",
}


def detect_reasons(header: str) -> list[str]:
    h = f" {(header or '').lower()} "
    reasons = [name for name, token in PATTERNS.items() if token in h]
    return reasons


def main() -> None:
    rows = []
    for root in ("DNA", "RNA"):
        for item in load_dataset(root, show_progress=False):
            header = item.get("header", "")
            reasons = detect_reasons(header)
            if not reasons:
                continue
            rows.append(
                {
                    "source_file": item.get("source_file", ""),
                    "domain": item.get("domain", ""),
                    "kingdom": item.get("kingdom", ""),
                    "genus": item.get("genus", ""),
                    "species": item.get("species", ""),
                    "reasons": ",".join(reasons),
                    "header": header,
                }
            )

    out_dir = Path("runs/training_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"header_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_file", "domain", "kingdom", "genus", "species", "reasons", "header"],
        )
        writer.writeheader()
        writer.writerows(rows)

    by_kingdom = Counter(r["kingdom"] for r in rows)
    by_reason = Counter(reason for r in rows for reason in r["reasons"].split(",") if reason)

    print(f"Suspicious records: {len(rows)}")
    print("By kingdom:")
    for k, v in sorted(by_kingdom.items()):
        print(f"  - {k}: {v}")

    print("By reason:")
    for k, v in sorted(by_reason.items()):
        print(f"  - {k}: {v}")

    print(f"CSV report: {out_path.resolve()}")


if __name__ == "__main__":
    main()
