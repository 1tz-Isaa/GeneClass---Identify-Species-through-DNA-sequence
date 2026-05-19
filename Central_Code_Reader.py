"""CLI for central kingdom routing + genus prediction from FASTA.

Examples:
  python3 Central_Code_Reader.py --input DNA/Fungi/Candida/albicans/CA_sample2.fasta
  python3 Central_Code_Reader.py --input my_wgs.fasta --source best --max-router-seqs 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from central_reader import parse_fasta_file, run_central_reader


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Central code reader for kingdom routing")
    p.add_argument("--input", required=True, help="Path to FASTA file")
    p.add_argument("--source", choices=["best", "latest"], default="best", help="Model source")
    p.add_argument("--min-len", type=int, default=120, help="Minimum length for router subset")
    p.add_argument("--max-router-seqs", type=int, default=80, help="Max sequences used for routing")
    p.add_argument("--max-predict-seqs", type=int, default=120, help="Max sequences shown in predictions")
    p.add_argument("--router-window", type=int, default=1200, help="Window size for kingdom routing")
    p.add_argument("--router-stride", type=int, default=600, help="Stride for kingdom routing windows")
    p.add_argument("--predict-window", type=int, default=1500, help="Window size for final genus prediction")
    p.add_argument("--predict-stride", type=int, default=750, help="Stride for final prediction windows")
    p.add_argument(
        "--max-predict-windows-per-record",
        type=int,
        default=40,
        help="Max windows per record in final prediction",
    )
    p.add_argument(
        "--reject-threshold",
        type=float,
        default=0.0,
        help="If confidence < threshold, output UNCERTAIN",
    )
    p.add_argument(
        "--force-target",
        choices=["bacteria", "fungi", "rna"],
        default=None,
        help="Force a target model and skip auto kingdom routing",
    )
    p.add_argument("--top-show", type=int, default=10, help="How many prediction rows to print")
    p.add_argument("--json", action="store_true", help="Print full JSON output")
    return p


def main() -> None:
    args = build_parser().parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    records = parse_fasta_file(path)
    if not records:
        raise ValueError("No valid sequences parsed from input")

    result = run_central_reader(
        records,
        model_source=args.source,
        min_len=args.min_len,
        max_router_seqs=args.max_router_seqs,
        max_predict_seqs=args.max_predict_seqs,
        router_window=args.router_window,
        router_stride=args.router_stride,
        predict_window=args.predict_window,
        predict_stride=args.predict_stride,
        max_predict_windows_per_record=args.max_predict_windows_per_record,
        reject_threshold=args.reject_threshold,
        force_target=args.force_target,
    )

    print(f"Input: {path.resolve()}")
    print(f"Parsed sequences: {result['input_sequences']}")
    print(f"Routing method: {result.get('routing_method', 'unknown')}")
    print(f"Detected kingdom: {result['selected_kingdom']} (target={result['selected_target']})")
    print(f"Selected model: {result['selected_model_path']}")
    print("Routing scores:")
    for row in result["routing_scores"]:
        print(
            f"  - {row['target']:8s} score={row['score'] * 100:.2f}% "
            f"mean_conf={row['mean_conf'] * 100:.2f}% margin={row['mean_margin'] * 100:.2f}% n={row['n_used']}"
        )

    if result.get("fit_scores"):
        print("Cross-model fit scores:")
        for target, item in result["fit_scores"].items():
            print(f"  - {target:8s} fit={item['score'] * 100:.2f}% n={item['n_records']}")

    print("Predictions (top rows):")
    for idx, row in enumerate(result["predictions"][: args.top_show], start=1):
        conf = row["confidence"] if row["confidence"] is not None else float("nan")
        print(
            f"  {idx:02d}. len={row['length']:6d} windows={row.get('n_windows', 1):3d} "
            f"pred={row['prediction']} conf={conf * 100:.2f}% "
            f"header={row['header']}"
        )

    if args.json:
        print("\nJSON output:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
