#!/usr/bin/env python3

Simple, ready-to-run pipeline for:
- Parsing one or more GenBank files containing candidate prophage / hallmark regions
- Summarizing region lengths and basic CDS counts (coarse insertion-site context)
- Integration hotspot-style summary over boundaries
- Mechanistic junction analysis:
  * Exact terminal repeat scan (candidate att-like cores)
  * Mismatch-tolerant local alignment between termini
  * Simple TSD-like boundary duplication scan

USAGE (examples):

  python insertion_hotspot_mechanistic_analysis.py       --genbank ASM1993100_hallmarkR1.gbk ASM2515056_hallmarkR2.gbk       --out_prefix my_prophage_analysis

This will create CSV outputs like:
  my_prophage_analysis_region_summary.csv
  my_prophage_analysis_mechanistic_junctions.csv
  my_prophage_analysis_hotspot_summary.csv

Dependencies: Biopython, numpy, pandas, tqdm
  pip install biopython numpy pandas tqdm

import argparse
import os

from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from Bio import pairwise2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insertion site mapping, integration hotspot, and mechanistic junction analysis for hallmark regions in GenBank format."
    )
    parser.add_argument(
        "--genbank",
        nargs="+",
        required=True,
        help="One or more input GenBank files containing individual hallmark regions (one record per file is assumed).",
    )
    parser.add_argument(
        "--out_prefix",
        required=True,
        help="Prefix for output CSV files (e.g. results/prophage_run1).",
    )
    parser.add_argument(
        "--term_window",
        type=int,
        default=2500,
        help="Size of terminal windows (bp) from each end of region for junction analyses (default: 2500).",
    )
    parser.add_argument(
        "--min_exact_repeat",
        type=int,
        default=12,
        help="Minimum length (bp) for reporting a shared exact terminal repeat (default: 12).",
    )
    parser.add_argument(
        "--max_exact_repeat",
        type=int,
        default=80,
        help="Maximum length (bp) for searching shared exact terminal repeats (default: 80).",
    )
    parser.add_argument(
        "--max_tsd_len",
        type=int,
        default=25,
        help="Maximum length (bp) for testing TSD-like boundary duplications (default: 25).",
    )
    return parser.parse_args()


def load_genbank_regions(gbk_files: List[str]) -> Dict[str, Dict[str, Any]]:
    regions: Dict[str, Dict[str, Any]] = {}
    for fn in gbk_files:
        recs = list(SeqIO.parse(fn, "genbank"))
        if not recs:
            print("Warning: no records found in " + fn)
            continue
        if len(recs) > 1:
            print("Warning: more than one record in " + fn + "; only the first will be used.")
        rec = recs[0]
        seq_str = str(rec.seq).upper()
        cds_count = sum(1 for f in rec.features if f.type == "CDS")
        regions[fn] = {
            "id": rec.id,
            "description": rec.description,
            "length": len(seq_str),
            "seq": seq_str,
            "cds_count": cds_count,
        }
    return regions


def summarize_regions(regions: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for fn, info in regions.items():
        rows.append(
            {
                "Region_file": os.path.basename(fn),
                "Record_id": info["id"],
                "Description": info["description"],
                "Length_bp": info["length"],
                "CDS_count": info["cds_count"],
            }
        )
    return pd.DataFrame(rows)


def best_exact_repeat_with_coords(seq_str: str, win: int, min_len: int, max_len: int) -> Dict[str, Any]:
    left_window = seq_str[:win]
    right_window = seq_str[-win:]

    for k in range(max_len, min_len - 1, -1):
        seen: Dict[str, int] = {}
        max_left = max(0, len(left_window) - k + 1)
        for i in range(0, max_left):
            mer = left_window[i : i + k]
            if "N" in mer:
                continue
            if mer not in seen:
                seen[mer] = i
        max_right = max(0, len(right_window) - k + 1)
        for j in range(0, max_right):
            mer = right_window[j : j + k]
            if mer in seen:
                i = seen[mer]
                left_start = i
                left_end = i + k
                right_start = (len(seq_str) - win) + j
                right_end = right_start + k
                return {
                    "Exact_terminal_repeat_bp": k,
                    "Exact_terminal_repeat_seq": mer,
                    "Left_repeat_start_0b": left_start,
                    "Left_repeat_end_0b": left_end,
                    "Right_repeat_start_0b": right_start,
                    "Right_repeat_end_0b": right_end,
                }
    return {
        "Exact_terminal_repeat_bp": np.nan,
        "Exact_terminal_repeat_seq": np.nan,
        "Left_repeat_start_0b": np.nan,
        "Left_repeat_end_0b": np.nan,
        "Right_repeat_start_0b": np.nan,
        "Right_repeat_end_0b": np.nan,
    }


def terminal_local_alignment(seq_str: str, win: int) -> Dict[str, Any]:
    left_window = seq_str[:win]
    right_window = seq_str[-win:]
    if len(left_window) == 0 or len(right_window) == 0:
        return {"Best_local_aln_len_bp": np.nan, "Best_local_aln_identity": np.nan}

    aln = pairwise2.align.localms(left_window, right_window, 2, -1, -5, -1, one_alignment_only=True)
    if not aln:
        return {"Best_local_aln_len_bp": np.nan, "Best_local_aln_identity": np.nan}

    aln_rec = aln[0]
    aln_left = aln_rec.seqA
    aln_right = aln_rec.seqB
    matches = 0
    aligned_bases = 0
    for a, b in zip(aln_left, aln_right):
        if a == "-" or b == "-":
            continue
        aligned_bases += 1
        if a == b:
            matches += 1
    if aligned_bases == 0:
        identity = np.nan
    else:
        identity = float(matches) / float(aligned_bases)

    return {"Best_local_aln_len_bp": aligned_bases, "Best_local_aln_identity": identity}


def tsd_like_boundary_scan(seq_str: str, max_tsd_len: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "TSD_found": False,
        "TSD_len_bp": np.nan,
        "TSD_seq": np.nan,
    }
    n = len(seq_str)
    if n == 0:
        return out

    left_flank = seq_str[: max_tsd_len]
    right_flank = seq_str[n - max_tsd_len :]

    for k in range(max_tsd_len, 1, -1):
        left_suffix = left_flank[-k:]
        right_prefix = right_flank[:k]
        if left_suffix == right_prefix:
            out["TSD_found"] = True
            out["TSD_len_bp"] = k
            out["TSD_seq"] = left_suffix
            return out
    return out


def integration_hotspot_summary(regions: Dict[str, Dict[str, Any]], term_window: int) -> pd.DataFrame:
    rows = []
    for fn, info in regions.items():
        length = info["length"]
        rows.append(
            {
                "Region_file": os.path.basename(fn),
                "Length_bp": length,
                "Left_boundary_start": 0,
                "Left_boundary_end": min(term_window, length),
                "Right_boundary_start": max(0, length - term_window),
                "Right_boundary_end": length,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    regions = load_genbank_regions(args.genbank)
    if not regions:
        print("No valid GenBank records loaded. Exiting.")
        return

    print("Loaded " + str(len(regions)) + " hallmark regions.")

    region_df = summarize_regions(regions)

    hotspot_df = integration_hotspot_summary(regions, args.term_window)

    mech_rows = []
    for fn, info in tqdm(regions.items(), desc="Mechanistic junction analysis"):
        seq_str = info["seq"]
        base_row: Dict[str, Any] = {
            "Region_file": os.path.basename(fn),
            "Length_bp": info["length"],
        }
        exact_res = best_exact_repeat_with_coords(
            seq_str,
            win=args.term_window,
            min_len=args.min_exact_repeat,
            max_len=args.max_exact_repeat,
        )
        aln_res = terminal_local_alignment(seq_str, win=args.term_window)
        tsd_res = tsd_like_boundary_scan(seq_str, max_tsd_len=args.max_tsd_len)

        comb = base_row.copy()
        comb.update(exact_res)
        comb.update(aln_res)
        comb.update(tsd_res)
        mech_rows.append(comb)

    mech_df = pd.DataFrame(mech_rows)

    out_prefix = args.out_prefix
    out_region = out_prefix + "_region_summary.csv"
    out_hotspot = out_prefix + "_hotspot_summary.csv"
    out_mech = out_prefix + "_mechanistic_junctions.csv"

    region_df.to_csv(out_region, index=False)
    hotspot_df.to_csv(out_hotspot, index=False)
    mech_df.to_csv(out_mech, index=False)

    print("Saved region summary to " + out_region)
    print("Saved hotspot summary to " + out_hotspot)
    print("Saved mechanistic junction summary to " + out_mech)


if __name__ == "__main__":
    main()
