
"""
prophage_lifestyle_analysis.py

Generic prophage lifestyle analysis from GenBank files.

Given one or more GenBank files (prophage or hallmark regions), this script:

1) Parses CDS features and builds a per-gene table.
2) Assigns each CDS to a functional module using simple keyword rules.
3) Builds a per-region module presence/absence + counts matrix.
4) Applies transparent rule-based logic to infer prophage lifestyle:
       - Temperate-capable (strong evidence)
       - Likely temperate (partial evidence)
       - Lytic-like
       - Incomplete/uncertain
5) Writes results as CSV tables.

Dependencies:
    biopython
    pandas

Install (if needed):
    pip install biopython pandas

Usage from command line:
    python prophage_lifestyle_analysis.py region1.gbk region2.gbk ...

Outputs:
    genes_annotated.csv              - per-CDS annotations and functional class
    region_module_counts.csv         - counts of genes per module per region
    region_module_presence.csv       - presence/absence (0/1) per module per region
    region_lifestyle_summary.csv     - lifestyle call + evidence flags per region
"""

import os
import sys
from typing import List, Dict, Tuple

import pandas as pd
from Bio import SeqIO

# -----------------------------
# 1. Parse GenBank -> CDS table
# -----------------------------

def parse_gbk_to_cds_df(gbk_paths: List[str]) -> pd.DataFrame:
    """Parse one or more GenBank files and return a CDS dataframe.

    Columns:
        region_id   basename of .gbk without extension
        record_id   GenBank record id
        locus_tag
        gene
        product
        start
        end
        strand      +1 or -1
        nt_len
        aa_len      length of translation (if available)
    """
    rows = []

    for gbk_path in gbk_paths:
        region_id = os.path.splitext(os.path.basename(gbk_path))[0]

        for record in SeqIO.parse(gbk_path, "genbank"):
            rec_id = record.id

            for feat in record.features:
                if feat.type != "CDS":
                    continue

                start = int(feat.location.nofuzzy_start)
                end = int(feat.location.nofuzzy_end)
                strand = int(feat.location.strand or 0)

                q = feat.qualifiers
                locus_tag = q.get("locus_tag", [""])[0]
                gene = q.get("gene", [""])[0]
                product = q.get("product", [""])[0]
                translation = q.get("translation", [""])[0]

                nt_len = abs(end - start)
                aa_len = len(translation) if translation else None

                rows.append({
                    "region_id": region_id,
                    "record_id": rec_id,
                    "locus_tag": locus_tag,
                    "gene": gene,
                    "product": product,
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "nt_len": nt_len,
                    "aa_len": aa_len,
                })

    if not rows:
        raise ValueError("No CDS features found in provided GenBank files.")

    return pd.DataFrame(rows)


# ---------------------------------
# 2. Assign functional module class
# ---------------------------------

MODULE_KEYWORDS: Dict[str, List[str]] = {
    "Integration_recombination": [
        "integrase", "recombinase", "xer", "resolvase", "int "
    ],
    "Lysogeny_regulation": [
        "repressor", "cI", "cro", "lexA", "antirepressor"
    ],
    "Excision": [
        "excisionase", "xis"
    ],
    "Replication_partition": [
        "replication", "rep protein", "replicase", "repA", "repB",
        "helicase", "primase", "dnaC", "dnaB", "parA", "parB",
        "partition", "par protein"
    ],
    "Packaging_structure": [
        "terminase", "portal", "capsid", "head", "tail", "tail fiber",
        "tape measure", "baseplate", "neck protein", "connector"
    ],
    "Lysis": [
        "holin", "endolysin", "lysin", "muramidase", "lysozyme"
    ],
}

MODULE_ORDER = [
    "Integration_recombination",
    "Lysogeny_regulation",
    "Excision",
    "Replication_partition",
    "Packaging_structure",
    "Lysis",
]


def assign_func_class(cds_df: pd.DataFrame) -> pd.DataFrame:
    """Assign a functional class to each CDS using keyword rules.

    Adds column:
        func_class (one of MODULE_ORDER or "Other_or_hypothetical")
    """
    def classify_row(row) -> str:
        text = ((row.get("product", "") or "") + " " + (row.get("gene", "") or "")).lower()

        # if clearly hypothetical/unknown and no keyword hits
        if "hypothetical" in text or "unknown" in text or "uncharacterized" in text:
            # we still allow keyword overrides below
            default_class = "Other_or_hypothetical"
        else:
            default_class = "Other_or_hypothetical"

        for module, kws in MODULE_KEYWORDS.items():
            for kw in kws:
                if kw.lower() in text:
                    return module
        return default_class

    cds_df = cds_df.copy()
    cds_df["func_class"] = cds_df.apply(classify_row, axis=1)
    return cds_df


# -----------------------------------------------
# 3. Summaries: module counts and presence by region
# -----------------------------------------------

def summarize_modules(cds_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create per-region module counts and presence/absence tables.

    Returns:
        counts_df   region_id x module (integer counts)
        presence_df region_id x module (0/1)
    """
    # restrict to named modules we care about
    temp = cds_df.copy()
    temp = temp[temp["func_class"].isin(MODULE_ORDER)]

    if temp.empty:
        # no markers, create empty frames
        regions = sorted(cds_df["region_id"].unique().tolist())
        counts_df = pd.DataFrame(0, index=regions, columns=MODULE_ORDER)
        counts_df.index.name = "region_id"
        presence_df = (counts_df > 0).astype(int)
        return counts_df, presence_df

    counts = temp.pivot_table(
        index="region_id",
        columns="func_class",
        values="locus_tag",
        aggfunc="count",
        fill_value=0,
    )

    # ensure all expected modules exist as columns
    for mod in MODULE_ORDER:
        if mod not in counts.columns:
            counts[mod] = 0

    counts = counts[MODULE_ORDER].sort_index()
    presence = (counts > 0).astype(int)

    counts.index.name = "region_id"
    presence.index.name = "region_id"

    return counts, presence


# ---------------------------------
# 4. Rule-based lifestyle inference
# ---------------------------------

def infer_lifestyle(counts_df: pd.DataFrame, presence_df: pd.DataFrame) -> pd.DataFrame:
    """Infer prophage lifestyle per region based on module presence.

    Returns dataframe with columns:
        region_id
        lifestyle_call
        evidence_integration
        evidence_lysogeny
        evidence_excision
        evidence_replication
        evidence_structure
        evidence_lysis
        comment

    Simple logic (you can adjust later):
        - Strong temperate:
            Integration_recombination AND Lysogeny_regulation present
        - Likely temperate:
            Integration_recombination present AND (Excision OR Replication_partition present)
        - Lytic-like:
            Packaging_structure AND Lysis present, but no integration/lysogeny modules
        - Incomplete/uncertain:
            everything else
    """
    rows = []

    for region_id, row in presence_df.iterrows():
        has_int = int(row.get("Integration_recombination", 0)) == 1
        has_lys = int(row.get("Lysogeny_regulation", 0)) == 1
        has_exc = int(row.get("Excision", 0)) == 1
        has_rep = int(row.get("Replication_partition", 0)) == 1
        has_str = int(row.get("Packaging_structure", 0)) == 1
        has_lys_mod = int(row.get("Lysis", 0)) == 1

        if has_int and has_lys:
            lifestyle = "Temperate-capable (integrase + repressor)"
        elif has_int and (has_exc or has_rep):
            lifestyle = "Likely temperate (integrase + supporting modules)"
        elif (not has_int) and (not has_lys) and has_str and has_lys_mod:
            lifestyle = "Lytic-like (structural + lysis, no integration/lysogeny markers)"
        else:
            lifestyle = "Incomplete/uncertain prophage-like region"

        comment_parts = []
        if has_int:
            comment_parts.append("integration markers")
        if has_lys:
            comment_parts.append("lysogeny regulators")
        if has_exc:
            comment_parts.append("excision genes")
        if has_rep:
            comment_parts.append("replication/partition genes")
        if has_str:
            comment_parts.append("structural/packaging genes")
        if has_lys_mod:
            comment_parts.append("lysis genes")

        comment = ", ".join(comment_parts) if comment_parts else "no hallmark modules detected by keyword rules"

        rows.append({
            "region_id": region_id,
            "lifestyle_call": lifestyle,
            "evidence_integration": has_int,
            "evidence_lysogeny": has_lys,
            "evidence_excision": has_exc,
            "evidence_replication": has_rep,
            "evidence_structure": has_str,
            "evidence_lysis": has_lys_mod,
            "comment": comment,
        })

    return pd.DataFrame(rows).sort_values("region_id")


# ---------------------------------
# 5. Simple command-line interface
# ---------------------------------

def main(argv: List[str]) -> None:
    if len(argv) < 2:
        print("Usage: python prophage_lifestyle_analysis.py <region1.gbk> [region2.gbk ...]")
        sys.exit(1)

    gbk_files = argv[1:]
    print("Parsing GenBank files...")
    cds_df = parse_gbk_to_cds_df(gbk_files)

    print("Assigning functional modules...")
    cds_df = assign_func_class(cds_df)

    print("Summarizing modules by region...")
    counts_df, presence_df = summarize_modules(cds_df)

    print("Inferring lifestyle per region...")
    lifestyle_df = infer_lifestyle(counts_df, presence_df)

    # Write outputs
    cds_df.to_csv("genes_annotated.csv", index=False)
    counts_df.to_csv("region_module_counts.csv")
    presence_df.to_csv("region_module_presence.csv")
    lifestyle_df.to_csv("region_lifestyle_summary.csv", index=False)

    print("Done.")
    print("Per-gene table: genes_annotated.csv")
    print("Module counts: region_module_counts.csv")
    print("Module presence/absence: region_module_presence.csv")
    print("Lifestyle summary: region_lifestyle_summary.csv")


if __name__ == "__main__":
    main(sys.argv)
