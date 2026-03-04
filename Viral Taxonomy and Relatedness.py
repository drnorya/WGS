# Viral taxonomy and relatedness analysis script
# ----------------------------------------------
# What it does:
# 1. Reads a FASTA file of viral sequences.
# 2. Runs BLAST (online) against NCBI nt or nr to find closest hits.
# 3. Fetches taxonomy information for top hits.
# 4. Computes pairwise distances and builds a neighbor-joining tree.
# 5. Saves CSV summaries and a tree figure.
#
# How to run:
#   python viral_taxonomy_relatedness.py \
#       --fasta input_sequences.fasta \
#       --output_prefix results/viral_analysis \
#       --email your_email@example.com \
#       --blast_db nt \
#       --seq_type nucl
#
# For protein sequences, use:
#   --blast_db nr --seq_type prot
#
# Note:
# - This uses NCBI web services. Do not overload their servers.
# - For larger datasets, increase delay between requests.
# - BLAST for many sequences can take time.

import argparse
import time
from io import StringIO

import pandas as pd
from tqdm import tqdm

from Bio import Entrez, SeqIO, Phylo
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Align import MultipleSeqAlignment
from Bio import AlignIO
from Bio.Align.Applications import ClustalOmegaCommandline
import matplotlib.pyplot as plt
import os
import tempfile
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description="Viral taxonomy and relatedness analysis using NCBI BLAST and phylogenetics."
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Input FASTA file with viral sequences (nucl or prot)."
    )
    parser.add_argument(
        "--output_prefix",
        required=True,
        help="Prefix for output files (e.g., results/viral_analysis)."
    )
    parser.add_argument(
        "--email",
        required=True,
        help="Your email for NCBI Entrez (required by NCBI)."
    )
    parser.add_argument(
        "--ncbi_api_key",
        default=None,
        help="Optional NCBI API key to increase rate limits."
    )
    parser.add_argument(
        "--blast_db",
        default="nt",
        choices=["nt", "nr"],
        help="BLAST database: nt for nucleotide, nr for protein."
    )
    parser.add_argument(
        "--seq_type",
        default="nucl",
        choices=["nucl", "prot"],
        help="Sequence type: nucl (DNA/RNA) or prot (protein)."
    )
    parser.add_argument(
        "--max_seqs",
        type=int,
        default=None,
        help="Optionally limit number of sequences to analyze (for testing)."
    )
    parser.add_argument(
        "--blast_max_hits",
        type=int,
        default=5,
        help="Number of BLAST hits to retrieve per sequence (top N)."
    )
    parser.add_argument(
        "--blast_delay",
        type=float,
        default=5.0,
        help="Seconds to sleep between BLAST requests (be kind to NCBI)."
    )
    parser.add_argument(
        "--taxonomy_delay",
        type=float,
        default=0.4,
        help="Seconds to sleep between taxonomy Entrez requests."
    )
    parser.add_argument(
        "--alignment_tool",
        default="clustalo",
        choices=["clustalo", "none"],
        help="Tool for multiple sequence alignment (clustalo requires Clustal Omega installed). "
             "If 'none', will compute distances using simple pairwise identity (very rough)."
    )

    return parser.parse_args()


def setup_entrez(email, api_key=None):
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key


def run_blast_for_sequence(record, blast_db, seq_type, blast_max_hits):
    """Run BLAST (blastn or blastp) for a single SeqRecord using NCBIWWW."""
    sequence_str = str(record.seq)
    if not sequence_str or sequence_str.strip() == "":
        return None

    if seq_type == "nucl":
        program = "blastn"
    else:
        program = "blastp"

    # Use NCBIWWW.qblast
    result_handle = NCBIWWW.qblast(
        program=program,
        database=blast_db,
        sequence=sequence_str,
        hitlist_size=blast_max_hits,
        format_type="XML"
    )

    blast_record = NCBIXML.read(result_handle)
    result_handle.close()
    return blast_record


def parse_blast_top_hit(blast_record):
    """Extract top hit info from a BLAST record."""
    if not blast_record.alignments:
        return None

    alignment = blast_record.alignments[0]
    hsp = alignment.hsps[0]

    hit_id = alignment.hit_id.split()[0]
    description = alignment.hit_def
    evalue = hsp.expect
    identity = float(hsp.identities) / float(hsp.align_length)
    coverage = float(hsp.align_length) / float(blast_record.query_length)

    return {
        "hit_id": hit_id,
        "hit_description": description,
        "evalue": evalue,
        "identity": identity,
        "coverage": coverage,
    }


def fetch_taxonomy_for_accession(accession, taxonomy_delay):
    """Fetch taxonomy lineage for an accession using NCBI Entrez."""
    time.sleep(taxonomy_delay)
    try:
        handle = Entrez.efetch(db="nuccore", id=accession, rettype="gb", retmode="text")
    except Exception:
        # Try protein DB as fallback
        try:
            handle = Entrez.efetch(db="protein", id=accession, rettype="gb", retmode="text")
        except Exception:
            return None

    gb_text = handle.read()
    handle.close()

    # Simple parsing of taxonomy line from GenBank text
    lineage = None
    organism = None
    for line in gb_text.splitlines():
        if line.startswith("  ORGANISM"):
            organism = line.replace("  ORGANISM", "").strip()
        if "ORGANISM" in line:
            # Next line should contain lineage; but GenBank format is multi-line.
            pass

    # A more robust way is to use Entrez.efetch with rettype xml and parse, but
    # here we will also try esummary for taxonomy information.
    try:
        time.sleep(taxonomy_delay)
        summary_handle = Entrez.esummary(db="nuccore", id=accession, retmode="xml")
        summary = Entrez.read(summary_handle)
        summary_handle.close()
        if summary and "TaxId" in summary[0]:
            taxid = summary[0]["TaxId"]
            time.sleep(taxonomy_delay)
            tax_handle = Entrez.efetch(db="taxonomy", id=taxid, retmode="xml")
            tax_data = Entrez.read(tax_handle)
            tax_handle.close()
            if tax_data:
                lineage_list = tax_data[0].get("LineageEx", [])
                lineage = ";".join([x["ScientificName"] for x in lineage_list])
                if not organism:
                    organism = tax_data[0].get("ScientificName", None)
                return {
                    "taxid": taxid,
                    "organism": organism,
                    "lineage": lineage,
                }
    except Exception:
        pass

    if not organism and not lineage:
        return None

    return {
        "taxid": None,
        "organism": organism,
        "lineage": lineage,
    }


def run_blast_and_taxonomy(records, blast_db, seq_type, blast_max_hits, blast_delay, taxonomy_delay):
    """Run BLAST and fetch taxonomy for a list of SeqRecords."""
    results = []

    for record in tqdm(records, desc="Running BLAST and fetching taxonomy"):
        try:
            blast_record = run_blast_for_sequence(record, blast_db, seq_type, blast_max_hits)
        except Exception as e:
            print("BLAST failed for sequence " + str(record.id) + " with error " + str(e))
            continue

        if blast_record is None:
            continue

        top_hit_info = parse_blast_top_hit(blast_record)
        if top_hit_info is None:
            continue

        accession = top_hit_info["hit_id"].split("|")[0]  # crude accession parsing

        tax_info = None
        try:
            tax_info = fetch_taxonomy_for_accession(accession, taxonomy_delay)
        except Exception as e:
            print("Taxonomy fetch failed for " + str(accession) + " with error " + str(e))

        combined = {
            "query_id": record.id,
            "query_length": len(record.seq),
            "hit_accession": accession,
            "hit_description": top_hit_info["hit_description"],
            "hit_evalue": top_hit_info["evalue"],
            "hit_identity": top_hit_info["identity"],
            "hit_coverage": top_hit_info["coverage"],
        }

        if tax_info:
            combined["taxid"] = tax_info.get("taxid", None)
            combined["organism"] = tax_info.get("organism", None)
            combined["lineage"] = tax_info.get("lineage", None)
        else:
            combined["taxid"] = None
            combined["organism"] = None
            combined["lineage"] = None

        results.append(combined)

        time.sleep(blast_delay)

    return pd.DataFrame(results)


def perform_alignment(records, alignment_tool):
    """Perform multiple sequence alignment; currently supports Clustal Omega."""
    if alignment_tool == "none":
        # Turn sequences into a MultipleSeqAlignment for distance calculator
        alignment = MultipleSeqAlignment(records)
        return alignment

    if alignment_tool == "clustalo":
        # Requires Clustal Omega installed and accessible as 'clustalo'
        with tempfile.TemporaryDirectory() as tmpdir:
            input_fasta = os.path.join(tmpdir, "input.fasta")
            output_fasta = os.path.join(tmpdir, "aligned.fasta")
            SeqIO.write(records, input_fasta, "fasta")

            clustal_cline = ClustalOmegaCommandline(
                infile=input_fasta,
                outfile=output_fasta,
                verbose=True,
                auto=True,
                force=True
            )

            # Run the command
            command_str = str(clustal_cline)
            print("Running alignment: " + command_str)
            subprocess.run(command_str.split(), check=True)

            # Read alignment
            alignment = AlignIO.read(output_fasta, "fasta")
            return alignment

    raise ValueError("Unsupported alignment tool " + str(alignment_tool))


def compute_distance_matrix(alignment, seq_type):
    """Compute a distance matrix from an alignment using Biopython DistanceCalculator."""
    if seq_type == "nucl":
        model = "identity"
    else:
        model = "blosum62"

    calculator = DistanceCalculator(model)
    dm = calculator.get_distance(alignment)
    return dm


def build_tree(distance_matrix):
    """Build a neighbor-joining tree from a distance matrix."""
    constructor = DistanceTreeConstructor()
    nj_tree = constructor.nj(distance_matrix)
    return nj_tree


def plot_and_save_tree(tree, output_prefix):
    """Plot and save the phylogenetic tree."""
    fig = plt.figure(figsize=(10, 8))
    axes = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree, do_show=False, axes=axes)
    plt.title("Neighbor-Joining Tree of Viral Sequences")
    tree_png = output_prefix + "_tree.png"
    plt.savefig(tree_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved tree image to " + tree_png)


def main():
    args = parse_args()

    setup_entrez(args.email, args.ncbi_api_key)

    # Load sequences
    records = list(SeqIO.parse(args.fasta, "fasta"))
    if args.max_seqs is not None:
        records = records[:args.max_seqs]

    print("Loaded " + str(len(records)) + " sequences from " + str(args.fasta))

    if len(records) < 2:
        print("Need at least two sequences for relatedness/tree analysis.")
        return

    # Create output directory if needed
    out_dir = os.path.dirname(args.output_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # BLAST + Taxonomy
    blast_tax_df = run_blast_and_taxonomy(
        records=records,
        blast_db=args.blast_db,
        seq_type=args.seq_type,
        blast_max_hits=args.blast_max_hits,
        blast_delay=args.blast_delay,
        taxonomy_delay=args.taxonomy_delay,
    )

    blast_tax_csv = args.output_prefix + "_blast_taxonomy.csv"
    blast_tax_df.to_csv(blast_tax_csv, index=False)
    print("Saved BLAST/taxonomy results to " + blast_tax_csv)

    # Alignment and distance matrix
    try:
        alignment = perform_alignment(records, args.alignment_tool)
    except Exception as e:
        print("Alignment failed with error: " + str(e))
        print("Falling back to unaligned distance (identity model over equal-length sequences).")

        # Simple fallback: convert to MultipleSeqAlignment without real MSA
        alignment = MultipleSeqAlignment(records)

    distance_matrix = compute_distance_matrix(alignment, args.seq_type)

    # Convert distance matrix to DataFrame for inspection/saving
    names = list(distance_matrix.names)
    dist_df = pd.DataFrame(
        distance_matrix.matrix,
        index=names,
        columns=names
    )

    dist_csv = args.output_prefix + "_distance_matrix.csv"
    dist_df.to_csv(dist_csv)
    print("Saved distance matrix to " + dist_csv)

    # Build and save tree
    tree = build_tree(distance_matrix)
    tree_nwk = args.output_prefix + "_tree.nwk"
    Phylo.write(tree, tree_nwk, "newick")
    print("Saved tree (Newick) to " + tree_nwk)

    plot_and_save_tree(tree, args.output_prefix)

    print("Analysis complete.")


if __name__ == "__main__":
    main()