"""
Build initial KG from PubChem and other data sources.

Extracts edges from: bioactivity, consolidated targets, clinical trials,
OpenTargets indications, chemical/gene co-occurrence JSONs, and literature
adverse-event mining. Outputs kg_edges_v2.csv and kg_nodes_v2.csv for use
by enrich_go.py.

Usage:
  python scripts/kg_construction/build_kg_from_sources.py --data-dir data --out-edges data/kg_edges_v2.csv --out-nodes data/kg_nodes_v2.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# Default config (overridable by CLI)
DRUG_NAME = "Pralsetinib"
CID = "129073603"


def process_bioactivity(filepath: Path, drug: str) -> pd.DataFrame:
    """Extract (drug, inhibits, protein) edges from PubChem bioactivity CSV."""
    if not filepath.exists():
        print(f"  Skipping (not found): {filepath}")
        return pd.DataFrame()
    print(f"  Processing bioactivity: {filepath.name}")
    df = pd.read_csv(filepath)
    df = df[["Target_Name", "Activity_Type", "Activity_Value", "BioAssay_AID"]].copy()
    df["value_nM"] = df["Activity_Value"] * 1000
    df = df.dropna(subset=["value_nM"])
    edges = pd.DataFrame({
        "source": drug,
        "relation": "inhibits",
        "target": df["Target_Name"],
        "value": df["value_nM"],
        "unit": "nM",
        "metadata": df["Activity_Type"] + " (AID: " + df["BioAssay_AID"].astype(str) + ")",
        "source_file": "Bioactivity",
        "target_type": "Protein",
    })
    print(f"    -> {len(edges)} edges")
    return edges


def process_targets(filepath: Path, drug: str) -> pd.DataFrame:
    """Extract (drug, relation, gene) edges from consolidated compound target CSV."""
    if not filepath.exists():
        print(f"  Skipping (not found): {filepath}")
        return pd.DataFrame()
    print(f"  Processing targets: {filepath.name}")
    df = pd.read_csv(filepath)
    action_series = df["Action"].fillna("targets").astype(str).str.lower()
    action_map = {
        "inhibitor": "inhibits",
        "inhibition": "inhibits",
        "antagonist": "inhibits",
        "agonist": "activates",
        "activator": "activates",
    }
    normalized_action = action_series.map(action_map).fillna("targets")
    edges = pd.DataFrame({
        "source": drug,
        "relation": normalized_action,
        "target": df["Gene"],
        "value": "Active",
        "unit": "N/A",
        "metadata": df["Source_Target"],
        "source_file": "ConsolidatedTargets",
        "target_type": "Gene/Protein",
    })
    print(f"    -> {len(edges)} edges")
    return edges


def process_clinical_and_indications(
    file_clinical: Path,
    file_opentargets: Path,
    drug: str,
) -> pd.DataFrame:
    """Extract (drug, treats, disease) from clinical trials and OpenTargets."""
    edges_list = []
    if file_clinical.exists():
        print(f"  Processing clinical trials: {file_clinical.name}")
        df = pd.read_csv(file_clinical)
        temp = df[["Conditions", "Phase", "CTID"]].copy()
        temp["Conditions"] = temp["Conditions"].fillna("").str.split("|")
        temp = temp.explode("Conditions")
        temp["Conditions"] = temp["Conditions"].str.strip()
        temp = temp[temp["Conditions"] != ""]
        edges_list.append(pd.DataFrame({
            "source": drug,
            "relation": "treats",
            "target": temp["Conditions"],
            "value": temp["Phase"],
            "unit": "Phase",
            "metadata": temp["CTID"],
            "source_file": "ClinicalTrials",
            "target_type": "Disease",
        }))
        print(f"    -> {len(edges_list[-1])} edges")
    if file_opentargets.exists():
        print(f"  Processing OpenTargets indications: {file_opentargets.name}")
        df = pd.read_csv(file_opentargets)
        temp = df[["Indication", "Max_Phase"]].copy()
        temp["Indication"] = temp["Indication"].fillna("").str.split("|")
        temp = temp.explode("Indication")
        temp["Indication"] = temp["Indication"].str.strip()
        temp = temp[temp["Indication"] != ""]
        edges_list.append(pd.DataFrame({
            "source": drug,
            "relation": "treats",
            "target": temp["Indication"],
            "value": temp["Max_Phase"],
            "unit": "Phase",
            "metadata": "OpenTargets",
            "source_file": "OpenTargets",
            "target_type": "Disease",
        }))
        print(f"    -> {len(edges_list[-1])} edges")
    if not edges_list:
        return pd.DataFrame()
    return pd.concat(edges_list, ignore_index=True)


def process_json_cooccurrence(
    filepath: Path,
    drug: str,
    relation_label: str,
    target_type: str,
) -> pd.DataFrame:
    """Extract (drug, relation, neighbor) from PubChem LinkData JSON."""
    if not filepath.exists():
        print(f"  Skipping (not found): {filepath}")
        return pd.DataFrame()
    print(f"  Processing JSON co-occurrence: {filepath.name}")
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"    Error: {e}")
        return pd.DataFrame()
    link_dataset = data.get("LinkDataSet", {})
    if isinstance(link_dataset, list):
        source_list = link_dataset
    else:
        source_list = link_dataset.get("LinkData", [])
    extracted = []
    for item in source_list:
        if not isinstance(item, dict):
            continue
        evidence = item.get("Evidence", {})
        if "ChemicalNeighbor" in evidence:
            name = evidence["ChemicalNeighbor"].get("NeighborName")
            if name:
                extracted.append(name)
        elif "ChemicalGeneSymbolNeighbor" in evidence:
            name = evidence["ChemicalGeneSymbolNeighbor"].get("NeighborName")
            if name:
                extracted.append(name)
        elif "NeighborName" in item:
            extracted.append(item["NeighborName"])
    if not extracted:
        print("    No data extracted (structure may differ)")
        return pd.DataFrame()
    source_file_str = str(filepath).replace("\\", "/")
    edges = pd.DataFrame({
        "source": drug,
        "relation": relation_label,
        "target": extracted,
        "value": "Co-occurrence",
        "unit": "Text Mining",
        "metadata": "PubChem Neighbor",
        "source_file": source_file_str,
        "target_type": target_type,
    })
    print(f"    -> {len(edges)} edges")
    return edges


def process_adverse_events(filepath: Path, drug: str) -> pd.DataFrame:
    """Extract (drug, associated_with, adverse_event) from literature CSV."""
    if not filepath.exists():
        print(f"  Skipping (not found): {filepath}")
        return pd.DataFrame()
    print(f"  Processing adverse events (literature): {filepath.name}")
    df = pd.read_csv(filepath)
    ae_keywords = {
        "hypertension": "Hypertension",
        "high blood pressure": "Hypertension",
        "neutropenia": "Neutropenia",
        "anemia": "Anemia",
        "pneumonitis": "Pneumonitis",
        "hepatotoxicity": "Hepatotoxicity",
        "fatigue": "Fatigue",
        "diarrhea": "Diarrhea",
        "constipation": "Constipation",
    }
    found = []
    for _, row in df.iterrows():
        text = (str(row.get("Title", "")) + " " + str(row.get("Abstract", ""))).lower()
        for keyword, standardized in ae_keywords.items():
            if keyword in text:
                found.append({
                    "source": drug,
                    "relation": "associated_with",
                    "target": standardized,
                    "value": "Text Mining",
                    "unit": "Mention",
                    "metadata": f"PMID: {row.get('PMID', 'N/A')}",
                    "source_file": "Literature_Abstracts",
                    "target_type": "Adverse Event",
                })
    if not found:
        print("    No adverse events found")
        return pd.DataFrame()
    edges = pd.DataFrame(found)
    print(f"    -> {len(edges)} edges")
    return edges


def _relation_to_type(rel: str) -> str:
    rel = str(rel).lower()
    if "inhibits" in rel or "inhibitor" in rel or "inhibition" in rel:
        return "Protein"
    if "targets" in rel:
        return "Gene/Protein"
    if "treats" in rel:
        return "Disease"
    if "gene" in rel:
        return "Gene"
    if "chemical" in rel:
        return "Chemical"
    if "side_effect" in rel or "adverse" in rel or "associated_with" in rel:
        return "Adverse Event"
    return "Entity"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build initial KG from PubChem and other data sources.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing PubChem/source CSVs and JSONs",
    )
    parser.add_argument(
        "--out-edges",
        type=Path,
        default=Path("data/kg_edges_v2.csv"),
        help="Output edges CSV path",
    )
    parser.add_argument(
        "--out-nodes",
        type=Path,
        default=Path("data/kg_nodes_v2.csv"),
        help="Output nodes CSV path",
    )
    parser.add_argument(
        "--drug",
        default=DRUG_NAME,
        help="Drug name for source node",
    )
    parser.add_argument(
        "--cid",
        default=CID,
        help="PubChem CID (used only for default filenames if needed)",
    )
    args = parser.parse_args()

    data = args.data_dir
    drug = args.drug

    # Build paths to source files
    bio_path = data / "pubchem_cid_129073603_bioactivity.csv"
    targets_path = data / "pubchem_cid_129073603_consolidatedcompoundtarget.csv"
    clinical_path = data / "pubchem_cid_129073603_clinicaltrials.csv"
    opentargets_path = data / "pubchem_cid_129073603_opentargetsdrugindication.csv"
    chem_cooc_path = data / "Chemical_Co-Occurrences-in-Literature_CID_129073603.json"
    gene_cooc_path = data / "Chemical_Gene-Co-Occurrences-in-Literature_CID_129073603.json"
    literature_path = data / "pubchem_cid_129073603_literature.csv"

    print("Extracting edges from sources...")
    dfs = []

    df_bio = process_bioactivity(bio_path, drug)
    if not df_bio.empty:
        dfs.append(df_bio)

    df_targets = process_targets(targets_path, drug)
    if not df_targets.empty:
        dfs.append(df_targets)

    df_clinical = process_clinical_and_indications(clinical_path, opentargets_path, drug)
    if not df_clinical.empty:
        dfs.append(df_clinical)

    df_chem = process_json_cooccurrence(
        chem_cooc_path, drug, "co_occurs_with_chemical", "Chemical"
    )
    if not df_chem.empty:
        dfs.append(df_chem)

    df_gene = process_json_cooccurrence(
        gene_cooc_path, drug, "co_occurs_with_gene", "Gene"
    )
    if not df_gene.empty:
        dfs.append(df_gene)

    df_ae = process_adverse_events(literature_path, drug)
    if not df_ae.empty:
        dfs.append(df_ae)

    if not dfs:
        print("No data extracted. Check that source files exist under --data-dir.")
        raise SystemExit(1)

    kg_edges = pd.concat(dfs, ignore_index=True)
    kg_edges = kg_edges.dropna(subset=["target"])
    kg_edges = kg_edges[kg_edges["target"].astype(str) != ""]
    kg_edges = kg_edges.drop_duplicates()

    # Build nodes
    nodes_src = kg_edges[["source"]].rename(columns={"source": "id"})
    nodes_src["type"] = "Drug"

    if "target_type" in kg_edges.columns:
        nodes_tgt = kg_edges[["target", "target_type"]].rename(
            columns={"target": "id", "target_type": "type"}
        )
    else:
        nodes_tgt = kg_edges[["target", "relation"]].rename(columns={"target": "id"})
        nodes_tgt["type"] = nodes_tgt["relation"].apply(_relation_to_type)
        nodes_tgt = nodes_tgt.drop(columns=["relation"])

    kg_nodes = pd.concat([nodes_src, nodes_tgt]).drop_duplicates(subset=["id"])

    # Drop target_type before saving edges (optional; enrich_go may not need it)
    edges_out = kg_edges.drop(columns=["target_type"], errors="ignore")

    args.out_edges.parent.mkdir(parents=True, exist_ok=True)
    args.out_nodes.parent.mkdir(parents=True, exist_ok=True)
    edges_out.to_csv(args.out_edges, index=False)
    kg_nodes.to_csv(args.out_nodes, index=False)

    print("--- Summary ---")
    print(f"  Edges: {len(kg_edges)} -> {args.out_edges}")
    print(f"  Nodes: {len(kg_nodes)} -> {args.out_nodes}")
    print("Done.")


if __name__ == "__main__":
    main()
