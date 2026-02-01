"""Enrich KG with GO pathways + protein outcomes (CTD/OpenTargets)."""

import argparse
import io
import itertools
import gzip
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd
import networkx as nx

DRUG_NAME = "Pralsetinib"
SHORTCUT_TYPES = {"Adverse Event", "Disease"}

OVERRIDE_SYMBOLS = {
    "RET - ret proto-oncogene (human)": "RET",
    "KDR - kinase insert domain receptor (human)": "KDR",
    "Tyrosine-protein kinase JAK2 (human)": "JAK2",
    "FLT3 - fms related receptor tyrosine kinase 3 (human)": "FLT3",
    "KIF5B - kinesin family member 5B (human)": "KIF5B",
    "CCDC6 - coiled-coil domain containing 6 (human)": "CCDC6",
}

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Enrich the KG with GO-based pathways and curated "
            "target-outcome associations."
        )
    )
    parser.add_argument("--nodes", default="data/kg_nodes.csv")
    parser.add_argument("--edges", default="data/kg_edges.csv")
    parser.add_argument("--go-mapping", default="data/protein_go_mapping.csv")
    parser.add_argument(
        "--outcome-mapping", default="data/protein_outcome_mapping.csv"
    )
    parser.add_argument("--out-nodes", default="data/kg_nodes_enriched.csv")
    parser.add_argument("--out-edges", default="data/kg_edges_enriched.csv")
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix for output files (e.g. v3, go10).",
    )
    parser.add_argument("--add-similarity", action="store_true")
    parser.add_argument("--similarity-min-shared", type=int, default=1)
    parser.add_argument("--max-go-terms", type=int, default=10)
    parser.add_argument("--go-min-proteins", type=int, default=2)
    parser.add_argument(
        "--similarity-degree-threshold",
        type=int,
        default=1,
        help="Keep similar_to only if either endpoint has degree <= threshold.",
    )
    parser.add_argument(
        "--drop-shortcuts",
        action="store_true",
        help="Remove Drug->(Disease/Adverse Event) shortcut edges from outputs.",
    )
    parser.add_argument("--generate-go-mapping", action="store_true")
    parser.add_argument("--organism-id", default="9606")
    parser.add_argument("--import-ctd", action="store_true")
    parser.add_argument(
        "--ctd-file", default="data/ctd/CTD_genes_diseases.tsv.gz"
    )
    parser.add_argument("--ctd-outcome-type", default="Disease")
    parser.add_argument(
        "--ctd-limit",
        type=int,
        default=0,
        help="Optional max lines to process from CTD (0 = no limit).",
    )
    parser.add_argument(
        "--ctd-stop-when-found",
        action="store_true",
        help="Stop once all target gene symbols are found in CTD.",
    )
    parser.add_argument(
        "--ctd-include-keywords",
        default="cancer,carcinoma,tumor,neoplasm,leukemia,lymphoma,adenocarcinoma,"
        "metastasis,thyroid,lung,nsclc,hypertension,neutropenia,pneumonitis",
        help="Comma-separated keywords to keep CTD diseases.",
    )
    parser.add_argument(
        "--ctd-exclude-keywords",
        default="mouse,rat,zebrafish",
        help="Comma-separated keywords to drop CTD diseases.",
    )
    return parser.parse_args()


def load_mapping(mapping_path):
    mapping_df = pd.read_csv(mapping_path)
    expected = {"protein_id", "go_id", "go_name"}
    missing = expected - set(mapping_df.columns)
    if missing:
        raise ValueError(
            f"Mapping file missing columns: {', '.join(sorted(missing))}"
        )
    mapping_df = mapping_df[list(expected)].copy()
    return mapping_df


def extract_gene_symbol(protein_id):
    cleaned = protein_id.replace("(human)", "").strip()
    if " - " in cleaned:
        return cleaned.split(" - ")[0].strip()
    tokens = [t.strip(",") for t in cleaned.split() if t.strip(",")]
    for token in reversed(tokens):
        if token.isalnum() and token.upper() == token and len(token) <= 10:
            return token
    return tokens[-1] if tokens else cleaned


def parse_go_terms(go_value):
    terms = []
    raw_entries = [e.strip() for e in go_value.split("|") if e.strip()]
    if len(raw_entries) == 1:
        raw_entries = [e.strip() for e in go_value.split(";") if e.strip()]
    for entry in raw_entries:
        if "[GO:" in entry:
            name_part, go_part = entry.rsplit("[GO:", 1)
            go_id = f"GO:{go_part.split(']')[0].strip()}"
            go_name = name_part.strip()
            if go_id and go_name:
                terms.append((go_id, go_name))
            continue
        parts = [p.strip() for p in entry.split(";")]
        go_id = next((p for p in parts if p.startswith("GO:")), "")
        go_name = ""
        for part in parts:
            if part.startswith("P:"):
                go_name = part[2:].strip()
                break
        if go_id:
            terms.append((go_id, go_name))
    return terms


def fetch_go_terms_for_query(query, organism_id):
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "fields": "accession,gene_primary,protein_name,go_p",
        "format": "tsv",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    print(f"[GO] Query URL: {url}")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (KG-Enrichment/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        body = response.read().decode("utf-8")
    preview = "\n".join(body.splitlines()[:5])
    print(f"[GO] Response preview:\n{preview}")
    df = pd.read_csv(io.StringIO(body), sep="\t")
    if df.empty:
        return []
    go_col = next(
        (col for col in df.columns if col.lower().startswith("gene ontology (biological process)")),
        None,
    )
    if go_col is None:
        return []
    for _, row in df.iterrows():
        go_value = str(row[go_col])
        if not go_value or go_value.lower() == "nan":
            continue
        terms = parse_go_terms(go_value)
        if terms:
            return terms
    return []


def fetch_go_terms(gene_symbol, protein_name, organism_id):
    queries = []
    if gene_symbol:
        gene_symbol = gene_symbol.upper()
        queries.append(f"gene_exact:{gene_symbol} AND organism_id:{organism_id}")
        queries.append(f"gene:{gene_symbol} AND organism_id:{organism_id}")
    if protein_name:
        queries.append(
            f"protein_name:\"{protein_name}\" AND organism_id:{organism_id}"
        )
    for query in queries:
        terms = fetch_go_terms_for_query(query, organism_id)
        if terms:
            return terms
    return []


def generate_go_mapping( protein_ids, output_path, max_go_terms, organism_id,):
    print(f"Generating GO mapping for {len(protein_ids)} proteins...")
    existing_rows = []
    existing_pairs = set()
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        if {"protein_id", "go_id", "go_name"}.issubset(existing_df.columns):
            existing_rows = existing_df[
                ["protein_id", "go_id", "go_name"]
            ].to_dict("records")
            existing_pairs = {
                (row["protein_id"], row["go_id"]) for row in existing_rows
            }

    rows = list(existing_rows)
    proteins_missing = []
    added_count = 0
    for protein_id in protein_ids:
        gene_symbol = OVERRIDE_SYMBOLS.get(
            protein_id, extract_gene_symbol(protein_id)
        )
        protein_name = ""
        if " - " in protein_id:
            protein_name = protein_id.split(" - ", 1)[1]
        protein_name = protein_name.replace("(human)", "").strip()
        try:
            go_terms = fetch_go_terms(gene_symbol, protein_name, organism_id)
        except Exception:
            proteins_missing.append(protein_id)
            print(f"[GO] Lookup failed for {protein_id}")
            continue
        if max_go_terms and max_go_terms > 0:
            go_terms = go_terms[:max_go_terms]
        if not go_terms:
            proteins_missing.append(protein_id)
            print(f"[GO] No terms found for {protein_id}")
            continue
        for go_id, go_name in go_terms:
            key = (protein_id, go_id)
            if key in existing_pairs:
                continue
            rows.append(
                {
                    "protein_id": protein_id,
                    "go_id": go_id,
                    "go_name": go_name,
                }
            )
            existing_pairs.add(key)
            added_count += 1
        print(f"[GO] Added {len(go_terms)} terms for {protein_id}")
    mapping_df = pd.DataFrame(rows)
    mapping_df.to_csv(output_path, index=False)
    print(
        f"GO mapping saved: {len(mapping_df)} rows total "
        f"({added_count} new)"
    )
    if proteins_missing:
        print(
            "GO annotations missing for:",
            ", ".join(sorted(set(proteins_missing))),
        )


def load_outcome_mapping(mapping_path):
    mapping_df = pd.read_csv(mapping_path)
    expected = {
        "protein_id",
        "outcome_id",
        "outcome_type",
        "outcome_name",
        "evidence",
        "source",
    }
    missing = expected - set(mapping_df.columns)
    if missing:
        raise ValueError(
            f"Outcome mapping missing columns: {', '.join(sorted(missing))}"
        )
    mapping_df = mapping_df[list(expected)].copy()
    mapping_df["outcome_type"] = mapping_df["outcome_type"].str.strip()
    return mapping_df


def add_pathway_nodes(nodes_df, mapping_df):
    existing_ids = set(nodes_df["id"])
    pathway_ids = (
        mapping_df["go_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    new_pathways = [pid for pid in pathway_ids if pid not in existing_ids]
    if not new_pathways:
        return nodes_df
    new_nodes_df = pd.DataFrame({"id": new_pathways, "type": "Pathway"})
    return pd.concat([nodes_df, new_nodes_df], ignore_index=True)


def build_involved_in_edges(mapping_df):
    involved = mapping_df.dropna(subset=["protein_id", "go_id"])
    if involved.empty:
        return pd.DataFrame()
    edges = pd.DataFrame(
        {
            "source": involved["protein_id"].astype(str),
            "relation": "involved_in",
            "target": involved["go_id"].astype(str),
            "value": "",
            "unit": "",
            "metadata": involved["go_name"].fillna("").astype(str),
            "source_file": "GO",
        }
    )
    return edges.drop_duplicates(
        subset=["source", "relation", "target", "metadata", "source_file"]
    )


def build_similarity_edges(mapping_df, min_shared):
    if mapping_df.empty:
        return pd.DataFrame()
    protein_to_go = (
        mapping_df.dropna(subset=["protein_id", "go_id"])
        .groupby("protein_id")["go_id"]
        .apply(lambda series: set(series.astype(str)))
    )
    if protein_to_go.empty:
        return pd.DataFrame()
    go_to_proteins = defaultdict(list)
    for protein, go_terms in protein_to_go.items():
        for go_id in go_terms:
            go_to_proteins[go_id].append(protein)

    pair_counts = defaultdict(int)
    for proteins in go_to_proteins.values():
        unique = sorted(set(proteins))
        for p1, p2 in itertools.combinations(unique, 2):
            pair_counts[(p1, p2)] += 1

    rows = []
    for (p1, p2), count in pair_counts.items():
        if count < min_shared:
            continue
        metadata = f"shared_go_count={count}"
        rows.append(
            {
                "source": p1,
                "relation": "similar_to",
                "target": p2,
                "value": "",
                "unit": "",
                "metadata": metadata,
                "source_file": "GO",
            }
        )
        rows.append(
            {
                "source": p2,
                "relation": "similar_to",
                "target": p1,
                "value": "",
                "unit": "",
                "metadata": metadata,
                "source_file": "GO",
            }
        )

    if not rows:
        return pd.DataFrame()
    similarity_df = pd.DataFrame(rows)
    return similarity_df.drop_duplicates(
        subset=["source", "relation", "target", "metadata", "source_file"]
    )


def filter_similarity_edges(similarity_df, degree_dict, threshold):
    if similarity_df.empty:
        return similarity_df
    keep_mask = []
    for _, row in similarity_df.iterrows():
        src = row["source"]
        tgt = row["target"]
        if degree_dict.get(src, 0) <= threshold or degree_dict.get(tgt, 0) <= threshold:
            keep_mask.append(True)
        else:
            keep_mask.append(False)
    return similarity_df[keep_mask]


def add_outcome_nodes(nodes_df, mapping_df):
    existing_ids = set(nodes_df["id"].astype(str))
    new_rows = []
    for _, row in mapping_df.iterrows():
        outcome_id = str(row["outcome_id"])
        if outcome_id in existing_ids:
            continue
        new_rows.append({"id": outcome_id, "type": row["outcome_type"]})
        existing_ids.add(outcome_id)
    if not new_rows:
        return nodes_df
    new_nodes_df = pd.DataFrame(new_rows)
    return pd.concat([nodes_df, new_nodes_df], ignore_index=True)


def build_association_edges(mapping_df):
    mapping_df = mapping_df.dropna(subset=["protein_id", "outcome_id"])
    if mapping_df.empty:
        return pd.DataFrame()
    edges = pd.DataFrame(
        {
            "source": mapping_df["protein_id"].astype(str),
            "relation": "associated_with",
            "target": mapping_df["outcome_id"].astype(str),
            "value": "",
            "unit": "",
            "metadata": mapping_df["evidence"].fillna("").astype(str),
            "source_file": mapping_df["source"].fillna("").astype(str),
        }
    )
    return edges.drop_duplicates(
        subset=["source", "relation", "target", "metadata", "source_file"]
    )


def import_ctd_gene_disease(
    ctd_path,
    nodes_df,
    outcome_mapping_path,
    outcome_type,
    ctd_limit,
    stop_when_found,
    include_keywords,
    exclude_keywords,
):
    if not ctd_path.exists():
        print(f"[CTD] File not found: {ctd_path}")
        return

    protein_nodes = nodes_df[nodes_df["type"] == "Protein"]["id"].astype(str)
    protein_symbol_map = defaultdict(list)
    for protein_id in protein_nodes:
        symbol = OVERRIDE_SYMBOLS.get(protein_id, extract_gene_symbol(protein_id))
        protein_symbol_map[symbol.upper()].append(protein_id)
    remaining_symbols = set(protein_symbol_map.keys())

    include_set = {k.strip().lower() for k in include_keywords.split(",") if k.strip()}
    exclude_set = {k.strip().lower() for k in exclude_keywords.split(",") if k.strip()}
    opener = gzip.open if ctd_path.suffix == ".gz" else open
    rows = []
    line_count = 0
    with opener(ctd_path, "rt", encoding="utf-8") as handle:
        header = None
        for line in handle:
            line_count += 1
            if line_count % 1_000_000 == 0:
                print(f"[CTD] Processed {line_count:,} lines...")
            if stop_when_found and not remaining_symbols:
                print("[CTD] All target gene symbols found. Stopping early.")
                break
            if ctd_limit and line_count > ctd_limit:
                print(f"[CTD] Reached limit of {ctd_limit:,} lines.")
                break
            if line.startswith("#"):
                if header is None and "GeneSymbol" in line:
                    header = [h.strip() for h in line.lstrip("#").strip().split("\t")]
                    print(f"[CTD] Parsed header with {len(header)} columns.")
                continue
            line = line.rstrip("\n")
            if not line:
                continue
            if header is None:
                header = [h.strip() for h in line.split("\t")]
                print(f"[CTD] Parsed header with {len(header)} columns.")
                continue
            parts = line.split("\t")
            if header and len(parts) < len(header):
                continue
            record = dict(zip(header, parts))
            gene_symbol = (
                record.get("GeneSymbol")
                or record.get("Gene Symbol")
                or ""
            ).upper()
            if gene_symbol not in protein_symbol_map:
                continue
            remaining_symbols.discard(gene_symbol)
            disease_name = (
                record.get("DiseaseName")
                or record.get("Disease Name")
                or ""
            ).strip()
            if not disease_name:
                continue
            disease_name_norm = disease_name.lower()
            if exclude_set and any(k in disease_name_norm for k in exclude_set):
                continue
            if include_set and not any(k in disease_name_norm for k in include_set):
                continue
            disease_id = (
                record.get("DiseaseID")
                or record.get("Disease ID")
                or ""
            ).strip()
            evidence = (
                record.get("DirectEvidence")
                or record.get("Direct Evidence")
                or ""
            ).strip()
            if disease_id:
                evidence = f"{evidence};{disease_id}".strip(";")

            for protein_id in protein_symbol_map[gene_symbol]:
                rows.append(
                    {
                        "protein_id": protein_id,
                        "outcome_id": disease_name,
                        "outcome_type": outcome_type,
                        "outcome_name": disease_name,
                        "evidence": evidence,
                        "source": "CTD",
                    }
                )

    if not rows:
        print("[CTD] No mappings matched current Protein IDs.")
        return

    new_df = pd.DataFrame(rows)
    new_df["outcome_name_norm"] = (
        new_df["outcome_name"]
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    existing_count = 0
    if outcome_mapping_path.exists():
        existing_df = pd.read_csv(outcome_mapping_path)
        existing_count = len(existing_df)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["protein_id", "outcome_id", "source"]
        )
    else:
        combined = new_df
    if "outcome_name_norm" in combined.columns:
        combined = combined.drop_duplicates(
            subset=["protein_id", "outcome_name_norm", "source"]
        ).drop(columns=["outcome_name_norm"])
    combined.to_csv(outcome_mapping_path, index=False)
    added_rows = len(combined) - existing_count
    print(
        f"[CTD] Added {added_rows} rows to {outcome_mapping_path} "
        f"(matched {len(new_df)} raw rows)"
    )


def main():
    args = parse_args()
    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)
    go_mapping_path = Path(args.go_mapping)
    outcome_mapping_path = Path(args.outcome_mapping)
    out_nodes_path = Path(args.out_nodes)
    out_edges_path = Path(args.out_edges)
    if args.tag:
        out_nodes_path = out_nodes_path.with_name(
            f"{out_nodes_path.stem}_{args.tag}{out_nodes_path.suffix}"
        )
        out_edges_path = out_edges_path.with_name(
            f"{out_edges_path.stem}_{args.tag}{out_edges_path.suffix}"
        )

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    nodes_df["id"] = nodes_df["id"].astype(str)
    protein_ids = set(nodes_df[nodes_df["type"] == "Protein"]["id"].astype(str))

    if args.import_ctd:
        import_ctd_gene_disease(
            Path(args.ctd_file),
            nodes_df,
            outcome_mapping_path,
            args.ctd_outcome_type,
            args.ctd_limit,
            args.ctd_stop_when_found,
            args.ctd_include_keywords,
            args.ctd_exclude_keywords,
        )

    if args.generate_go_mapping:
        generate_go_mapping(
            sorted(protein_ids),
            go_mapping_path,
            args.max_go_terms,
            args.organism_id,
        )

    outcome_mapping_df = load_outcome_mapping(outcome_mapping_path)
    outcome_mapping_df["protein_id"] = outcome_mapping_df["protein_id"].astype(
        str
    )

    go_mapping_df = load_mapping(go_mapping_path)
    go_mapping_df["protein_id"] = go_mapping_df["protein_id"].astype(str)
    go_mapping_df = go_mapping_df[
        go_mapping_df["protein_id"].isin(protein_ids)
    ].copy()
    outcome_mapping_df = outcome_mapping_df[
        outcome_mapping_df["protein_id"].isin(protein_ids)
    ].copy()

    if args.max_go_terms and args.max_go_terms > 0:
        go_mapping_df = (
            go_mapping_df.groupby("protein_id", sort=False)
            .head(args.max_go_terms)
            .reset_index(drop=True)
        )

    if args.go_min_proteins and args.go_min_proteins > 1:
        go_counts = (
            go_mapping_df.groupby("go_id")["protein_id"]
            .nunique()
            .sort_values(ascending=False)
        )
        keep_go = set(go_counts[go_counts >= args.go_min_proteins].index)
        go_mapping_df = go_mapping_df[go_mapping_df["go_id"].isin(keep_go)].copy()

    nodes_enriched = add_pathway_nodes(nodes_df, go_mapping_df)
    nodes_enriched = add_outcome_nodes(nodes_enriched, outcome_mapping_df)
    involved_edges = build_involved_in_edges(go_mapping_df)
    association_edges = build_association_edges(outcome_mapping_df)

    new_edges = [involved_edges, association_edges]
    if args.add_similarity:
        similarity_edges = build_similarity_edges(
            go_mapping_df, args.similarity_min_shared
        )
        base_edges = pd.concat(
            [edges_df, involved_edges, association_edges],
            ignore_index=True,
        )
        base_graph = nx.from_pandas_edgelist(
            base_edges,
            source="source",
            target="target",
            create_using=nx.DiGraph(),
        )
        base_degree = dict(base_graph.degree())
        similarity_edges = filter_similarity_edges(
            similarity_edges, base_degree, args.similarity_degree_threshold
        )
        new_edges.append(similarity_edges)

    new_edges_df = pd.concat(
        [df for df in new_edges if not df.empty], ignore_index=True
    )

    if new_edges_df.empty:
        edges_enriched = edges_df.copy()
    else:
        edges_enriched = pd.concat([edges_df, new_edges_df], ignore_index=True)

    if args.drop_shortcuts:
        node_type_map = nodes_enriched.set_index("id")["type"].to_dict()
        is_shortcut = (
            (edges_enriched["source"] == DRUG_NAME)
            & (edges_enriched["target"].map(node_type_map).isin(SHORTCUT_TYPES))
        )
        shortcut_count = int(is_shortcut.sum())
        edges_enriched = edges_enriched[~is_shortcut].copy()
    else:
        shortcut_count = 0

    nodes_enriched.to_csv(out_nodes_path, index=False)
    edges_enriched.to_csv(out_edges_path, index=False)

    print("KG enrichment complete.")
    print(f"Nodes: {len(nodes_df)} -> {len(nodes_enriched)}")
    print(f"Edges: {len(edges_df)} -> {len(edges_enriched)}")
    print(f"New involved_in edges: {len(involved_edges)}")
    print(f"New associated_with edges: {len(association_edges)}")
    if args.drop_shortcuts:
        print(f"Removed shortcut edges: {shortcut_count}")
    print(f"Output nodes file: {out_nodes_path}")
    print(f"Output edges file: {out_edges_path}")
    if args.add_similarity:
        similarity_count = (
            len(new_edges_df[new_edges_df["relation"] == "similar_to"])
            if not new_edges_df.empty
            else 0
        )
        print(f"New similar_to edges: {similarity_count}")


if __name__ == "__main__":
    main()
