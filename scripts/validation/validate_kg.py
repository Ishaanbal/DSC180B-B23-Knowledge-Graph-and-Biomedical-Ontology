"""
Validate knowledge graph integrity and data quality.

Usage:
    python scripts/validation/validate_kg.py --nodes data/kg_nodes_final.csv --edges data/kg_edges_final.csv
"""

import argparse
import pandas as pd
from pathlib import Path


def validate_kg(nodes_path: Path, edges_path: Path) -> dict:
    """Run all validation checks and return results."""
    
    print(f"Loading KG from:\n  Nodes: {nodes_path}\n  Edges: {edges_path}\n")
    
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    results = {
        'passed': [],
        'warnings': [],
        'errors': []
    }
    
    print("CHECK 1: Duplicate Edges")
    edge_cols = ['source', 'relation', 'target']
    duplicates = edges_df.duplicated(subset=edge_cols, keep=False)
    dup_count = duplicates.sum()
    
    if dup_count == 0:
        print("PASS: No duplicate edges found\n")
        results['passed'].append("No duplicate edges")
    else:
        print(f"ERROR: Found {dup_count} duplicate edges")
        print(edges_df[duplicates][edge_cols].head(10))
        print()
        results['errors'].append(f"{dup_count} duplicate edges")
    
    print("CHECK 2: Edge Endpoint Validity")
    node_ids = set(nodes_df['id'])
    
    invalid_sources = edges_df[~edges_df['source'].isin(node_ids)]
    invalid_targets = edges_df[~edges_df['target'].isin(node_ids)]
    
    if len(invalid_sources) == 0 and len(invalid_targets) == 0:
        print("PASS: All edge endpoints exist in nodes\n")
        results['passed'].append("All edge endpoints valid")
    else:
        if len(invalid_sources) > 0:
            print(f"ERROR: {len(invalid_sources)} edges with invalid sources")
            print(invalid_sources[['source', 'relation', 'target']].head(5))
            results['errors'].append(f"{len(invalid_sources)} invalid sources")
        if len(invalid_targets) > 0:
            print(f"ERROR: {len(invalid_targets)} edges with invalid targets")
            print(invalid_targets[['source', 'relation', 'target']].head(5))
            results['errors'].append(f"{len(invalid_targets)} invalid targets")
        print()
    
    print("CHECK 3: Node Type Distribution")
    node_types = nodes_df['type'].value_counts()
    print(node_types)
    
    protein_count = node_types.get('Protein', 0) + node_types.get('Gene/Protein', 0)
    if protein_count < 20:
        print(f"\nWARNING: Only {protein_count} protein nodes (recommend 40+)\n")
        results['warnings'].append(f"Low protein count: {protein_count}")
    else:
        print()
        results['passed'].append(f"Sufficient proteins: {protein_count}")
    
    print("CHECK 4: Edge Type Distribution")
    edge_types = edges_df['relation'].value_counts()
    print(edge_types)
    print()
    
    print("CHECK 5: Pralsetinib Connectivity")
    pral_edges = edges_df[edges_df['source'] == 'Pralsetinib']
    inhibits_edges = pral_edges[pral_edges['relation'] == 'inhibits']
    
    print(f"Total Pralsetinib edges: {len(pral_edges)}")
    print(f"Inhibits edges: {len(inhibits_edges)}")
    
    if len(inhibits_edges) < 10:
        print(f"WARNING: Only {len(inhibits_edges)} inhibits edges\n")
        results['warnings'].append(f"Low inhibits count: {len(inhibits_edges)}")
    else:
        print()
        results['passed'].append(f"Sufficient inhibits edges: {len(inhibits_edges)}")
    
    print("CHECK 6: Orphaned Nodes")
    nodes_in_edges = set(edges_df['source']).union(set(edges_df['target']))
    orphaned = set(node_ids) - nodes_in_edges
    
    if len(orphaned) == 0:
        print("PASS: No orphaned nodes\n")
        results['passed'].append("No orphaned nodes")
    else:
        print(f"WARNING: {len(orphaned)} nodes have no edges")
        print(f"Orphaned nodes: {list(orphaned)[:10]}")
        print()
        results['warnings'].append(f"{len(orphaned)} orphaned nodes")
    
    print("VALIDATION SUMMARY")
    print(f"Passed: {len(results['passed'])}")
    print(f"Warnings: {len(results['warnings'])}")
    print(f"Errors: {len(results['errors'])}")
    
    if results['errors']:
        print("\nVALIDATION FAILED")
        return results
    elif results['warnings']:
        print("\nVALIDATION PASSED WITH WARNINGS")
        return results
    else:
        print("\nVALIDATION PASSED")
        return results


def main():
    parser = argparse.ArgumentParser(description="Validate KG integrity")
    parser.add_argument('--nodes', default='data/kg_nodes_final.csv', help='Nodes CSV')
    parser.add_argument('--edges', default='data/kg_edges_final.csv', help='Edges CSV')
    args = parser.parse_args()
    
    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)
    
    if not nodes_path.exists():
        print(f"ERROR: Nodes file not found: {nodes_path}")
        return 1
    if not edges_path.exists():
        print(f"ERROR: Edges file not found: {edges_path}")
        return 1
    
    results = validate_kg(nodes_path, edges_path)
    
    if results['errors']:
        return 1
    return 0


if __name__ == '__main__':
    exit(main()) 