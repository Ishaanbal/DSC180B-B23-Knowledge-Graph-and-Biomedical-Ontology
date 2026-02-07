# Pralsetinib Polypharmacology & Toxicity Knowledge Graph

[DSC180B-B23] **Neuro-symbolic, ontology-based off-target prediction** for **Pralsetinib** (PubChem [129073603](https://pubchem.ncbi.nlm.nih.gov/compound/129073603)), a recently approved RET kinase inhibitor.

## Project goal

Pralsetinib has limited long-term real-world safety data. Post-marketing pharmacovigilance already shows unexpected adverse effects (e.g., rhabdomyolysis, cognitive disorders, severe infections) beyond on-target effects. Because it is new, used in a small subset of patients, and has complex kinase biology, there is a high **“unknown off-target space”** — making it well-suited for an ontology-based off-target prediction study.

**Our approach (neuro-symbolic):** Build a knowledge graph that links mechanisms to toxicity (`Pralsetinib --[inhibits]--> Off-Target --[associated_with]--> Adverse Event`), enrich it with ontologies (GO, target–outcome). The **GNN** runs on the full graph (including Disease and Adverse Event nodes) and is trained to predict both **(drug, inhibits, protein)** and **(protein, associated_with, Disease/AE)**. The output is ranked off-target proteins plus **GNN-predicted Disease/AE** per protein; an optional **ontology lookup** adds KG-derived effects for comparison. The final deliverable is **predicted effects** — hypotheses for safety-relevant outcomes tied to specific off-targets.

---

## Pipeline overview

1. **Build KG** — Data extraction (bioassays, literature, clinical trials, co-occurrences) → `kg_nodes.csv`, `kg_edges.csv`.
2. **Enrich** — GO pathways + target–outcome links → `kg_nodes_final.csv`, `kg_edges_final.csv`.
3. **Visualize** — Interactive PyVis HTML.
4. **Predict off-targets and outcomes** — GNN runs on the full graph (including Disease/Adverse Event nodes). It is trained on two link-prediction tasks: **(Pralsetinib, inhibits, protein)** and **(protein, associated_with, Disease/AE)**. Output: ranked proteins + score + **GNN-ranked Disease/AE** per protein (`gnn_predicted_outcomes` column).
5. **Optional: ontology lookup** — `add_effects_to_predictions.py` adds KG-derived `associated_adverse_effects` (exact KG edges) for comparison or combined use.

---

## Running scripts

**Environment:** `conda env create -f environment.yml` (or `conda env update -f environment.yml`). Dependencies: Python 3.10, pandas, networkx, pytorch, torch-geometric (pip), pyvis, matplotlib, seaborn.

### Enrichment (GO + target–outcome)

```bash
python scripts/enrich_go.py --nodes data/kg_nodes_v2.csv --edges data/kg_edges_v2.csv --go-mapping data/protein_go_mapping.csv --outcome-mapping data/protein_outcome_mapping_onc.csv --out-nodes data/kg_nodes_final.csv --out-edges data/kg_edges_final.csv --add-similarity --drop-shortcuts
```

Optional auto-annotate GO mapping:

```bash
python scripts/enrich_go.py --generate-go-mapping --go-mapping data/protein_go_mapping.csv --outcome-mapping data/protein_outcome_mapping.csv
```

### Visualization

```bash
python scripts/visualize_kg.py --nodes data/kg_nodes_final.csv --edges data/kg_edges_final.csv --out figures/kg_interactive.html --max-nodes 300
```

Mechanistic-only view (no direct drug→disease/AE edges):

```bash
python scripts/visualize_kg.py --nodes data/kg_nodes_final.csv --edges data/kg_edges_final.csv --out figures/kg_interactive_mech.html --mechanistic-only --max-nodes 250
```

### GNN off-target prediction (train + infer)

The GNN uses the **full graph** (Drug, Protein, Disease, Adverse Event, Pathway, etc.): message passing runs over all nodes. It is trained on two tasks — **(drug, inhibits, protein)** and **(protein, associated_with, Disease/AE)** — so Disease/AE nodes are used both in the graph and as prediction targets.

```bash
python scripts/kg_gnn_link_prediction.py --nodes data/kg_nodes_final.csv --edges data/kg_edges_final.csv --out data/off_target_predictions_gnn.csv --epochs 200
```

**Output:** `data/off_target_predictions_gnn.csv` with columns `rank`, `protein_id`, `score`, and **`gnn_predicted_outcomes`** (top-k Disease/Adverse Event nodes the GNN scores as most associated with that protein, e.g. "Anemia | Hypertension | Neutropenia").

**Optional args:** `--hidden 64 --embed 32 --top 100 --neg-per-pos 5 --save-model models/kg_gnn.pt`; `--outcome-weight 0.5` (weight for the protein–outcome loss); `--top-outcomes 5` (number of Disease/AE per protein); `--no-outcome-task` to disable the (protein, outcome) task and the `gnn_predicted_outcomes` column.

### Map predictions to adverse effects (ontology lookup)

To add **KG-derived** effects (exact `associated_with` edges from the graph) alongside the GNN-ranked outcomes:

```bash
python scripts/add_effects_to_predictions.py --predictions data/off_target_predictions_gnn.csv --edges data/kg_edges_final.csv --nodes data/kg_nodes_final.csv --out data/off_target_predictions_with_effects.csv
```

Output: `data/off_target_predictions_with_effects.csv` adds `associated_adverse_effects` (ontology lookup). You can compare with `gnn_predicted_outcomes` from the GNN.

**Difference between the two prediction CSVs:**

| | `off_target_predictions_gnn.csv` | `off_target_predictions_with_effects.csv` |
|--|-----------------------------------|-------------------------------------------|
| **Produced by** | GNN script (`kg_gnn_link_prediction.py`) | Post-processing (`add_effects_to_predictions.py`) |
| **Effect column** | `gnn_predicted_outcomes` | `associated_adverse_effects` |
| **How effects are obtained** | **Model prediction:** GNN scores every (protein, Disease/AE) pair and returns the top-k. Learned from the graph; can list outcomes even when the KG has no direct edge. | **Lookup:** For each protein, the script finds all KG edges `(protein, associated_with, adverse_event)` and lists those targets. No model — only outcomes that already exist as edges in the KG; empty if there are none. |

Use the **GNN CSV** for model-predicted outcomes; use the **with_effects CSV** for ontology-derived effects and to compare with the GNN.

---

## Data files & sources

All input data are under `data/`. KG outputs: `kg_nodes.csv` / `kg_edges.csv` (initial), `kg_nodes_final.csv` / `kg_edges_final.csv` (after enrichment).

| File | Source | Role in pipeline |
|------|--------|-------------------|
| `pubchem_cid_129073603_bioactivity.csv` | PubChem / ChEMBL | IC50, Kd, Ki → **inhibits** edges (RET, off-targets). |
| `pubchem_cid_129073603_literature.csv` | PubChem | Abstracts → text mining for **associated_with** (target → AE). |
| `Chemical_Co-Occurrences-in-Literature_*.json` | PubChem | Chemical–chemical co-occurrence. |
| `Chemical_Gene-Co-Occurrences-in-Literature_*.json` | PubChem | Chemical–gene co-occurrence → latent off-targets (e.g. KDR, EGFR). |
| `pubchem_cid_129073603_clinicaltrials.csv` | PubChem / FDA | Trial conditions → **treats** edges. |
| `pubchem_cid_129073603_opentargetsdrugindication.csv` | OpenTargets | Indications → **treats**. |
| `pubchem_cid_129073603_consolidatedcompoundtarget.csv` | ChEMBL / TTD | Primary MoA → node typing (e.g. RET inhibitor). |
| `protein_go_mapping.csv` | GO / UniProt or manual | Protein → GO Biological Process → **involved_in** (Pathway nodes). |
| `protein_outcome_mapping.csv` / `protein_outcome_mapping_onc.csv` | CTD / OpenTargets | Protein → Disease/AE → **associated_with** edges. |

**Graph schema:** Nodes = Drug, Protein, Gene/Protein, Disease, Adverse Event, Pathway, Chemical, Gene. Edges = inhibits, treats, associated_with, involved_in, co_occurs_with_*, etc. **Key chains:** Hypertension ↔ KDR; Neutropenia ↔ JAK2/FLT3; Pneumonitis (text-mined).

---

## Repo layout

| Path | Purpose |
|------|--------|
| `data/` | KG CSVs, PubChem exports, GO/outcome mappings |
| `data_extraction.ipynb` | Build initial KG from PubChem |
| `scripts/enrich_go.py` | GO + target–outcome enrichment |
| `scripts/visualize_kg.py` | PyVis HTML export |
| `scripts/kg_gnn_*.py` | GNN data, model, train/infer (predicts proteins + GNN-ranked Disease/AE) |
| `scripts/add_effects_to_predictions.py` | Map predicted proteins → adverse effects via KG (ontology lookup) |
| `eda/eda.ipynb` | Exploratory analysis on final KG |
