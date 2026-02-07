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
4. **Predict off-targets and outcomes** — GNN runs on the full graph (including Disease/Adverse Event nodes). It uses **two separate link-prediction heads**: one for **(Pralsetinib, inhibits, protein)** and one for **(protein, associated_with, Disease/AE)**. The outcome head is trained and used only for protein–outcome scoring, so **GNN-ranked Disease/AE** can vary by protein. Output: ranked proteins + score + **`gnn_predicted_outcomes`** (top-k Disease/AE per protein, protein-specific).
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

The GNN uses the **full graph** (Drug, Protein, Disease, Adverse Event, Pathway, etc.): message passing runs over all nodes. It is trained on two **separate** link-prediction tasks:

- **(drug, inhibits, protein)** — one MLP head; used for off-target ranking.
- **(protein, associated_with, Disease/AE)** — a **dedicated outcome MLP head** (separate from the drug–protein head), so outcome rankings can differ per protein.

```bash
python scripts/kg_gnn_link_prediction.py --nodes data/kg_nodes_final.csv --edges data/kg_edges_final.csv --out data/off_target_predictions_gnn.csv --epochs 200
```

**Output:**
- **`data/off_target_predictions_gnn.csv`** — columns: `rank`, `protein_id`, `score`, **`known_target`** (True if this protein already has a (Pralsetinib, inhibits, protein) edge in the KG; False otherwise), and **`gnn_predicted_outcomes`** (top-k Disease/AE from the outcome head, protein-specific). Use `known_target` to separate **KG-consistent known targets** from **candidate novel off-targets**.
- **`data/off_target_predictions_gnn_candidates.csv`** — top-k proteins with **no** inhibits edge in the KG, ranked by score (hypotheses for validation). Columns: `candidate_rank`, `protein_id`, `score`, and optionally `gnn_predicted_outcomes`. Written only if `--top-candidates > 0`.

**Optional args:** `--hidden 64 --embed 32 --top 100 --neg-per-pos 5 --save-model models/kg_gnn.pt`; `--outcome-weight 0.5` or `1.0` (weight for the protein–outcome loss; higher can improve outcome variation); `--top-outcomes 5` (number of Disease/AE per protein); **`--top-candidates 20`** (number of candidate novel off-targets to write to the _candidates CSV; use 0 to disable); `--no-outcome-task` to disable the (protein, outcome) task and the `gnn_predicted_outcomes` column.

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
| **How effects are obtained** | **Model prediction:** A dedicated outcome head scores every (protein, Disease/AE) pair and returns the top-k per protein. Learned from the graph; rankings can vary by protein; can list outcomes even when the KG has no direct edge. | **Lookup:** For each protein, the script finds all KG edges `(protein, associated_with, adverse_event)` and lists those targets. No model — only outcomes that already exist as edges in the KG; empty if there are none. |

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

## Next steps & concerns

**Current limitation — small protein set.** With the current KG we have only **15 candidate proteins** (Protein/Gene nodes) and **13 of them are known targets** (they already have a (Pralsetinib, inhibits, protein) edge). So the GNN is mostly recalling known edges and only **2 proteins** appear as candidate novel off-targets. The result therefore doesn’t yet tell us much we didn’t already encode in the graph.

**Expand the KG with more proteins.** To get more informative predictions:
- Add more **Protein/Gene nodes** to the graph (e.g. from a broader kinase panel, STRING/OpenTargets, or a curated list of plausible off-targets).
- Keep (Pralsetinib, inhibits, protein) edges only for **known** targets; leave the rest as “no edge” so the model can rank them. Then the **candidate novel off-targets** list (and `off_target_predictions_gnn_candidates.csv`) will be a larger, ranked set of hypotheses to validate.

**Other next steps.**  
- **Held-out evaluation:** Reserve some (Pralsetinib, inhibits, protein) edges for testing (don’t use them in training) to measure link-prediction performance.  
- **Richer outcome signal:** Add more (protein, associated_with, Disease/AE) edges so the outcome head has more signal and per-protein outcomes are more differentiated.  
- **Interpretation:** Use `known_target` and the _candidates file to separate “KG-consistent known targets” from “candidate novel off-targets” and focus validation on the latter.

---

## Repo layout

| Path | Purpose |
|------|--------|
| `data/` | KG CSVs, PubChem exports, GO/outcome mappings |
| `data_extraction.ipynb` | Build initial KG from PubChem |
| `scripts/enrich_go.py` | GO + target–outcome enrichment |
| `scripts/visualize_kg.py` | PyVis HTML export |
| `scripts/kg_gnn_*.py` | GNN data, model (two heads: drug–protein + protein–outcome), train/infer (predicts proteins + protein-specific GNN-ranked Disease/AE) |
| `scripts/add_effects_to_predictions.py` | Map predicted proteins → adverse effects via KG (ontology lookup) |
| `eda/eda.ipynb` | Exploratory analysis on final KG |
