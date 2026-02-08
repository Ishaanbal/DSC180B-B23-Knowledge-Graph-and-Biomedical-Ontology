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
4. **Predict off-targets and outcomes** — GNN runs on the full graph; outputs **`predictions/off_target_predictions_gnn.csv`** (intermediate). Then **`build_off_target_predictions.py`** adds KG-derived effects and **path-based chain-of-thought reasoning** (actual KG paths) → single canonical **`predictions/off_target_predictions.csv`**.

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
python scripts/kg_gnn_link_prediction.py --nodes data/kg_nodes_final.csv --edges data/kg_edges_final.csv --out predictions/off_target_predictions_gnn.csv --epochs 200
```

**Output (intermediate):**
- **`predictions/off_target_predictions_gnn.csv`** — raw GNN output: `rank`, `protein_id`, `score`, `known_target`, `gnn_predicted_outcomes`. Input to the build script below.
- **`predictions/off_target_predictions_candidates.csv`** — proteins with **no** (Pralsetinib, inhibits, protein) edge in the KG, ranked by score (hypotheses for validation). Written only if `--top-candidates > 0`.

**Optional args:** `--hidden 64 --embed 32 --top 100 --neg-per-pos 5 --save-model models/kg_gnn.pt`; `--outcome-weight 0.5` or `1.0`; `--top-outcomes 5`; **`--top-candidates 20`** (use 0 to disable); `--no-outcome-task`.

### Build final predictions (effects + path-based reasoning)

One script adds KG-derived effects and **path-based chain-of-thought** (explicit KG paths) so predictions are non-redundant and auditable:

```bash
python scripts/build_off_target_predictions.py --predictions predictions/off_target_predictions_gnn.csv --edges data/kg_edges_final.csv --nodes data/kg_nodes_final.csv --out predictions/off_target_predictions.csv
```

**Output: `predictions/off_target_predictions.csv`** (canonical file). Columns: `rank`, `protein_id`, `score`, `known_target`, `gnn_predicted_outcomes`, `associated_adverse_effects`, **`reasoning`**.

The **`reasoning`** column is path-based on the KG:
- **Path 1:** `Pralsetinib --[inhibits]--> protein_id` (with evidence from KG, e.g. IC50), or “No edge in KG; GNN predicts link (score=…)” for novel predictions.
- **Path 2:** `protein_id --[associated_with]--> outcome1 | outcome2 | …` (actual KG edges to Disease/AE), or “No edges in KG; GNN top predicted: …” when the KG has no (protein, outcome) edges.

This keeps a single predictions file and makes the reasoning explicit and path-based for safety/validation.

---

## Data and predictions layout

- **`data/`** — Inputs used to build the KG: PubChem exports, GO/outcome mappings, and the KG itself (`kg_nodes*.csv`, `kg_edges*.csv`). No model outputs live here.
- **`predictions/`** — Model outputs only: `off_target_predictions_gnn.csv` (raw GNN), `off_target_predictions.csv` (canonical, with effects and reasoning), and `off_target_predictions_candidates.csv` (novel off-target candidates).

---

## Data files & sources

All KG input data are under `data/`. KG outputs: `kg_nodes.csv` / `kg_edges.csv` (initial), `kg_nodes_final.csv` / `kg_edges_final.csv` (after enrichment).

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
- Keep (Pralsetinib, inhibits, protein) edges only for **known** targets; leave the rest as “no edge” so the model can rank them. Then **`predictions/off_target_predictions_candidates.csv`** will be a larger, ranked set of hypotheses to validate.

**Other next steps.**  
- **Held-out evaluation:** Reserve some (Pralsetinib, inhibits, protein) edges for testing (don’t use them in training) to measure link-prediction performance.  
- **Richer outcome signal:** Add more (protein, associated_with, Disease/AE) edges so the outcome head has more signal and per-protein outcomes are more differentiated.  
- **Interpretation:** Use `known_target` and `predictions/off_target_predictions_candidates.csv` to separate “KG-consistent known targets” from “candidate novel off-targets” and focus validation on the latter.

---

## Repo layout

| Path | Purpose |
|------|--------|
| `data/` | KG inputs: KG CSVs, PubChem exports, GO/outcome mappings (no prediction outputs) |
| `predictions/` | Model outputs: off_target_predictions*.csv (GNN results, canonical file, candidates) |
| `data_extraction.ipynb` | Build initial KG from PubChem |
| `scripts/enrich_go.py` | GO + target–outcome enrichment |
| `scripts/visualize_kg.py` | PyVis HTML export |
| `scripts/kg_gnn_*.py` | GNN data, model (two heads: drug–protein + protein–outcome), train/infer |
| `scripts/build_off_target_predictions.py` | Add KG effects + path-based reasoning → canonical `predictions/off_target_predictions.csv` |
| `eda/eda.ipynb` | Exploratory analysis on final KG |
