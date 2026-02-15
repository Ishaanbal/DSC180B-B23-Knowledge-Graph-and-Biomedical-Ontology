# Pralsetinib Polypharmacology & Toxicity Knowledge Graph

[DSC180B-B23] **Neuro-symbolic, ontology-based off-target prediction** for **Pralsetinib** (PubChem [129073603](https://pubchem.ncbi.nlm.nih.gov/compound/129073603)), a recently approved RET kinase inhibitor.

## Project goal

Pralsetinib has limited long-term real-world safety data. Post-marketing pharmacovigilance already shows unexpected adverse effects (e.g., rhabdomyolysis, cognitive disorders, severe infections) beyond on-target effects. Because it is new, used in a small subset of patients, and has complex kinase biology, there is a high **“unknown off-target space”** — making it well-suited for an ontology-based off-target prediction study.

**Our approach (neuro-symbolic):** Build a knowledge graph that links mechanisms to toxicity (`Pralsetinib --[inhibits]--> Off-Target --[associated_with]--> Adverse Event`), enrich it with ontologies (GO, target–outcome). The **GNN** runs on the full graph (including Disease and Adverse Event nodes) and is trained to predict both **(drug, inhibits, protein)** and **(protein, associated_with, Disease/AE)**. The output is ranked off-target proteins plus **GNN-predicted Disease/AE** per protein; an optional **ontology lookup** adds KG-derived effects for comparison. The final deliverable is **predicted effects** — hypotheses for safety-relevant outcomes tied to specific off-targets.

---

## Pipeline overview

1. **Build KG** — Run `scripts/kg_construction/build_kg_from_sources.py` to extract from PubChem (bioactivity, targets, clinical trials, indications, co-occurrence JSONs, literature AEs) → `data/kg_nodes_v2.csv`, `data/kg_edges_v2.csv`.
2. **Enrich + expand** — GO pathways + target–outcome links **and** expansion with additional proteins from STRING (PPI network) and UniProt (human kinases) → `data/kg_nodes_final.csv`, `data/kg_edges_final.csv`.
3. **Visualize** — Interactive PyVis HTML.
4. **Predict off-targets and outcomes** — GNN runs on the full graph; outputs **`predictions/off_target_predictions_gnn.csv`** (intermediate). Then **`build_off_target_predictions.py`** adds KG-derived effects and **path-based chain-of-thought reasoning** (actual KG paths) → single canonical **`predictions/off_target_predictions.csv`**.
5. **Baseline comparison** — Simple KG baseline model (sums edge weights) for comparison with GNN.

---

## Running scripts

**Environment:** `conda env create -f environment.yml` (or `conda env update -f environment.yml`).

### Reproduce results (single entry point)

From the project root, use **`run.py`** with a keyword to run one step or the full pipeline:

```bash
python run.py all              # Full pipeline: build_kg → enrich (GO+expand) → visualize → train → predict
python run.py all --no-viz     # Same but skip visualization
python run.py build_kg         # Build initial KG only
python run.py enrich           # Enrich KG (GO + outcomes) and expand with proteins
python run.py visualize        # Export PyVis HTML only
python run.py train            # Train GNN and write raw predictions
python run.py predict          # Add effects + reasoning → canonical predictions CSV
```

If you're not using the conda environment, install `pyvis` for the visualize step: `pip install pyvis` (or use `python run.py all --no-viz` to skip it).

Pass extra arguments to the underlying script after the keyword, e.g.:

```bash
python run.py train --epochs 100 --top-candidates 20
```

For full control (custom paths, options), run the individual scripts under `scripts/` as in the sections below.

### Build initial KG from sources

Extract edges and nodes from PubChem and other data under `data/`:

```bash
python scripts/kg_construction/build_kg_from_sources.py --data-dir data --out-edges data/kg_edges_v2.csv --out-nodes data/kg_nodes_v2.csv
```

**Output:** `data/kg_edges_v2.csv`, `data/kg_nodes_v2.csv` (inputs for enrichment below). Sources: bioactivity, consolidated targets, clinical trials, OpenTargets indications, chemical/gene co-occurrence JSONs, literature adverse-event mining.

### Enrichment (GO + target–outcome + expand proteins)

```bash
python scripts/kg_construction/enrich_and_expand_kg.py \
    --nodes data/kg_nodes_v2.csv \
    --edges data/kg_edges_v2.csv \
    --go-mapping data/protein_go_mapping.csv \
    --outcome-mapping data/protein_outcome_mapping_onc.csv \
    --out-nodes data/kg_nodes_final.csv \
    --out-edges data/kg_edges_final.csv \
    --add-similarity --drop-shortcuts \
    --drug Pralsetinib \
    --max-proteins 500
```

Output: `data/kg_nodes_final.csv`, `data/kg_edges_final.csv`. Use `enrich_go.py` directly for more fine-grained control.

### Visualization

```bash
python scripts/viz/visualize_kg.py --nodes data/kg_nodes_final.csv --edges data/kg_edges_final.csv --out figures/kg_interactive.html --max-nodes 300
```

Mechanistic-only view (no direct drug→disease/AE edges):

```bash
python scripts/viz/visualize_kg.py --nodes data/kg_nodes_final.csv --edges data/kg_edges_final.csv --out figures/kg_interactive_mech.html --mechanistic-only --max-nodes 250
```

### GNN off-target prediction (train + infer)

```bash
python scripts/modeling/gnn/kg_gnn_link_prediction.py --nodes data/kg_nodes_final.csv --edges data/kg_edges_final.csv --out predictions/off_target_predictions_gnn.csv --epochs 200
```

**Output (intermediate):**
- **`predictions/off_target_predictions_gnn.csv`** — raw GNN output: `rank`, `protein_id`, `score`, `known_target`, `gnn_predicted_outcomes`. Input to the build script below.
- **`predictions/off_target_predictions_candidates.csv`** — proteins with **no** (Pralsetinib, inhibits, protein) edge in the KG, ranked by score (hypotheses for validation). Written only if `--top-candidates > 0`.

**Optional args:** `--hidden 64 --embed 32 --top 100 --neg-per-pos 5 --save-model models/kg_gnn.pt`; `--outcome-weight 0.5` or `1.0`; `--top-outcomes 5`; **`--top-candidates 20`** (use 0 to disable); `--no-outcome-task`.

### Build final predictions (effects + path-based reasoning)

```bash
python scripts/modeling/gnn/build_off_target_predictions.py --predictions predictions/off_target_predictions_gnn.csv --edges data/kg_edges_final.csv --nodes data/kg_nodes_final.csv --out predictions/off_target_predictions.csv
```

**Output: `predictions/off_target_predictions.csv`** (canonical file). Columns include **`reasoning`**, which is path-based on the KG:
- **Path 1:** `Pralsetinib --[inhibits]--> protein_id` (with evidence from KG, e.g. IC50), or “No edge in KG; GNN predicts link (score=…)” for novel predictions.
- **Path 2:** `protein_id --[associated_with]--> outcome1 | outcome2 | …` (actual KG edges to Disease/AE), or “No edges in KG; GNN top predicted: …” when the KG has no (protein, outcome) edges.

### Baseline KG model (for comparison)

Sums the absolute `value` from `(drug, inhibits, protein)` edges in the KG (no ML). For comparison with the GNN.

```bash
python scripts/modeling/baseline/kg_baseline_link_prediction.py \
    --nodes data/kg_nodes_final.csv \
    --edges data/kg_edges_final.csv \
    --out predictions/off_target_predictions_baseline.csv \
    --top 100
```

**Output:** `predictions/off_target_predictions_baseline.csv` — `rank`, `protein_id`, `score`, `known_target`.

---

## Data and predictions layout

- **`data/`** — KG inputs (PubChem exports, GO/outcome mappings, `kg_nodes*.csv`, `kg_edges*.csv`).
- **`predictions/`** — Model outputs only.

**Difference between prediction CSVs:**

| File | Produced by | Rows | Purpose |
|------|-------------|------|---------|
| **`predictions/off_target_predictions_gnn.csv`** | GNN script | Top-k proteins (all) | Raw GNN output; input to build script. Columns: rank, protein_id, score, known_target, gnn_predicted_outcomes. |
| **`predictions/off_target_predictions.csv`** | Build script | Same as _gnn | **Canonical file** — use this for analysis. Adds `associated_adverse_effects` and path-based `reasoning`. |
| **`predictions/off_target_predictions_baseline.csv`** | Baseline script | Top-k proteins | Simple KG baseline (sums edge weights) for comparison with GNN. Columns: rank, protein_id, score, known_target. |
| **`predictions/off_target_predictions_candidates.csv`** | GNN script | Only proteins with *no* (Pralsetinib, inhibits, protein) edge in KG | Convenience list of **novel off-target candidates** to validate. Redundant with filtering the canonical file on `known_target == False`. |

---

## Data files & sources

KG outputs: `data/kg_nodes_v2.csv` / `data/kg_edges_v2.csv` (from build), then `data/kg_nodes_final.csv` / `data/kg_edges_final.csv` (after enrich).

| File | Source | Role in pipeline |
|------|--------|-------------------|
| `data/pubchem_cid_129073603_bioactivity.csv` | PubChem / ChEMBL | IC50, Kd, Ki → **inhibits** edges (RET, off-targets). |
| `data/pubchem_cid_129073603_literature.csv` | PubChem | Abstracts → text mining for **associated_with** (target → AE). |
| `data/Chemical_Co-Occurrences-in-Literature_*.json` | PubChem | Chemical–chemical co-occurrence. |
| `data/Chemical_Gene-Co-Occurrences-in-Literature_*.json` | PubChem | Chemical–gene co-occurrence → latent off-targets (e.g. KDR, EGFR). |
| `data/pubchem_cid_129073603_clinicaltrials.csv` | PubChem / FDA | Trial conditions → **treats** edges. |
| `data/pubchem_cid_129073603_opentargetsdrugindication.csv` | OpenTargets | Indications → **treats**. |
| `data/pubchem_cid_129073603_consolidatedcompoundtarget.csv` | ChEMBL / TTD | Primary MoA → node typing (e.g. RET inhibitor). |
| `data/protein_go_mapping.csv` | GO / UniProt or manual | Protein → GO Biological Process → **involved_in** (Pathway nodes). |
| `data/protein_outcome_mapping.csv` / `data/protein_outcome_mapping_onc.csv` | CTD / OpenTargets | Protein → Disease/AE → **associated_with** edges. |

**Graph schema:** Nodes = Drug, Protein, Gene/Protein, Disease, Adverse Event, Pathway, Chemical, Gene. Edges = inhibits, treats, associated_with, involved_in, co_occurs_with_*, etc. **Key chains:** Hypertension ↔ KDR; Neutropenia ↔ JAK2/FLT3; Pneumonitis (text-mined).

---

## Results & performance

**KG statistics (final graph, after enrichment & expansion):**
- **1,084 nodes** (≈580 before expansion)
- **1,464 edges** (≈1,456 before expansion)
- **516 candidate proteins** for ranking (13 known Pralsetinib off-targets in KG) — ~34× increase over pre-expansion protein set

**Prediction performance** (from `notebooks/model_eval.ipynb`):

*Fair evaluation (held-out test set):* Train/test split on known (Pralsetinib, inhibits, protein) edges: **9 train**, **4 held-out**, 516 candidates. GNN trained only on train positives, evaluated on held-out proteins.
- **Recall@5:** 25% · **Recall@10:** 50% · **Recall@20:** 100%
- **MRR:** 0.147 · **Mean rank (test):** 11.5

*Pipeline comparison (canonical `predictions/off_target_predictions.csv` vs baseline):* Full pipeline GNN (trained on all 13 known targets) vs baseline on top-k. Known targets in top-k: **GNN** 4, 9, 13 at k=5,10,20; **Baseline** 5, 7, 7. Overlap in top-k: 2, 3, 7. This comparison is optimistic for the GNN because it is evaluated on targets it was trained on; the fair comparison is the held-out metrics above.

**Key finding:** On the held-out test set, the GNN recovers all four held-out known targets within top-20 (Recall@20 = 100%). The baseline uses only direct KG edge weights; the GNN uses the full graph and multi-hop structure to rank proteins, including those without direct (drug, inhibits, protein) edges.

## Repo layout

| Path | Purpose |
|------|--------|
| **`run.py`** | Entry point: `build_kg`, `enrich`, `visualize`, `train`, `predict`, `all` |
| `data/` | KG inputs and KG CSVs |
| `predictions/` | off_target_predictions*.csv (see table above) |
| **Scripts** (by type) | |
| `scripts/kg_construction/` | **KG build, enrichment & expansion:** `build_kg_from_sources.py`, `enrich_and_expand_kg.py` |
| `scripts/modeling/gnn/` | **GNN:** `kg_gnn_data.py`, `kg_gnn_model.py`, `kg_gnn_link_prediction.py`, `build_off_target_predictions.py`, `eval_gnn_holdout.py` |
| `scripts/modeling/baseline/` | **Baseline:** `kg_baseline_link_prediction.py` |
| `scripts/viz/` | **Visualization:** `visualize_kg.py` |
| `notebooks/eda.ipynb` | Exploratory analysis on final KG |
| `notebooks/model_eval.ipynb` | GNN link-prediction evaluation (train/test split, metrics, figures) |
