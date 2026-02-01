# DSC180B-B23-Knowledge-Graph-and-Biomedical-Ontology

https://pubchem.ncbi.nlm.nih.gov/compound/129073603

# Pralsetinib Polypharmacology & Toxicity Knowledge Graph

## 1. Project Overview
This project establishes a **Knowledge Graph (KG)** to model the polypharmacology of Pralsetinib, a selective RET kinase inhibitor. Beyond its primary therapeutic mechanism, this graph maps **off-target interactions** (e.g., KDR, JAK2) directly to clinical **Adverse Events** (e.g., Hypertension, Neutropenia).

**The Objective:** To demonstrate a **Mechanism-to-Toxicity** analysis where the graph structure reveals how specific off-target molecular inhibition drives observed clinical side effects.

## 2. Dataset Specifications (Version 1.0)
We have consolidated structured bioassay data and unstructured literature text into a normalized Graph format suitable for ingestion into graph databases (e.g., Neo4j).

* **Total Nodes:** 240 (Drugs, Proteins, Genes, Diseases, Adverse Events)
* **Total Edges:** 313 (Relationships linking mechanisms to outcomes)
* **File Outputs:**
    * `kg_nodes.csv`: The entity dictionary containing unique IDs and Types.
    * `kg_edges.csv`: The relationship triplets (Source -> Relation -> Target).

## 3. Graph Schema & Logic
The graph connects molecular mechanisms to patient outcomes using two distinct reasoning paths:

**Path A: Therapeutic Efficacy (Primary Mechanism)**
> `Pralsetinib` --[inhibits]--> `RET (Primary Target)` --[treats]--> `NSCLC (Disease)`

**Path B: Toxicity Etiology (Off-Target Mechanism)**
> `Pralsetinib` --[inhibits]--> `KDR (Off-Target)` --[associated_with]--> `Hypertension (Adverse Event)`

*Note: By explicitly mapping Path B, our model can infer the molecular root cause of specific side effects.*

## 4. Data Sources & Pipeline
We utilized a Python-based pipeline to ingest, clean, and normalize disparate data sources from PubChem:

| Source File | Data Type | Contribution to Graph |
| :--- | :--- | :--- |
| **Bioassays** | Structured CSV | Provided quantitative **IC50** values (nM) for RET and verified off-targets. |
| **Literature** | Unstructured Text | Used text mining to extract **Adverse Events** (e.g., "Hypertension") from abstracts. |
| **Co-occurrences** | JSON | Identified latent chemical-gene associations not present in standard bioassays. |
| **Clinical Trials** | Structured CSV | Mapped the drug to approved **Disease Indications**. |

## 5. Key Findings Validated in the Graph
The current graph structure successfully captures the following "Mechanism-to-Toxicity" chains:

* **Hypertension:** Linked to **KDR (VEGFR2)** inhibition, a known class effect of VEGF pathway interference.
* **Neutropenia:** Linked to **JAK2/FLT3** inhibition, which impacts hematopoietic cell proliferation.
* **Pneumonitis:** Identified via text mining as a significant clinical risk.

## 6. Data Dictionary & Usage
The knowledge graph was constructed using specific datasets exported from PubChem. Below is the breakdown of each file and its role in the pipeline:

### A. Bioactivity Data (`...bioactivity.csv`)
* **Data Content:** Quantitative experimental results ($IC_{50}$, $K_i$, $K_d$) from biochemical assays.
* **Project Usage:** Defines the **"Inhibits"** edges. We use this to establish the primary potency against RET (0.3-0.4 nM) and identify confirmed off-targets (e.g., JAK2).

### B. Literature Abstracts (`...literature.csv`)
* **Data Content:** Titles, abstracts, and metadata for PubMed articles referencing Pralsetinib.
* **Project Usage:** The source for **Text Mining**. We scan these abstracts for keywords (e.g., "hypertension", "neutropenia") to generate **"Associated_With"** edges linking targets to Adverse Events.

### C. Co-Occurrence Data (`...Co-Occurrences...json`)
* **Data Content:** Computed associations between the drug and other genes/chemicals based on frequency in scientific text.
* **Project Usage:** Uncovers **"Hidden" Off-Targets**. This dataset revealed high-confidence associations with targets like KDR (VEGFR2) and EGFR that were underrepresented in the structured bioassay data.

### D. Clinical Trials (`...clinicaltrials.csv` & `...opentargets...csv`)
* **Data Content:** FDA clinical trial records, phases, and approved conditions.
* **Project Usage:** Defines the **"Treats"** edges. This establishes the "Clean" therapeutic pathway (Pralsetinib → NSCLC) to contrast against the toxicity pathways.

### E. Consolidated Targets (`...consolidatedcompoundtarget.csv`)
* **Data Content:** High-level summary of the drug's primary mechanisms of action from multiple databases (ChEMBL, TTD).
* **Project Usage:** Serves as the ground truth for **Node Classification**, ensuring the drug is correctly typed as a "RET Inhibitor" in the graph schema.

## 7. Ontology-Guided Enrichment (GO + Target-Outcome)
We extend the KG using ontology-backed biological facts to restore **Target-to-Outcome** reasoning without adding drug-specific shortcuts.

### A. GO Pathway Mapping (`protein_go_mapping.csv`)
* **Data Content:** Protein-to-GO Biological Process annotations (1-3 terms per protein).
* **Project Usage:** Adds `Pathway` nodes and **"involved_in"** edges to create shared intermediate biology.

### B. Target-Outcome Mapping (`protein_outcome_mapping.csv`)
* **Data Content:** Curated Protein → Disease/Adverse Event links from sources like CTD or OpenTargets.
* **Project Usage:** Adds **"Associated_With"** edges that bridge molecular targets to clinical outcomes.

### C. Enrichment Script (`scripts/enrich_go.py`)
* **Data Content:** Single-pass enrichment script for GO pathways + target-outcome links.
* **Project Usage:** Outputs updated `kg_nodes_enriched.csv` and `kg_edges_enriched.csv`.

### D. Optional GO Auto-Annotation
* **Data Content:** UniProt GO Biological Process annotations pulled by gene symbol.
* **Project Usage:** Auto-populates `protein_go_mapping.csv` for rapid bootstrapping.

**Run:**
```
python scripts/enrich_go.py --nodes data/kg_nodes_v2.csv --edges data/kg_edges_v2.csv --go-mapping data/protein_go_mapping.csv --outcome-mapping data/protein_outcome_mapping_onc.csv --out-nodes data/kg_nodes_final.csv --out-edges data/kg_edges_final.csv --add-similarity --drop-shortcuts
```

**Auto-annotate:**
```
python scripts/enrich_go.py --generate-go-mapping --go-mapping data/protein_go_mapping.csv --outcome-mapping data/protein_outcome_mapping.csv
```

### E. Visualization (`scripts/visualize_kg.py`)
* **Data Content:** Interactive PyVis view of the KG.
* **Project Usage:** Exports an HTML graph for quick inspection.

**Run (final KG):**
```
python scripts/visualize_kg.py --nodes data/kg_nodes_final.csv --edges data/kg_edges_final.csv --out figures/kg_interactive.html --max-nodes 300
```

**Run (mechanistic only):**
```
python scripts/visualize_kg.py --nodes data/kg_nodes_final.csv --edges data/kg_edges_final.csv --out figures/kg_interactive_mech.html --mechanistic-only --max-nodes 250
```