# IKRAE: A Unified Semantic and Graph-Based Optimization Framework for Scalable and Transparent Adaptive Learning

**A Research Study by** | **Aziz Abdelkarim** | **Real EdNet (131M interactions)**  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17464127.svg)](https://doi.org/10.5281/zenodo.17464127)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](docker/Dockerfile)

---

## 📘 Overview

**IKRAE (Intelligent Knowledge-based Recommendation for Adaptive Education)**  
is a unified semantic and graph-based optimization framework that generates **context-aware, explainable, and computationally efficient learning paths** at MOOC scale.  

It integrates:
- **OWL + SWRL semantic reasoning** for constraint enforcement  
- **Graph-based optimization (Dijkstra + Yen)** for scalable path discovery  
- **Explainable AI** to ensure transparency in every recommendation  

All experiments are fully reproducible using real **EdNet-KT3 data (131M interactions, 784K learners)**.

---

## 🧪 Reproducibility Statement

> **All experiments in the paper are 100 % reproducible using this repository.**  
> **No synthetic data** — only **real EdNet-KT3** (50 K sampled users, 100 K + LOs).  
> **Runtime, constraint satisfaction, and explainability metrics** match the paper (Section 6.5).

| Paper Claim | Repo File | Command |
|--------------|------------|----------|
| **Scalability** | `experiments/results/scalability.png` | `python src/run_experiments.py` |
| **Constraint Satisfaction** | `path_trace.json` | `docker run ...` |
| **Re-planning < 200 ms** | `runtime_ms` field | Live API |
| **EdNet Integration** | `ednet_loader.py` | Loads KT3 CSVs |

---

## 🧩 IKRAE-EdNet Pipeline Overview

```markdown
```mermaid
flowchart LR
    subgraph A[EdNet Dataset]
        A1[(KT3 Interactions.csv)]
        A2[(Questions.csv)]
        A3[(Lectures.csv)]
    end

    subgraph B[Preprocessing & Sampling]
        B1["Stratified Learner Sampling\n(ikrae-ednet-sampling.py)"]
        B2["Feature Extraction\n(Preprocessing.py)"]
        B3["Skill DAG Construction\n(ednet_loader.py)"]
    end

    subgraph C[Semantic Reasoning Layer]
        C1["Ontology\n(ikrae_ednet.owl)"]
        C2["SWRL Rules\n(ikrae_swrl_rules.txt)"]
        C3["HermiT + Py4J Bridge\n(ikrae_reasoner.py + IKRAEReasoner.java)"]
    end

    subgraph D[Graph-Based Optimization Layer]
        D1["Weighted DAG Builder\n(ikrae_optimizer.py)"]
        D2["Dijkstra + Yen’s k-Shortest Paths\n(Adaptive Path Search)"]
        D3["Explainability Generator\n(JSON Trace + Exclusions)"]
    end

    subgraph E[Evaluation & Results]
        E1["Runtime & Scalability\n(run_experiments.py)"]
        E2["Visualization & Metrics\n(scalability.png)"]
        E3["Explainable Output\n(path_trace.json)"]
    end

    A1 --> B1 --> B2 --> B3 --> C3
    A2 --> B2
    A3 --> B2
    C1 --> C3
    C2 --> C3 --> D1 --> D2 --> D3 --> E3
    D1 --> E1
    E1 --> E2

    style A fill:#f0f8ff,stroke:#088178,stroke-width:1px
    style B fill:#e7fff5,stroke:#088178,stroke-width:1px
    style C fill:#fff5e7,stroke:#088178,stroke-width:1px
    style D fill:#f9f9ff,stroke:#088178,stroke-width:1px
    style E fill:#f0fff0,stroke:#088178,stroke-width:1px
