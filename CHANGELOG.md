# Changelog

All notable changes to IKRAE will be documented in this file.

---

## [v1.0.4] — 2025-02-20
### 🎉 First Stable, Reproducible Release

This is the first fully stable release of **IKRAE** (Intelligent Knowledge-based Recommendation for Adaptive Education), integrating semantic reasoning (OWL/SWRL + HermiT) with graph-based optimization (Dijkstra + k-shortest paths) on real **EdNet-KT3** data.

### ✨ Added
- Full end-to-end adaptive learning pipeline:
  - **ednet_loader.py** — LO construction & prerequisite extraction  
  - **ikrae_reasoner.py** — semantic feasibility (OWL + SWRL → Gf)  
  - **run_experiments.py** — optimization, path generation, cost metrics  
- **Java reasoning engine** (`IKRAEReasoner.java`)  
  - OWLAPI + HermiT  
  - JPype-based Python wrapper  
- **Zenodo dataset integration**  
  - Auto-download of `ikrae_kt3_clean.zip`  
  - Removed all Google Drive dependencies  
- Stable **Dockerfile** (Python 3.10 + OpenJDK 17)  
- Clean, versioned **requirements.txt**  
- Full **pipeline script** (`run_pipeline.sh`)  
- All outputs stored in `experiments/results/`

### 🔧 Improved
- Refactored Java reasoner (OWLAPI 6 compliant)  
- Improved LO builder (duration/accuracy aggregation)  
- Reliable KT3 preprocessing (cleaned, deduplicated dataset)  
- Better folder structure (`src/java`, `ontology`, `graphs`, `contexts`)  
- Faster downloads and safer ZIP extraction  
- CI-friendly pipeline (GitHub Actions compatible)

### 🧹 Removed
- `gdown` and Google Drive download logic  
- Deprecated HermiT constructor  
- Old inconsistent paths in scripts  
- Previous temporary or redundant data files

### 📦 Dataset
- **DOI:** https://doi.org/10.5281/zenodo.17664110  
- Clean EdNet-KT3 dataset prepared for IKRAE  
- ~700K learners, 131M interactions  
- Preprocessed nodes + sequential prerequisites

### 📚 Software DOI
- **DOI:** https://doi.org/10.5281/zenodo.17464127

### 🚧 Known Limitations
- Semantic layer optimized for a single domain  
- Static weights in cost function  
- Evaluation based on cleaned KT3, not real-time learner logs  

### 🗺 Roadmap (v1.1.0)
- Add adaptive cost weighting (Bayesian / RL-based tuning)  
- Multi-domain support in ontology  
- Visualization dashboard for feasibility graph  
- Optional FastAPI microservice for real-time API calls  
- Integration with EdNet-KT4 for sequential-path comparison  

---

## Legend
- **Added** → new features  
- **Improved** → updates & refactoring  
- **Removed** → deprecated or unnecessary components  
