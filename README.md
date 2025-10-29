# IKRAE: A Unified Semantic and Graph-Based Optimization Framework for Scalable and Transparent Adaptive Learning

**A Research study  by** | **AZIZ ABDELKARIM** | **Real EdNet (131M interactions)**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17464127.svg)](https://doi.org/10.5281/zenodo.17464127)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)  
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](docker/Dockerfile)

---

## Reproducibility Statement

> **All experiments in the paper are 100% reproducible using this repository.**  
> **No synthetic data** — **only real EdNet-KT3** (50K sampled users, 100K+ LOs).  
> **Runtime, CS, and explainability metrics match paper (Section 6.5)**.

| Paper Claim | Repo File | Command |
|-----------|---------|--------|
| **Scalability** | `experiments/results/scalability.png` | `python src/run_experiments.py` |
| **Constraint Satisfaction** | `path_trace.json` | `docker run ...` |
| **Re-planning <200ms** | `runtime_ms` field | Live API |
| **EdNet Integration** | `ednet_loader.py` | Loads KT3 CSVs |

---

## For Users (Students, Educators, Developers)

### What is IKRAE-EdNet?

**IKRAE-EdNet** generates **personalized, explainable learning paths** using **real learner data** from [EdNet](https://github.com/riiid/ednet) (131M interactions, 784K users).  
It answers:  
> *"What should this learner do next — and why?"*

- **Input**: Device, time, language, mastery  
- **Output**: Optimal sequence of questions + lectures + **full explanation**  
- **Speed**: <200ms per recommendation  
- **Explainable**: Every exclusion has a rule-based reason

---

### Try It in 3 Minutes

```bash
# 1. Clone
git clone https://github.com/yourname/ikrae-ednet.git
cd ikrae-ednet

# 2. Run with Docker (no setup!)
docker build -t ikrae-ednet docker/
docker run --rm -v $(pwd)/experiments:/app/experiments ikrae-ednet

# 3. See your path!
cat experiments/results/path_trace.json

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
