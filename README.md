---
title: "IKRAE Ednet: Explainable Adaptive Learning with Real EdNet Data"
subtitle: "User & Reviewer Guide"
author: "AZIZ ABDELKARIM"
date: "October 27, 2025"
geometry: margin=1in
colorlinks: true
urlcolor: blue
fontfamily: helvet
---

# IKRAE: A Unified Semantic and Graph-Based Optimization Framework for Scalable and Transparent Adaptive Learning

![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12345678.svg){width=30%}  
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)  
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)

---

## For Users (Students, Educators, Developers)

### What is IKRAE?

**IKRAE** generates **personalized, explainable learning paths** using **real learner data** from [EdNet](https://github.com/riiid/ednet) (131M interactions, 784K users).  
It answers:  
> *"What should this learner do next — and why?"*

- **Input**: Your device, time, language, and current skills  
- **Output**: Optimal sequence of questions + lectures + **full explanation**  
- **Speed**: <200ms per recommendation  
- **Explainable**: Every skip has a reason (e.g., "low mastery in algebra")

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
