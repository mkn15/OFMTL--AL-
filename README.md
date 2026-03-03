# OFMTL-AL: Robust Online Federated Multi-task Learning with Adaptive Loss

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2403.XXXXX-b31b1b.svg)](https://arxiv.org)

<div align="center">
  <img src="docs/figures/overview.png" alt="OFMTL-AL Overview" width="600"/>
  <p><em>Figure 1: Overview of OFMTL-AL framework</em></p>
</div>

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Contributions](#-key-contributions)
- [Methodology](#-methodology)
- [Results](#-results)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Reproducing Experiments](#-reproducing-experiments)
- [Citation](#-citation)
- [License](#-license)

## 🔍 Overview

Federated learning enables collaborative model training across distributed devices while preserving data privacy. However, real-world deployments face two critical challenges: (1) devices join the system dynamically (online setting), and (2) sensor data often contains noisy labels. Existing Online Federated Multi-task Learning (OFMTL) efficiently handles new devices but lacks robustness to label noise.

**OFMTL-AL** addresses this gap by incorporating an **adaptive loss function** that dynamically down-weights noisy samples during training, providing provable robustness while maintaining the efficiency of online federated learning.

## 🏆 Key Contributions

| Contribution | Description |
|--------------|-------------|
| **1. Adaptive Loss Function** | Novel loss that smoothly interpolates between ℓ₂ and ℓ₂,₁ norms to identify and mitigate noisy labels |
| **2. Theoretical Guarantees** | Proven convergence rates (𝒪(1/T)) and noise robustness bounds (𝒪(η·dₘₐₓ/dₘᵢₙ)) |
| **3. Comprehensive Evaluation** | Extensive experiments on 3 real-world datasets with injected label noise (0-40%) |
| **4. Open-Source Implementation** | Complete, well-documented code for reproducibility |

## 📊 Results

<div align="center">
  <img src="results/figures/summary.png" alt="Results Summary" width="700"/>
  <p><em>Figure 2: Performance improvement over baselines</em></p>
</div>

### Performance at 30% Label Noise

| Method | HAR | GLEAM | EHM | Average Improvement |
|--------|-----|-------|-----|---------------------|
| **OFMTL-AL (Ours)** | **3.37%** | **3.51%** | **3.25%** | **—** |
| OFMTL | 4.90% | 5.12% | 4.72% | 31.3% ↑ |
| MOCHA | 6.72% | 6.95% | 6.48% | 49.7% ↑ |
| Local SVM | 8.86% | 9.18% | 8.54% | 61.9% ↑ |

### Robustness Scores (Higher = Better)

| Dataset | OFMTL-AL | OFMTL | MOCHA | Local SVM |
|---------|----------|-------|-------|-----------|
| HAR | **48.8** | 45.8 | 44.5 | 45.6 |
| GLEAM | **47.8** | 44.7 | 44.1 | 44.8 |
| EHM | **48.6** | 46.0 | 44.5 | 46.0 |

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/OFMTL-AL.git
cd OFMTL-AL

# Install dependencies
pip install -r requirements.txt

# Install the package (optional)
pip install -e .
