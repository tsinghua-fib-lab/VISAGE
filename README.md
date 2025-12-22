# VISAGE
---
This is the implementation of our paper Perceiving Exposure Segregation with AI Urban Scientist.

<p align="center">
    <a href="https://github.com/tsinghua-fib-lab/VISAGE/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
    <a href="https://www.python.org/downloads/release/python-390/"><img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python"></a>
    <a href="https://github.com/tsinghua-fib-lab/VISAGE"><img src="https://img.shields.io/badge/Status-Research-success.svg" alt="Status"></a>
</p>

---

## 💡 Introduction

**VISAGE** is the first **AI Urban Scientist** framework designed for autonomous urban sensing. It addresses the challenge of measuring **socioeconomic exposure segregation**—the degree to which daily encounters cross different income groups—using only open satellite and street-level imagery. 

By bridging high-level social theory with large-scale observation through automated reasoning, VISAGE completes a full scientific closed-loop:

1.  **Literature Agent**: LLM agents distill cross-disciplinary theory from extensive literature into an interpretable visual codebook.
2.  **Experiment Agent**: Automatically detects codebook cues in imagery and generates structured, stepwise reasoning templates.
3.  **Perception Agent**: A domain-adapted Large Multi-modal Model (LMM) that reasons from scene semantics to infer exposure segregation.
4.  **Feedback Loop**: Out-of-sample performance is fed back to the system to iteratively update hypotheses and reasoning protocols.

---

## 🌟 Framework

VISAGE reframes urban perception as an interpretable reasoning task by organizing three specialized agents into a closed-loop discovery process.

![Loading Overview](assets/framwork.png "The VISAGE architecture: integrating literature, experiment, and perception agents.")


---

## ⚙️ Installation

### Environment
* **OS**: Linux (Ubuntu 20.04/22.04 recommended).
* **Python**: >= 3.9.
* **GPU**: Training requires 4 x NVIDIA A100 (80GB VRAM) for multi-modal processing.

### Setup
```bash
# Clone the repository
git clone https://github.com/tsinghua-fib-lab/VISAGE.git
cd VISAGE

# Create and activate environment
conda create -n visage python=3.9
conda activate visage

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Literature Agent Workflow
Automate the knowledge distillation process to generate a visual cue codebook:
```bash
python scripts/run_literature_agent.py --config configs/literature.yaml
```

### 2. Cue Detection & Aggregation
Process community imagery to generate normalized frequency tables:
```bash
python scripts/extract_cues.py --data_path ./data/communities/ --codebook ./codebooks/final_v1.json
```

### 3. Perception Inference
Run the domain-adapted LMM to infer segregation indices with Chain-of-Thought traces:
```bash
python scripts/run_perception.py --task inference --communities ./data/test_split.json
```

## 📊 Performance

VISAGE establishes a reliable, scalable pathway for urban perception using open data.

- **Predictive Reliability**: Achieves a Pearson correlation of $r=0.770$ across 10,030 communities in 31 U.S. cities.
- **Mechanism Discovery**: Unravels how "defensiveness" cues (e.g., high fences) drive segregation, while "interaction" cues (e.g., public spaces) foster mixing.
- **Policy Sensitivity**: Successfully evaluates the impact of Inclusionary Housing programs, identifying lower segregation in policy-active areas.

## 📧 Contact

For questions regarding the code or data, please contact:
- Yong Li: liyong07@tsinghua.edu.cn
- FIB-Lab, Department of Electronic Engineering, Tsinghua University
