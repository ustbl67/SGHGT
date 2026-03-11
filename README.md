# SGHGT: A Saliency-Guided Hierarchical Gated Transformer for No-Reference Image Quality Assessment
This is the official implementation of the paper: **"SGHGT: A Saliency-Guided Hierarchical Gated Transformer for No-Reference Image Quality Assessment"**.
--
## Introduction
This repository provides the core code for **SGHGT**, a novel No-Reference Image Quality Assessment (NR-IQA) framework. The model improves assessment accuracy through several key innovations:
* **Saliency-Guided Encoding**: Leverages saliency maps to guide the model to focus on regions sensitive to the Human Visual System (HVS).
* **Hierarchical Attention Pooling (HAP)**: Replaces traditional global average pooling to achieve efficient multi-scale feature aggregation.
* **Gated Transformer Architecture**: Effectively fuses semantic and texture features using a gating mechanism to improve feature representation robustness.
---
## Project Structure
```text
SGHGT/
├── model.py            # Core architecture of the SGHGT model
├── config.py           # Configuration for paths and hyperparameters
├── dataset.py          # Data parsing and loading for TID, KADID, CLIVE, etc.
├── train.py            # Main training script
├── evaluate.py         # Evaluation script (SROCC/PLCC, scatter plots)
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation
```

## Dataset
TID2013: <https://ponomarenko.info/tid2013.htm>

CID2013: <https://zenodo.org/records/2647033>

LIVEC: <https://live.ece.utexas.edu/research/ChallengeDB/index.html>

KADID-10K: <https://database.mmsp-kn.de/kadid-10k-database.html>

---
## Saliency map
DeepgazeIIE: <https://github.com/matthias-k/DeepGaze>

---

## Training
```text
run train.py
```

## Evaluate
```text
run evaluate.py
```

