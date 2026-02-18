# Class-Incremental-Few-Shot-Underwater-Object-Detection-Framework
To this end, this paper proposes a class-incremental few-shot underwater object detection framework(CIFS-UD)
This repository provides the official implementation of CIFS-UD, a Class-Incremental Few-Shot Underwater Object Detection framework designed to address catastrophic forgetting and data scarcity in underwater object detection.
<div align="center">

# CIFS-UD  
## Class-Incremental Few-Shot Underwater Object Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-orange.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-red.svg)]()

</div>

---

## 📌 Overview

CIFS-UD is a **Class-Incremental Few-Shot Underwater Object Detection framework** designed to address:

- Catastrophic forgetting in class-incremental learning  
- Data scarcity in underwater environments  
- Prototype-guided incremental adaptation  

The framework adopts a **three-stage training paradigm**:

1. Base Training  
2. Prototype Extraction  
3. Incremental Few-Shot Fine-Tuning  

---

## 🗂 Dataset Preparation

Supported datasets:

- Brackish  
- TrashCan  
- RUOD  

All datasets must be converted to **Pascal VOC 2007 format**.

Directory structure:


---

## ⚙️ Installation

### Requirements

- Python >= 3.8  
- CUDA 11.8  
- PyTorch 2.0.0  

Install PyTorch:

```bash
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### Evaluation

python fs_ruod/test_fasterrcnn.py --config-file configs/RUOD/split1/inc/10shot_INC_CLP.yml --eval-only
