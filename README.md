# Class-Incremental-Few-Shot-Underwater-Object-Detection-Framework
To this end, this paper proposes a class-incremental few-shot underwater object detection framework(CIFS-UD)
This repository provides the official implementation of CIFS-UD, a Class-Incremental Few-Shot Underwater Object Detection framework designed to address catastrophic forgetting and data scarcity in underwater object detection.

The framework follows a three-stage training paradigm:

Base Training

Prototype Extraction

Few-Shot Incremental Fine-Tuning

1. Environment Setup

1.1 Requirements

Recommended environment:

Python >= 3.8

CUDA 11.8

Linux (Ubuntu recommended)

Core dependencies:

torch==2.0.0+cu118
torchvision==0.15.1+cu118
pycocotools==2.0.7
fvcore==0.1.5.post20221221
opencv-python==4.11.0.86

Install PyTorch (CUDA 11.8):

pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

Install other dependencies:

pip install -r requirements.txt

2. Build

Compile custom operators (if any):

python setup.py build develop

Ensure that no compilation errors occur.

3. Dataset Preparation

3.1 Supported Datasets

Brackish Dataset

TrashCan Dataset

RUOD Dataset

All datasets must be converted to VOC2007 format.

3.2 VOC2007 Directory Structure

datasets/
│
├── VOC2007/
│   ├── JPEGImages/
│   ├── Annotations/
│   ├── ImageSets/
│   │   ├── Main/
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   │   ├── test.txt

Important notes:

Annotation files must follow Pascal VOC XML format.

Class splits must match configuration files.

Base and novel splits must remain consistent across all stages.

4. Training Pipeline

The CIFS-UD framework adopts a three-stage incremental learning strategy.

Stage 1: Base Training

Train the detector on base classes:

python fs_nwpu/train_fasterrcnn.py --config-file configs/NWPU/base_training/split1.yml

Outputs:

Base model weights

Backbone trained on base categories

Stage 2: Prototype Extraction

After base training, extract class prototypes.

Step 1: Extract Features

python fs_nwpu/prototype_fasterrcnn.py --config-file configs/NWPU/prototype/split1.yml

Step 2: Convert Features to Prototypes

python fs_nwpu/scripts/convert_feature_as_prototype.py --split 1

Outputs:

Stored prototype representations for each base class

Stage 3: Few-Shot Incremental Fine-Tuning

3.1 Checkpoint Surgery

Prepare model weights for incremental learning:

python fs_nwpu/scripts/ckpt_surgery.py \
    --src1 weights/NWPU_R101_split1.pth \
    --method randinit \
    --save-dir work_dirs/nwpu_resnet101_base1_all_redetect/

3.2 Fine-Tuning on Novel Classes

Example: 10-shot incremental training

python fs_nwpu/inc_train_fasterrcnn.py --config-file configs/NWPU/split1/inc/10shot_INC_CLP.yml

3.3 Evaluation

Evaluate incremental model:

python fs_nwpu/test_fasterrcnn.py --config-file configs/NWPU/split1/inc/10shot_INC_CLP.yml --eval-only

Evaluation metrics:

bAP50 (Base classes AP@50)

nAP50 (Novel classes AP@50)

mAP50 (Overall AP@50)

5. Project Structure

fs_nwpu/
│
├── train_fasterrcnn.py
├── inc_train_fasterrcnn.py
├── prototype_fasterrcnn.py
├── test_fasterrcnn.py
├── scripts/
│   ├── convert_feature_as_prototype.py
│   ├── ckpt_surgery.py
│
configs/
│   ├── NWPU/
│       ├── base_training/
│       ├── prototype/
│       ├── split1/

6. Notes

Prototype extraction must be performed after base training.

Ensure consistent backbone weights across all stages.

Confirm dataset splits before training.

Check that class ordering matches configuration files.

7. Citation

If you find this work useful, please cite:
