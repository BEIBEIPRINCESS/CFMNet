# CFMNet

Official research code for **CFMNet: A Lightweight Backbone with Cooperative Feature Modeling for Remote Sensing Vision Tasks**. CFMNet is built around four complementary modules: **TCRM**, **SSRM**, **LDFM**, and **EGPCM**. This repository covers scene classification, oriented object detection, and semantic segmentation.

The manuscript is included as [`CFMNet_manuscript.pdf`](CFMNet_manuscript.pdf).

## Highlights

- Unified CFMNet naming across classification, detection, and segmentation.
- A single canonical model configuration matching the paper's main comparisons and full-model ablations.
- CFMNet implementations only; checkpoints, logs, generated results, and change-detection code are excluded.
- Detection code retains only the Oriented R-CNN components used by the paper.
- Segmentation code provides CFMNet backbones with UNetFormer-style decoders.

## Repository Layout

```text
CFMNet/
├── classification/          # Scene classification code
├── detection/               # MMRotate-based oriented object detection code
├── segmentation/            # Semantic segmentation code
├── CFMNet_manuscript.pdf    # Manuscript
├── CITATION.cff
├── LICENSE
└── README.md
```

## Core Implementations

| Task | Main implementation | Typical entry |
| --- | --- | --- |
| Classification | `classification/models/CFMNet.py` | `classification/train.py` |
| Oriented detection | `detection/mmrotate/models/backbones/CFMNet.py` | `detection/configs/cfmnet/oriented_rcnn_cfmnet_dotav10.py` |
| Semantic segmentation | `segmentation/geoseg/models/cfmnet.py` | `segmentation/config/loveda/unetformer_cfmnet_e30.py` |

## Model Configuration

The release exposes one canonical CFMNet configuration:

| Setting | Value |
| --- | --- |
| Stage channels | `[96, 192, 384, 768]` |
| Stage depths | `[1, 4, 4, 2]` |
| SSRM directional kernels | `[11, 11, 11, 11]` |
| EGPCM dynamic kernel | `3 x 3` |
| EGPCM large kernel | `13 x 13` |
| EGPCM modeled-channel ratios | `[1/8, 1/4, 1/2, 1/2]` |
| MLP ratio | `2` |
| Activation | ReLU |
| Drop-path rate | `0.1` |

This is the full four-branch model used in the paper's main comparisons and complete-model ablations. The five scaled models reported in the controlled Pareto analysis are not exposed as separate code variants in this release.

## Modules

| Module | Full name | Role |
| --- | --- | --- |
| TCRM | Texture-Aware Channel Recalibration Module | Adaptive sparse channel interaction and semantic reallocation. |
| SSRM | Spatial Structure Reorganization Module | Directional structure modeling for anisotropic targets. |
| LDFM | Local Detail Feature Modulator | Lightweight local-detail preservation and refinement. |
| EGPCM | Efficient Global Perception Convolution Module | Partial-channel dynamic and large-kernel global context. |

## Main Results

| Task | Dataset | Metric | CFMNet |
| --- | --- | --- | ---: |
| Scene classification | NWPU-RESISC45 | Top-1 accuracy | 96.13% |
| Scene classification | AID | Top-1 accuracy | 95.50% |
| Scene classification | UCM | Top-1 accuracy | 98.10% |
| Oriented detection | DOTA-v1.0 | mAP | 79.82% |
| Oriented detection | DOTA-v1.5 | mAP | 73.02% |
| Oriented detection | HRSC2016 | mAP | 90.82% |
| Semantic segmentation | Vaihingen | mIoU | 83.8% |
| Semantic segmentation | LoveDA | mIoU | 53.8% |

## Installation

Use separate environments for the three tasks when reproducing results, because they depend on different upstream frameworks.

```bash
# Classification
cd classification
pip install -r requirements.txt

# Detection
cd ../detection
pip install -r requirements.txt
pip install -v -e .

# Segmentation
cd ../segmentation
pip install -r requirements.txt
```

## Data and Weights

Datasets and pretrained weights are not included. A recommended local layout is:

```text
CFMNet/
├── data/
│   ├── AID/
│   ├── UCMerced/
│   ├── NWPU-RESISC45-82/
│   ├── split_ms_dota/
│   ├── split_ss_dota/
│   ├── split_ms_dota15/
│   ├── split_ss_dota15/
│   ├── HRSC2016/
│   ├── LoveDA/
│   └── Vaihingen/
└── pretrained/
    └── cfmnet_imagenet1k.pth
```

The reported downstream experiments use a CFMNet backbone pretrained on ImageNet-1K for 300 epochs. Set `--pretrained` for classification, `pretrained_ckpt` in a detection config, or `backbone_ckpt_path` in a segmentation config. Leave these values empty to train from scratch.

## Quick Start

### Scene Classification

```bash
cd classification
python train.py --dataset AID-82 --data-path ../data/AID \
  --pretrained ../pretrained/cfmnet_imagenet1k.pth --device cuda:0
```

### Oriented Object Detection

```bash
cd detection
bash tools/dist_train.sh configs/cfmnet/oriented_rcnn_cfmnet_dotav10.py 1
```

### Semantic Segmentation

```bash
cd segmentation
python train_supervision.py -c config/loveda/unetformer_cfmnet_e30.py
```

## License

This repository contains separately licensed components:

| Scope | License | License file |
| --- | --- | --- |
| Top-level and classification code | MIT | [`LICENSE`](LICENSE) |
| Detection code derived from MMRotate | Apache-2.0 | [`detection/LICENSE`](detection/LICENSE) |
| Segmentation code derived from GeoSeg/UNetFormer | GPL-3.0 | [`segmentation/LICENSE`](segmentation/LICENSE) |

The manuscript PDF is not distributed under the software licenses above. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for upstream attribution and
license scope.
