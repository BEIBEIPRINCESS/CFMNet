#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 python train.py --dataset AID-82
CUDA_VISIBLE_DEVICES=0 python train.py --dataset UCM-82
CUDA_VISIBLE_DEVICES=0 python train.py --dataset RESISC45-82
