# CFMNet Classification

Scene classification entry for CFMNet.

## Train

```bash
python train.py --dataset AID-82 --data-path ../data/AID \
  --pretrained ../pretrained/cfmnet_imagenet1k.pth --device cuda:0
```

Supported datasets in the training script: `AID-82`, `UCM-82`, and `RESISC45-82`. The dataset root should contain `train/` and `val/` folders.

To create the paper's 80/20 split without modifying the source images:

```bash
python split_datasets/split_train_val_dataset.py SOURCE OUTPUT --val-ratio 0.2 --seed 0
```

The repository exposes the canonical CFMNet configuration used in the paper's main comparisons and full-model ablations.

## Main Files

- `train.py`: classification training entry.
- `utils.py`: dataset loading, model selection, training, and evaluation utilities.
- `speed_test.py` and `onnx_fps_test.py`: CFMNet runtime benchmarks.

Generated `class_train_indices.json` and `class_val_indices.json` files are ignored by Git.
