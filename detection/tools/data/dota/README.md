# Preparing DOTA Dataset

<!-- [DATASET] -->

```bibtex
@InProceedings{Xia_2018_CVPR,
author = {Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
title = {DOTA: A Large-Scale Dataset for Object Detection in Aerial Images},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```

## Download DOTA

Download the dataset from the [DOTA website](https://captain-whu.github.io/DOTA/dataset.html).

The data structure is as follows:

```none
CFMNet
├── data
│   ├── DOTA
│   │   ├── train
│   │   ├── val
│   │   ├── test
```

## Split DOTA

Edit `img_dirs`, `ann_dirs`, and `save_dir` in the selected JSON file. The paper uses multi-scale 1024 x 1024 training crops with scale rates `[0.5, 1.0, 1.5]` and single-scale validation/test crops.

```shell
python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ms_trainval.json

python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_val.json
```

Set `data_root` and `eval_data_root` in the selected file under `configs/cfmnet/` to the generated directories. The splitting utility is adapted from BboxToolkit.
