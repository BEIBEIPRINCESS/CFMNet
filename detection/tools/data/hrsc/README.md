# Preparing HRSC Dataset

<!-- [DATASET] -->

```bibtex
@conference{hrsc,
    author = {Zikun Liu. and Liu Yuan. and Lubin Weng. and Yiping Yang.},
    title = {A High Resolution Optical Satellite Image Dataset for Ship Recognition and Some New Baselines},
    booktitle = {Proceedings of the 6th International Conference on Pattern Recognition Applications and Methods - ICPRAM,},
    year = {2017},
    pages = {324-331},
    publisher = {SciTePress},
    organization = {INSTICC},
    doi = {10.5220/0006120603240331},
    isbn = {978-989-758-222-6},
    issn = {2184-4313},
}
```

## Prepare HRSC2016


The data structure is as follows:

```none
CFMNet
├── data
│   ├── HRSC2016
│   │   ├── FullDataSet
│   │   │   ├─ AllImages
│   │   │   ├─ Annotations
│   │   │   ├─ LandMask
│   │   │   ├─ Segmentations
│   │   ├── ImageSets
```

Set `data_root` in `configs/cfmnet/oriented_rcnn_cfmnet_hrsc2016.py` if the dataset is stored elsewhere.
