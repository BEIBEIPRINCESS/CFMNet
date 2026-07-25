# CFMNet Segmentation

Semantic segmentation code for CFMNet with UNetFormer-style decoders.

## Train

```bash
python train_supervision.py -c config/loveda/unetformer_cfmnet_e30.py
```

## Main Files

- `geoseg/models/cfmnet.py`: CFMNet segmentation backbone.
- `geoseg/models/UNetFormer_cfmnet.py`: CFMNet segmentation model wrapper.
- `config/loveda/`: LoveDA training config.
- `config/vaihingen/`: Vaihingen training config.

Only the canonical CFMNet configuration used in the paper's main comparisons and full-model ablations is provided.

Place LoveDA and Vaihingen datasets under `../data/` or edit `data_root` in the selected config.
Set `backbone_ckpt_path` in the selected config to the ImageNet-1K CFMNet checkpoint used by the paper. Leave it as `None` to initialize from scratch.

## License

The segmentation subproject is distributed under GPL-3.0 because it is derived
from GeoSeg/UNetFormer. See [`LICENSE`](LICENSE) and
[`UPSTREAM.md`](UPSTREAM.md).
