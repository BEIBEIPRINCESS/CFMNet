# CFMNet Detection

MMRotate-based Oriented R-CNN code for CFMNet. Components for unrelated detector families are intentionally excluded.

## Train

```bash
bash tools/dist_train.sh configs/cfmnet/oriented_rcnn_cfmnet_dotav10.py 1
```

## Main Files

- `mmrotate/models/backbones/CFMNet.py`: CFMNet detection backbone.
- `configs/cfmnet/`: Oriented R-CNN with CFMNet configs for DOTA-v1.0, DOTA-v1.5, and HRSC2016.
- `UPSTREAM.md`: attribution for the MMRotate codebase.

All three datasets use the same canonical CFMNet backbone configuration.

Place DOTA/HRSC2016 datasets under `../data/` or edit `data_root` in the selected config.
Set `pretrained_ckpt` in the selected config to the ImageNet-1K CFMNet checkpoint used by the paper. Leave it as `None` to initialize from scratch.

## License

The detection subproject is distributed under the Apache License 2.0. It is
derived from MMRotate 0.3.4 and includes CFMNet-specific modifications. See
[`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and [`UPSTREAM.md`](UPSTREAM.md).
