import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from geoseg.models.cfmnet import CFMNet


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = CFMNet()
    total, trainable = count_params(model)
    print(f"Total params: {total:,} ({total / 1e6:.3f} M)")
    print(f"Trainable params: {trainable:,} ({trainable / 1e6:.3f} M)")
