# Copyright (c) OpenMMLab. All rights reserved.
from .base import RotatedBaseDetector
from .oriented_rcnn import OrientedRCNN
from .two_stage import RotatedTwoStageDetector

__all__ = [
    'RotatedBaseDetector', 'RotatedTwoStageDetector', 'OrientedRCNN'
]
