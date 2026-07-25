# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_dataset  # noqa: F401, F403
from .dota import DOTADataset  # noqa: F401, F403
from .hrsc import HRSCDataset  # noqa: F401, F403
from .pipelines import *  # noqa: F401, F403
from .dota_1_5 import DOTADataset15
__all__ = ['DOTADataset', 'DOTADataset15', 'HRSCDataset', 'build_dataset']
