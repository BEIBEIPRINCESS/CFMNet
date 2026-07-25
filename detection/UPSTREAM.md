# Upstream Attribution

The detection code is derived from MMRotate 0.3.4:

- Repository: https://github.com/open-mmlab/mmrotate
- Release: v0.3.4
- License: Apache-2.0

CFMNet retains the MMRotate package structure and the subset of Oriented R-CNN
components required by the paper. CFMNet-specific backbone code, configs, and
documentation have been added or modified for this release.

MMRotate depends on MMDetection, MMCV, and other OpenMMLab components. Those
dependencies are not vendored in this repository and remain subject to their
respective licenses.

When redistributing this detection subproject, retain `LICENSE`, `NOTICE`, the
upstream copyright notices in source files, and this attribution file.
