# Upstream Attribution

The segmentation code is derived from GeoSeg and its UNetFormer
implementation:

- Repository: https://github.com/WangLibo1995/GeoSeg
- License: GPL-3.0

CFMNet retains the task framework, dataset utilities, losses, and
UNetFormer-style decoder components needed by the paper. The CFMNet backbone
and task configs have been added or modified for this release.

Because this subproject is derived from GPL-3.0 code, redistribution and
modification of the segmentation code must comply with GPL-3.0. Retain
`LICENSE`, this attribution file, and applicable source-file notices.
