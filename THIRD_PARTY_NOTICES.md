# Third-Party Notices

This file documents the license scope and upstream projects used by CFMNet.
The referenced license files, rather than this notice, define the applicable
legal terms.

## License Scope

- Top-level files and `classification/` are distributed under the MIT License
  in `LICENSE`, except where an individual source file states otherwise.
- `detection/` is distributed under the Apache License 2.0 in
  `detection/LICENSE`.
- `segmentation/` is distributed under GPL-3.0 in `segmentation/LICENSE`.
- `CFMNet_manuscript.pdf` is not covered by these software licenses.

## Microsoft MIT-Licensed Code

Source files carrying the Microsoft copyright and MIT license header retain
those notices. The MIT license text is available in the root `LICENSE`.

## MMRotate

The detection subproject is derived from MMRotate 0.3.4.

- Repository: https://github.com/open-mmlab/mmrotate
- License: Apache-2.0
- Copyright: OpenMMLab contributors

See `detection/LICENSE`, `detection/NOTICE`, and `detection/UPSTREAM.md`.

## GeoSeg and UNetFormer

The segmentation subproject is derived from GeoSeg and its UNetFormer
implementation.

- Repository: https://github.com/WangLibo1995/GeoSeg
- License: GPL-3.0
- Copyright: GeoSeg contributors

See `segmentation/LICENSE` and `segmentation/UPSTREAM.md`.

## External Dependencies

Python packages installed through the task-specific requirements files are not
vendored. Each dependency remains subject to its own license.
