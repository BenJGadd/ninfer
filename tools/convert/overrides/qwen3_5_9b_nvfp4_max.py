"""The maximum-performance 9B NVFP4 artifact: A4 sites plus the NVFP4 head and MTP stem/MLP.

`--override` takes one file, so this composes the two overrides that the port measured
separately (docs/maintainer/qwen3.5-9b-port.md §8.1-8.2). NINFER_ACTIVATION_AMAX names the
calibration JSON as for qwen3_5_9b_nvfp4_a4.py.
"""

from __future__ import annotations

from tools.convert.overrides.qwen3_5_9b_nvfp4_a4 import configure as configure_a4
from tools.convert.overrides.qwen3_5_9b_nvfp4_head_mtp import configure as configure_head_mtp


def configure(model, recipe, sources) -> None:
    configure_head_mtp(model, recipe, sources)
    configure_a4(model, recipe, sources)
