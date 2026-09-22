"""Registered Qwen3.5-9B geometry: eight primitive dimensions and every derived shape.

This is the single source of truth for the converter target. ``inventory.py``,
``recipe.py``, ``verify.py`` and the tests derive all tensor shapes from the
``Geometry`` returned by :func:`registered`; nothing else in the package spells a
matrix dimension.  The C++ target repeats the same eight primitives in
``src/targets/qwen3_5_9b/impl/config.h``; ``tests/convert/qwen3_5_9b/test_config_h.py``
checks that copy against this one.

``from_hf_config`` builds the same structure from a checkpoint's ``config.json`` so
conversion can prove the source checkpoint has the registered shape instead of
trusting its name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Geometry:
    """Primitive text-decoder dimensions; everything else is a property."""

    hidden: int
    layers: int
    intermediate: int
    vocab_rows: int
    query_heads: int
    kv_heads: int
    head_dim: int
    gdn_key_heads: int
    gdn_value_heads: int
    gdn_head_dim: int = 128
    gdn_conv_kernel: int = 4
    full_attention_interval: int = 4
    vision_hidden: int = 1152
    vision_merger_input: int = 4608

    # --- hybrid layer schedule --------------------------------------------------
    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        interval = self.full_attention_interval
        return tuple(range(interval - 1, self.layers, interval))

    @property
    def gdn_layers(self) -> tuple[int, ...]:
        full = set(self.full_attention_layers)
        return tuple(layer for layer in range(self.layers) if layer not in full)

    # --- full attention -----------------------------------------------------------
    @property
    def query_size(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def query_projection_rows(self) -> int:
        """``q_proj`` rows: query and per-head output gate interleaved per head."""
        return 2 * self.query_size

    @property
    def query_key_rows(self) -> int:
        return self.query_size + self.kv_size

    @property
    def mtp_attention_input_rows(self) -> int:
        return 2 * self.query_size + 2 * self.kv_size

    # --- gated DeltaNet -----------------------------------------------------------
    @property
    def key_dim(self) -> int:
        return self.gdn_key_heads * self.gdn_head_dim

    @property
    def value_dim(self) -> int:
        return self.gdn_value_heads * self.gdn_head_dim

    @property
    def convolution_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def gdn_query_key_rows(self) -> int:
        return 2 * self.key_dim

    @property
    def gdn_value_z_rows(self) -> int:
        return 2 * self.value_dim

    # --- MLP / MTP ------------------------------------------------------------------
    @property
    def gate_up_rows(self) -> int:
        return 2 * self.intermediate

    @property
    def mtp_input_rows(self) -> int:
        return 2 * self.hidden

    # --- checkpoint comparison ------------------------------------------------------
    @classmethod
    def from_hf_config(cls, config: Mapping[str, object]) -> "Geometry":
        text = config["text_config"]
        vision = config["vision_config"]
        if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
            raise ValueError("config.json must contain text_config and vision_config")
        return cls(
            hidden=int(text["hidden_size"]),
            layers=int(text["num_hidden_layers"]),
            intermediate=int(text["intermediate_size"]),
            vocab_rows=int(text["vocab_size"]),
            query_heads=int(text["num_attention_heads"]),
            kv_heads=int(text["num_key_value_heads"]),
            head_dim=int(text["head_dim"]),
            gdn_key_heads=int(text["linear_num_key_heads"]),
            gdn_value_heads=int(text["linear_num_value_heads"]),
            gdn_head_dim=int(text["linear_key_head_dim"]),
            gdn_conv_kernel=int(text["linear_conv_kernel_dim"]),
            full_attention_interval=int(text["full_attention_interval"]),
            vision_hidden=int(vision["hidden_size"]),
            vision_merger_input=int(vision["hidden_size"])
            * int(vision["spatial_merge_size"]) ** 2,
        )

    def text_config_expectations(self) -> dict[str, object]:
        """The ``text_config`` members a matching checkpoint must carry."""
        return {
            "num_hidden_layers": self.layers,
            "full_attention_interval": self.full_attention_interval,
            "hidden_size": self.hidden,
            "intermediate_size": self.intermediate,
            "vocab_size": self.vocab_rows,
            "num_attention_heads": self.query_heads,
            "num_key_value_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "linear_num_key_heads": self.gdn_key_heads,
            "linear_num_value_heads": self.gdn_value_heads,
            "linear_key_head_dim": self.gdn_head_dim,
            "linear_value_head_dim": self.gdn_head_dim,
            "linear_conv_kernel_dim": self.gdn_conv_kernel,
        }


def registered() -> Geometry:
    """The Qwen/Qwen3.5-9B shape this target is compiled for."""
    return Geometry(
        hidden=4096,
        layers=32,
        intermediate=12288,
        vocab_rows=248320,
        query_heads=16,
        kv_heads=4,
        head_dim=256,
        gdn_key_heads=16,
        gdn_value_heads=32,
    )


GEOMETRY = registered()

__all__ = ["GEOMETRY", "Geometry", "registered"]
