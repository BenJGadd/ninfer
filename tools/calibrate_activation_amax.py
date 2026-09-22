"""Record per-site activation max-abs for Qwen3.5 Dense checkpoints (NVFP4 W4A4 calibration).

The engine's W4A4 route quantizes each linear's input per token with E4M3FN block scales and one
site-level FP32 divisor d_x (src/ops/linear/nvfp4/nvfp4_codec.cuh quantize_nvfp4_k16):
scale = e4m3_satfinite(d_x * max_abs_block / 6). A block whose max-abs exceeds 448 * 6 / d_x
saturates its scale and clips, so d_x = 448 * 6 / A with A the largest activation the site
sees. This tool measures A with the HF implementation: forward pre-hooks on the modules that
consume each NInfer input site, over whole windows (every window starts a fresh sequence, so
attention-sink tokens are included), and writes {"sites": {site: amax}} for
tools/convert/overrides/qwen3_5_9b_nvfp4_a4.py.

    python -m tools.calibrate_activation_amax --model ../models-src/Qwen3.5-9B \
        --text eval/corpora/perplexity-1m/data/wikitext/01.txt ... \
        --window 4096 --tokens-per-text 32768 --out calibration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# HF module suffix -> NInfer input site suffix (tools/convert/qwen3_5.py names).
SITES = {
    "linear_attn.in_proj_qkv": "mixer_input",
    "linear_attn.out_proj": "gdn/gated_output",
    "self_attn.q_proj": "mixer_input",
    "self_attn.o_proj": "attention/gated_output",
    "mlp.gate_proj": "ffn_input",
    "mlp.down_proj": "mlp/product",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", nargs="+", required=True, help="UTF-8 text files")
    parser.add_argument("--window", type=int, default=4096)
    parser.add_argument("--tokens-per-text", type=int, default=32768)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    amax: dict[str, float] = {}
    hooks = []
    for name, module in model.named_modules():
        for suffix, site in SITES.items():
            if not name.endswith(suffix):
                continue
            prefix = name[: -len(suffix)]
            if not prefix.startswith("model.language_model.layers."):
                continue  # MTP and vision are not NVFP4 sites
            layer = prefix.split(".")[3]
            key = f"text/layers/{layer}/{site}"

            def hook(_module, inputs, key=key):
                value = float(inputs[0].detach().abs().amax())
                if value > amax.get(key, 0.0):
                    amax[key] = value

            hooks.append(module.register_forward_pre_hook(hook))
    if not hooks:
        raise SystemExit("no calibration sites found; is this a Qwen3.5 Dense checkpoint?")

    seen = 0
    with torch.inference_mode():
        for path in args.text:
            text = Path(path).read_text(encoding="utf-8")
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
            ids = ids[: args.tokens_per_text]
            for begin in range(0, len(ids), args.window):
                chunk = ids[begin : begin + args.window]
                if len(chunk) < 64:
                    break
                model(input_ids=chunk[None].cuda(), use_cache=False)
                seen += len(chunk)
            print(f"{path}: {min(len(ids), args.tokens_per_text)} tokens, total {seen}", flush=True)

    sites = dict(sorted(amax.items()))
    if any(not (v > 0) for v in sites.values()):
        raise SystemExit("a site recorded a non-positive max-abs")
    Path(args.out).write_text(
        json.dumps({"model": str(args.model), "tokens": seen, "window": args.window,
                    "texts": args.text, "sites": sites}, indent=1)
    )
    print(f"wrote {args.out}: {len(sites)} sites over {seen} tokens")


if __name__ == "__main__":
    main()
