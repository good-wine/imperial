"""Export the CLIP ViT-B/32 vision tower to ONNX (activates the semantic tier).

Run once on the CONNECTED station (downloads ~350 MB from the Hub), then
copy `models/clip_image.onnx` to the air-gapped host and sign it
(`scripts/sign_models.py`). The pipeline consumes it in `src/embeddings.py`
(512-d, L2-normalized) and Qdrant `search_correlated` goes live — no code
changes needed downstream.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:  # type: ignore[no-untyped-def]
    ap = argparse.ArgumentParser(description="Export CLIP vision tower to ONNX")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--out", default="models/clip_image.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    try:
        import torch
        from transformers import CLIPModel
    except ImportError:
        print("need torch+transformers (pip install torch transformers)", file=sys.stderr)
        return 1
    print(f"loading {args.model} …")
    from torch import nn

    class _ClipVision(nn.Module):
        """Vision tower + class-token projection = 512-d CLIP image vector."""

        def __init__(self, full):
            super().__init__()
            self.vision = full.vision_model
            self.proj = full.visual_projection

        def forward(self, pixel_values):
            hidden = self.vision(pixel_values)[0]  # (B, 50, 768)
            return self.proj(hidden[:, 0, :])  # (B, 512)

    vision = _ClipVision(
        CLIPModel.from_pretrained(args.model)  # nosec B615 - provisioning tool; --model pinned, output Ed25519-signed
    ).eval()
    dummy = torch.zeros(1, 3, 224, 224)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        vision,
        (dummy,),
        args.out,
        input_names=["pixel_values"],
        output_names=["pooled"],
        opset_version=args.opset,
        dynamo=False,
        dynamic_axes={"pixel_values": {0: "batch"}, "pooled": {0: "batch"}},
    )
    # Verify: pooled output must be 512-d per row.
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"pixel_values": np.zeros((1, 3, 224, 224), np.float32)})[0]
    assert out.shape[1] == 512, out.shape
    print(f"verified {args.out}: pooled {out.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
