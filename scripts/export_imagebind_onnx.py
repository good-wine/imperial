"""Export ImageBind vision tower to ONNX (joint image/text/audio space).

Run once on a CONNECTED station that has the 4.8 GB checkpoint and a clone of
facebookresearch/ImageBind. The upstream commit is NOT optional: the
checkpoint is pinned by digest, but the clone supplies the model definition
that decides the exported graph, so the export refuses to run until you name
the revision it is building from (and checks that the clone agrees).

    git clone --filter=blob:none --sparse <repo> /tmp/ib_repo
    git -C /tmp/ib_repo sparse-checkout set imagebind
    git -C /tmp/ib_repo rev-parse HEAD          # -> pass as --upstream-commit
    python scripts/export_imagebind_onnx.py --weights /tmp/imagebind_huge.pth \
                                            --repo /tmp/ib_repo \
                                            --upstream-commit <sha>

The resolved commit is written back into `models/registry.json` under the
model's `source`, so the graph's origin is recorded next to its digest.

Then copy `models/imagebind_vision.onnx` + `models/imagebind_vision.onnx.data`
to the air-gapped host and sign it (`scripts/sign_models.py --sign-all`).
The pipeline consumes it in `src/embeddings.py` (1024-d, L2-normalized);
Qdrant keeps it in a SEPARATE `provenance_ib` collection so the CLIP-512
collection needs no migration.

Scope honesty: vision tower only. The audio tower needs torchaudio with a
loadable native extension (the stock wheel links libcudart and is unusable on
CPU-only hosts), so audio<->image joint search stays a documented gap.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Names imagebind imports at package init that a CPU-only export host lacks.
# None of them are exercised by the vision tower; they are stubbed so the
# import succeeds, and any real use would fail loudly.
_STUB_MODULES = ("pkg_resources", "torchaudio", "pytorchvideo", "torchvision")
_STUB_SUBMODULES = {
    "pytorchvideo.transforms": (),
    "pytorchvideo.data": (),
    "pytorchvideo.data.clip_sampling": ("ConstantClipsPerVideoSampler",),
    "pytorchvideo.data.encoded_video": ("EncodedVideo",),
    "torchvision.transforms": (),
}


def _make_module(name: str, attrs: tuple[str, ...] = ()) -> types.ModuleType:
    mod = types.ModuleType(name)
    for attr in attrs:
        setattr(mod, attr, type(attr, (), {}))
    sys.modules[name] = mod
    if "." in name:
        parent, child = name.rsplit(".", 1)
        if parent in sys.modules:
            setattr(sys.modules[parent], child, mod)
    return mod


def _stub_timm() -> None:
    """timm is used for trunc_normal_ (re-export of torch's) + DropPath.

    DropPath is stochastic depth: identity at eval time, which is the only
    mode this export runs in.
    """
    try:
        __import__("timm.layers")
        return
    except Exception:  # any failure means the real one is unusable here
        pass
    import torch
    import torch.nn.init as _tni

    class _DropPath(torch.nn.Module):
        def __init__(self, drop_prob: float = 0.0) -> None:
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x: Any) -> Any:  # eval => identity
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep = 1 - self.drop_prob
            mask = torch.zeros_like(x)
            mask[:, 0] = keep + torch.rand_like(x[:, 0])
            return x * mask.floor_() / keep

    _make_module("timm", ("layers",))
    layers = _make_module("timm.layers", ("trunc_normal_", "DropPath"))
    layers.trunc_normal_ = _tni.trunc_normal_
    layers.DropPath = _DropPath


def _prepare_imports() -> None:
    """Make `import imagebind.models` work on a CPU-only export host."""
    for name in _STUB_MODULES:
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except Exception:  # ImportError, or a broken native wheel
            sys.modules[name] = types.ModuleType(name)
    with contextlib.suppress(ImportError):
        from pytorchvideo import transforms  # noqa: F401  (real or stub)
    for name, attrs in _STUB_SUBMODULES.items():
        if name not in sys.modules:
            _make_module(name, attrs)
    _stub_timm()


def _consolidate_external_data(model_path: Path) -> Path:
    """Collapse the per-tensor sidecars ONNX emitted into ONE data blob.

    ViT-Huge exceeds the 2 GB protobuf limit, so torch spills each of the 392
    initializers into its own file next to the model — hundreds of untracked
    names in `models/`. Re-save with every tensor packed into a single
    `<model>.data` sibling; onnxruntime then loads one file.
    """
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    m = onnx.load(str(model_path))
    if not any(t.HasField("data_location") for t in m.graph.initializer):
        return model_path
    blob = model_path.with_suffix(model_path.suffix + ".data")
    convert_model_to_external_data(
        m, all_tensors_to_one_file=True, location=blob.name, size_threshold=1024
    )
    onnx.save(m, str(model_path))
    # Delete ONLY the tensors this export spilled (names torch derived from the
    # wrapper's submodules); never touch neighbouring signed models.
    prefixes = ("pre.", "trunk.", "head.", "post.", "onnx__")
    for p in sorted(model_path.parent.iterdir()):
        if p.is_file() and p != blob and p.name.startswith(prefixes):
            p.unlink()
    print(f"consolidated external data -> {blob.name} ({blob.stat().st_size >> 20} MB)")
    return model_path


def _build_tower(weights: str) -> tuple[Any, Any]:
    """Load imagebind_huge + wrap its vision path as real submodules.

    The wrapper holds the tower parts as child modules instead of closing over
    the parent model: TorchScript can only trace attribute access on children,
    otherwise external tensors leak into the graph as grad-requiring constants.
    """
    import torch
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType

    print("building imagebind_huge (random init, weights loaded from disk)…")
    model = imagebind_model.imagebind_huge(pretrained=False).eval()
    # weights_only=False: the checkpoint is a plain state_dict, but torch's
    # unpickler still runs it. This is an OFFLINE export tool run by the
    # operator against a checkpoint they fetched and verified themselves —
    # never on the analysis path, which only loads signed ONNX.
    state = torch.load(weights, map_location="cpu", weights_only=False)  # nosec B614
    # strict=False: modalities we do not export (text/audio/depth/thermal/imu)
    # may not all match, and only the vision path matters here. Report it.
    keys = model.load_state_dict(state, strict=False)
    print(f"missing={len(keys.missing_keys)} unexpected={len(keys.unexpected_keys)}")

    class VisionTower(torch.nn.Module):
        def __init__(self, m: Any) -> None:
            super().__init__()
            self.pre = m.modality_preprocessors[ModalityType.VISION]
            self.trunk = m.modality_trunks[ModalityType.VISION]
            self.head = m.modality_heads[ModalityType.VISION]
            self.post = m.modality_postprocessors[ModalityType.VISION]

        def forward(self, pixel_values: Any) -> Any:
            pre = self.pre(**{ModalityType.VISION: pixel_values})
            x = self.trunk(**pre["trunk"])
            x = self.head(x, **pre["head"])
            return self.post(x)

    return VisionTower(model).eval(), torch


def _resolve_upstream_commit(repo: Path, declared: str) -> str:
    """Make the upstream code revision explicit, and verify it.

    Reproducibility here rests on two things, and it is worth being precise
    about which is which: the CHECKPOINT bytes are pinned by SHA-256 in the
    registry (authoritative — same weights in, same tensors out), while the
    upstream clone supplies the model DEFINITION that decides the graph
    structure. A `--depth 1` clone of `main` silently floats that second
    input, so the export refuses to run unless the operator states the commit
    and the working tree agrees with it.
    """
    import subprocess  # nosec B404 - fixed argv, no shell, operator-run export script

    if not declared:
        raise SystemExit(
            "refusing to export: pass --upstream-commit <sha> (the ImageBind clone's HEAD). "
            "The checkpoint is pinned by digest, but the model definition that shapes the "
            "graph comes from this clone and must be identified."
        )
    try:
        # argv is fully static apart from `repo`, which is passed as its own
        # argument (never interpolated into a shell string) and comes from the
        # operator running this export by hand.
        head = subprocess.run(  # noqa: S603  # nosec B603
            [shutil.which("git") or "git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"could not read the git HEAD of {repo}: {exc}") from exc
    if not declared.startswith(head[:12]) and head[:12] not in declared:
        raise SystemExit(
            f"upstream commit mismatch: --upstream-commit {declared} but {repo} is at {head}. "
            "Check out the pinned revision or pass the one you actually built."
        )
    return head


def _record_source(out: Path, commit: str, checkpoint: Path) -> None:
    """Write the export provenance next to the pin, so the registry says where
    the graph came from instead of leaving it to memory."""
    import json

    reg = Path("models/registry.json")
    if not reg.is_file():
        print(f"note: {reg} not found; not recording upstream provenance")
        return
    doc = json.loads(reg.read_text(encoding="utf-8"))
    touched = False
    for m in doc.get("models", []):
        if Path(m.get("path", "")).name == out.name:
            m["source"] = {
                "repo": "https://github.com/facebookresearch/ImageBind",
                "commit": commit,
                "checkpoint": checkpoint.name,
                "note": (
                    "checkpoint identity is the registry sha256 pin; "
                    "this commit defines the exported graph"
                ),
            }
            touched = True
    if not touched:
        print(f"note: no registry entry for {out.name}; not recording provenance")
        return
    reg.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"recorded upstream commit {commit[:12]} in {reg}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export ImageBind vision tower")
    ap.add_argument("--weights", default="/tmp/opencode/ib/imagebind_huge.pth")
    ap.add_argument("--repo", default="/tmp/opencode/ib_repo")
    ap.add_argument("--out", default="models/imagebind_vision.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--upstream-commit",
        default=os.environ.get("IMAGEBIND_UPSTREAM_COMMIT", ""),
        help="git SHA of the ImageBind clone that defines the exported graph",
    )
    args = ap.parse_args()

    commit = _resolve_upstream_commit(Path(args.repo), args.upstream_commit)
    print(f"upstream ImageBind commit: {commit}")

    sys.path.insert(0, args.repo)
    _prepare_imports()

    tower, torch = _build_tower(args.weights)
    tower.requires_grad_(False)
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        ref = tower(dummy)
    assert tuple(ref.shape) == (1, 1024), ref.shape
    print(f"torch reference: {tuple(ref.shape)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():  # export traces a graph: grads off, or tensors leak as constants
        torch.onnx.export(
            tower,
            (dummy,),
            str(out),
            input_names=["pixel_values"],
            output_names=["embedding"],
            opset_version=args.opset,
            dynamo=False,
            dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        )
    _consolidate_external_data(out)

    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["pixel_values"]
    got = sess.run(None, {"pixel_values": np.zeros((1, 3, 224, 224), np.float32)})[0]
    maxdiff = float(np.abs(ref.numpy() - got).max())
    print(f"onnx parity max|Δ|={maxdiff:.2e}")
    assert maxdiff < 1e-4, "torch/onnxruntime divergence"
    print(f"verified {out} (+ {out.name}.data)")
    _record_source(out, commit, Path(args.weights))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
