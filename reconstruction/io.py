"""Atomic run records, renderer setup, and checkpoint I/O."""
import contextlib
import fcntl
import json
import os
import sys
from pathlib import Path

import numpy as np

from .config import ROOT, sha256


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_array(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        if isinstance(value, dict):
            np.savez_compressed(handle, **value)
        else:
            np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def load_state(path):
    with np.load(str(path), allow_pickle=False) as data:
        return {key: data[key] for key in ("x", "v", "F", "C")}


@contextlib.contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process is using " + str(output))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare_renderer(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    bundled = ROOT / "third_party" / "diff-gaussian-rasterization"
    if list((bundled / "diff_gaussian_rasterization").glob("_C*.so")):
        sys.path.insert(0, str(bundled))
    import diff_gaussian_rasterization as rasterizer

    if getattr(rasterizer, "RGB_ADJOINT_VERSION", None) != 3:
        raise RuntimeError(
            "Rebuild this repository's rasterizer: python -m pip install "
            "./third_party/diff-gaussian-rasterization --no-cache-dir --force-reinstall --no-deps"
        )
    return sha256(rasterizer._C.__file__)


def code_manifest():
    paths = []
    for directory in (
        "reconstruction",
        "simulator",
        "scene",
        "gaussian_renderer",
        "utils",
        "third_party/diff-gaussian-rasterization",
    ):
        paths.extend(
            p
            for p in (ROOT / directory).rglob("*")
            if p.suffix in (".py", ".cu", ".h", ".cpp") and "build" not in p.parts
        )
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def export_image(image, path):
    import torch
    from PIL import Image

    if not torch.isfinite(image).all():
        raise FloatingPointError("Nonfinite rendered pixels")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = (image.detach() * 255 + 0.5).clamp(0, 255).to(torch.uint8)
    Image.fromarray(rgb.permute(1, 2, 0).cpu().numpy()).save(str(path))


def render_views(engine, state, root, frame, split):
    import torch

    offset = 0 if split == "train" else 4
    for view in range(4):
        with torch.no_grad():
            image, _, _ = engine.render(state, view + offset)
        export_image(image, root / ("%s_view%d" % (split, view)) / ("%05d.png" % frame))
