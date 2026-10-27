"""Native exported-PNG PSNR, matching WD-Objects' per-image averaging."""
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .io import write_json

PAPER_PSNR = dict(
    potted_plant=52.54,
    ficus=48.82,
    sweater=41.72,
    pants=58.38,
    alocasia=44.94,
    flag=41.86,
    dress=41.76,
)


def image_psnr(reference, prediction):
    with Image.open(reference) as image:
        target = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    with Image.open(prediction) as image:
        actual = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    if target.shape != actual.shape:
        raise ValueError("Prediction and target resolutions differ")
    mse = float(np.mean(np.square(target - actual), dtype=np.float64))
    return float("inf") if mse == 0 else -10 * math.log10(mse)


def score(gt, rendered, output, scene, frames=48, split="test"):
    if split not in ("train", "test") or not 1 <= frames <= 48:
        raise ValueError("Invalid evaluation split or frame range")
    if split == "test" and frames != 48:
        raise ValueError("The paper comparison requires all 48 frames")
    rows, per_view = [], {}
    for view in range(4):
        label = "%s_view%d" % (split, view)
        values = []
        for frame in range(1, frames + 1):
            relative = Path(label) / ("%05d.png" % frame)
            value = image_psnr(Path(gt) / relative, Path(rendered) / relative)
            values.append(value)
            rows.append(dict(view=label, frame=frame, psnr=value))
        per_view[label] = float(np.mean(values))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["view", "frame", "psnr"])
        writer.writeheader()
        writer.writerows(rows)
    result = dict(
        scene=scene,
        split=split,
        frames=frames,
        image_count=len(rows),
        psnr=float(np.mean([r["psnr"] for r in rows])),
        per_view=per_view,
        image_domain="native exported RGB PNG",
        reduction="mean per-image PSNR",
    )
    if split == "test":
        result.update(
            paper_psnr=PAPER_PSNR[scene],
            delta_vs_paper=result["psnr"] - PAPER_PSNR[scene],
        )
    write_json(output / "metrics.json", result)
    return result
