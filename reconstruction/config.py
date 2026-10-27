"""Portable reconstruction presets and input validation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = ("potted_plant", "ficus", "sweater", "pants", "alocasia", "flag", "dress")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_config(name, data_root):
    if name not in SCENES:
        raise ValueError("Unknown scene: " + name)
    path = ROOT / "configs" / "reconstruction" / (name + ".json")
    settings = json.loads(path.read_text())
    for asset, expected in settings["asset_sha256"].items():
        if sha256(ROOT / asset) != expected:
            raise ValueError("Asset differs from the reconstruction preset: " + asset)
    camera = settings["camera"]
    if len(camera["views"]) != 8 or len(set(camera["views"])) != 8:
        raise ValueError("Expected four training and four distinct held-out cameras")
    physics = settings["physics"]
    if len({len(physics[k]) for k in ("part_counts", "E_list", "poisson_list")}) != 1:
        raise ValueError("Material partitions have inconsistent lengths")
    return dict(
        settings=settings,
        config=path,
        gs=ROOT / settings["gaussians"],
        fill=ROOT / settings["filling"] if settings["filling"] else None,
        gt=Path(data_root).resolve() / settings["data_scene"] / "gt",
        camera_json=ROOT / camera["file"] if camera["kind"] == "json" else None,
        views=camera["views"],
        chunk=settings["cache_chunk"],
    )


def input_manifest(spec, frames):
    """Hash the scene configuration, assets, and training images."""
    from PIL import Image

    training = {}
    for view in range(4):
        for frame in range(1, frames + 1):
            relative = "train_view%d/%05d.png" % (view, frame)
            path = spec["gt"] / relative
            with Image.open(path) as image:
                if list(image.size) != spec["settings"]["camera"]["resolution"]:
                    raise ValueError("Unexpected image size: " + str(path))
            training[relative] = sha256(path)
    return dict(
        config=sha256(spec["config"]),
        assets=spec["settings"]["asset_sha256"],
        training_images=training,
    )
