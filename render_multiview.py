"""Render a static Gaussian PLY from deterministic orbit cameras."""

from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--gs_ply", required=True)
    model = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    args, _ = get_combined_args(parser)
    safe_state(args.quiet)
    dataset = model.extract(args)
    output_dir = Path(dataset.model_path) / "multiview"
    output_dir.mkdir(parents=True, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(args.gs_ply)
    scene = Scene(dataset, gaussians)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    pipeline = pipeline_params.extract(args)

    with torch.no_grad():
        for index, camera in enumerate(
            tqdm(scene.getTrainCameras(), desc="Multiview render")
        ):
            displacement = torch.zeros_like(gaussians.get_xyz)
            result = render(
                camera, gaussians, pipeline, background, displacement, 0.0, 0.0, False
            )
            torchvision.utils.save_image(
                result["render"], output_dir / "camera_{:03d}.png".format(index)
            )


if __name__ == "__main__":
    main()
