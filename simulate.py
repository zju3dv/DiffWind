"""Run DiffWind's coupled LBM–MPM forward simulation and Gaussian rendering."""

import shutil
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import taichi as ti
import torch
import torchvision
from plyfile import PlyData
from tqdm import tqdm

from HOME_LBM import HOME_LBM
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render_phygs
from scene import GaussianModel, Scene
from simulator import Estimator_Force
from utils.general_utils import safe_state


GRID_SIZE = 128
DOMAIN_SIZE = 4.0


def load_points(path):
    vertex = PlyData.read(str(path))["vertex"]
    points = np.column_stack((vertex["x"], vertex["y"], vertex["z"]))
    return torch.as_tensor(points, device="cuda", dtype=torch.float32).contiguous()


def normalize_points(surface_points, filling_points=None):
    if filling_points is None:
        points = surface_points
        surface_mask = torch.ones(len(points), dtype=torch.bool, device=points.device)
    else:
        points = torch.cat([surface_points, filling_points], dim=0)
        surface_mask = torch.zeros(len(points), dtype=torch.bool, device=points.device)
        surface_mask[: len(surface_points)] = True

    center = points.mean(dim=0)
    centered = points - center
    position_min = centered.min()
    position_max = centered.max()
    scale = position_max - position_min
    if scale <= 0:
        raise ValueError("Point cloud has zero spatial extent")
    shift = -position_min + scale * 0.5
    normalized = (centered + shift) / scale
    return normalized, surface_mask, center, scale, shift


def configure_fluid(wind_direction):
    viscosity = 1.5e-5
    dx = 1.0 / 32.0
    lattice_time = 1.0 / 32.0
    lattice_viscosity = viscosity / (dx * dx / lattice_time)
    solver = HOME_LBM(GRID_SIZE, GRID_SIZE, GRID_SIZE, lattice_viscosity)

    setters = (
        (wind_direction[0], solver.set_bc_vel_x0, solver.set_bc_vel_x1),
        (wind_direction[1], solver.set_bc_vel_y0, solver.set_bc_vel_y1),
        (wind_direction[2], solver.set_bc_vel_z0, solver.set_bc_vel_z1),
    )
    for component, positive_setter, negative_setter in setters:
        if component > 0:
            positive_setter(wind_direction)
        elif component < 0:
            negative_setter(wind_direction)
    solver.init_simulation()
    return solver


def occupancy_grid(points):
    indices = torch.floor(points * (GRID_SIZE / DOMAIN_SIZE)).long()
    indices = torch.clamp(indices, 0, GRID_SIZE - 1).cpu().numpy()
    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.int32)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
    return grid


def sanitize_state(state):
    positions, velocities, deformation, affine = state
    abnormal = torch.linalg.norm(velocities, dim=1) > 5.0
    if abnormal.any():
        velocities[abnormal] = 0.0
        identity = torch.eye(3, device=deformation.device, dtype=deformation.dtype)
        deformation[abnormal] = identity
        affine[abnormal] = identity
        print("Reset {} unstable particles".format(int(abnormal.sum())))
    return positions, velocities, deformation, affine


def prepare_output(root, save_state, save_fields):
    root.mkdir(parents=True, exist_ok=True)
    view_dirs = []
    for split in ("train", "test"):
        for index in range(4):
            path = root / "{}_view{}".format(split, index)
            path.mkdir(parents=True, exist_ok=True)
            view_dirs.append(path)
    if save_state:
        (root / "simulator_state").mkdir(exist_ok=True)
    if save_fields:
        (root / "force_field").mkdir(exist_ok=True)
        (root / "object_grid").mkdir(exist_ok=True)
    return view_dirs


def save_state(root, frame, state):
    names = ("xyz", "velocity", "deformation", "affine")
    for name, tensor in zip(names, state):
        torch.save(
            tensor.detach().cpu(),
            root / "simulator_state" / "{}_{:04d}.pt".format(name, frame),
        )


def encode_videos(view_dirs, fps):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg was not found; PNG frames were kept without MP4 encoding")
        return
    for directory in view_dirs:
        command = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(directory / "%05d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            str(directory / "output.mp4"),
        ]
        subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )


def run_simulation(
    dataset,
    pipeline,
    physics,
    gaussian_path,
    filling_path,
    wind_direction,
    wind_scale,
    render_views,
    frames,
    wind_frames,
    lbm_steps,
    warmup_steps,
    save_states,
    save_fields,
    encode_video,
):
    surface = load_points(gaussian_path)
    filling = load_points(filling_path) if filling_path is not None else None
    particles, surface_mask, center, scale, shift = normalize_points(surface, filling)

    expected_particles = sum(physics.part_counts)
    if expected_particles != len(particles):
        raise ValueError(
            "Config part_counts sum to {}, but the PLY assets contain {} particles".format(
                expected_particles, len(particles)
            )
        )

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(str(gaussian_path))
    scene = Scene(dataset, gaussians)
    cameras = scene.getTrainCameras()
    if (
        len(render_views) != 8
        or max(render_views) >= len(cameras)
        or min(render_views) < 0
    ):
        raise ValueError("--render_view needs eight valid orbit-camera indices")
    selected = [cameras[index] for index in render_views]

    output_root = Path(dataset.model_path) / "render_wind_couple"
    view_dirs = prepare_output(output_root, save_states, save_fields)
    fluid = configure_fluid(wind_direction)
    estimator = Estimator_Force(
        physics,
        "float32",
        init_vol=particles,
    )
    estimator.configure_forward_coupling(physics.static_part)

    background_color = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_color, dtype=torch.float32, device="cuda")
    state = None

    with torch.no_grad():
        for frame in tqdm(range(frames), desc="Forward simulation"):
            if 0 < frame < wind_frames:
                mpm_momentum = estimator.simulator.get_grid_v() / np.sqrt(wind_scale)
                fluid.load_mpm_velo(mpm_momentum)
                for _ in range(lbm_steps):
                    fluid.step()
                fluid.cal_wind_force_field()
                force_numpy = (
                    fluid.get_wind_force_field().to_numpy() * wind_scale
                ).astype(np.float32)
            else:
                force_numpy = np.zeros(
                    (GRID_SIZE, GRID_SIZE, GRID_SIZE, 3), dtype=np.float32
                )

            estimator.set_force_field(force_numpy)
            estimator.initialize()
            if state is not None:
                state = sanitize_state(state)
                estimator.load_simulator_state(*state)
                state = estimator.forward_sim(1)
            else:
                state = estimator.forward_sim(0)
            if not isinstance(state, (tuple, list)) or len(state) != 4:
                raise RuntimeError(
                    "MPM simulation failed its stability check; increase --mpm_iter_cnt"
                )

            if save_states:
                save_state(output_root, frame, state)

            positions, _, deformation, _ = state
            grid = occupancy_grid(positions)
            fluid.load_geo(grid)
            if frame == 0:
                for _ in range(warmup_steps):
                    fluid.step()

            if save_fields:
                np.save(
                    output_root / "force_field" / "force_{:04d}.npy".format(frame),
                    force_numpy,
                )
                np.save(
                    output_root / "object_grid" / "occupancy_{:04d}.npy".format(frame),
                    grid,
                )

            surface_positions = positions[surface_mask]
            surface_deformation = deformation[surface_mask]
            world_positions = surface_positions * scale - shift + center
            displacement = world_positions - gaussians.get_xyz.detach()
            for view_index, camera in enumerate(selected):
                result = render_phygs(
                    camera,
                    gaussians,
                    pipeline,
                    background,
                    displacement,
                    0.0,
                    0.0,
                    surface_deformation,
                    False,
                )
                torchvision.utils.save_image(
                    result["render"], view_dirs[view_index] / "{:05d}.png".format(frame)
                )

    if encode_video:
        encode_videos(view_dirs, physics.fps)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--gs_ply", required=True)
    parser.add_argument("--fill_ply")
    parser.add_argument("--wind_dir", type=float, nargs=3, required=True)
    parser.add_argument("--wind_scale", type=float, default=1.0)
    parser.add_argument("--render_view", type=int, nargs=8, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument(
        "--wind_frames",
        type=int,
        default=50,
        help="Apply coupled LBM wind only before this frame, matching the final release runs",
    )
    parser.add_argument(
        "--mpm_iter_cnt",
        type=int,
        help="Override the config's MPM substeps per rendered frame",
    )
    parser.add_argument("--lbm_steps", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=400)
    parser.add_argument("--save_state", action="store_true")
    parser.add_argument("--save_fields", action="store_true")
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--device_memory_fraction", type=float, default=0.7)
    model = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    args, physics = get_combined_args(parser)
    if physics is None:
        parser.error("The JSON config must contain a 'physics' object")

    if args.mpm_iter_cnt is not None:
        physics.mpm_iter_cnt = args.mpm_iter_cnt
    safe_state(args.quiet)
    dataset = model.extract(args)
    Path(dataset.model_path).mkdir(parents=True, exist_ok=True)

    ti.init(
        arch=ti.cuda,
        debug=False,
        fast_math=False,
        device_memory_fraction=args.device_memory_fraction,
    )
    filling_path = None if args.fill_ply in (None, "None") else Path(args.fill_ply)
    start = time.time()
    run_simulation(
        dataset,
        pipeline_params.extract(args),
        physics,
        Path(args.gs_ply),
        filling_path,
        args.wind_dir,
        args.wind_scale,
        args.render_view,
        args.frames,
        args.wind_frames,
        args.lbm_steps,
        args.warmup_steps,
        args.save_state,
        args.save_fields,
        not args.no_video,
    )
    print("Completed in {:.1f} seconds".format(time.time() - start))


if __name__ == "__main__":
    main()
