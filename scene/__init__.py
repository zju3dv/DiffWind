"""Camera construction for DiffWind simulation and multiview rendering."""

import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos


class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    image: Image.Image
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    flow: object


def _look_at_pose(up, target, position):
    """Return an OpenCV-style camera-to-world matrix."""
    up = up / np.linalg.norm(up)
    z_axis = target - position
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = -up
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = position
    return pose


def _orbit_poses(count, radius, elevation_deg, adjust_camera=False):
    azimuths = np.deg2rad(np.linspace(0.0, 360.0, count, endpoint=False))
    elevation = np.deg2rad(elevation_deg)
    positions = np.stack(
        [
            radius * np.cos(elevation) * np.cos(azimuths),
            radius * np.cos(elevation) * np.sin(azimuths),
            np.full_like(azimuths, radius * np.sin(elevation)),
        ],
        axis=-1,
    )
    poses = np.stack(
        [
            _look_at_pose(
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                position,
            )
            for position in positions
        ]
    )
    if adjust_camera:
        z_flip = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.float32)
        poses = np.stack([z_flip @ pose for pose in poses])
    return poses


def _make_orbit_cameras(args):
    count = args.mv_num
    resolution = args.camera_resolution
    width = args.camera_width or resolution
    height = args.camera_height or resolution
    fov = 2.0 * math.atan(resolution / (2.0 * args.camera_focal))
    blank = Image.new("RGB", (width, height), color=(255, 255, 255))
    world_to_cameras = np.linalg.inv(
        _orbit_poses(count, args.radius_set, args.elevation_set, args.adjust_cam)
    )
    cameras = []
    for index, pose in enumerate(world_to_cameras):
        cameras.append(
            CameraInfo(
                uid=index,
                R=pose[:3, :3].T,
                T=pose[:3, 3],
                FovY=fov,
                FovX=fov,
                image=blank,
                image_path="",
                image_name=f"orbit_{index:03d}",
                width=width,
                height=height,
                fid=0.0,
                flow=None,
            )
        )
    return cameras


def _make_asset_cameras(path, args):
    """Load Blender-format camera transforms without requiring source images."""
    with path.open("r") as handle:
        data = json.load(handle)

    width = int(data.get("width", args.camera_width))
    height = int(data.get("height", args.camera_height))
    if width <= 0 or height <= 0:
        raise ValueError(
            "Camera width and height must be stored in cameras.json or the config"
        )
    fov_x = float(data["camera_angle_x"])
    focal = width / (2.0 * math.tan(fov_x / 2.0))
    fov_y = 2.0 * math.atan(height / (2.0 * focal))
    blank = Image.new("RGB", (width, height), color=(255, 255, 255))

    cameras = []
    for index, frame in enumerate(data["frames"]):
        camera_to_world = np.asarray(frame["transform_matrix"], dtype=np.float32).copy()
        # Convert Blender/OpenGL axes (Y up, Z back) to COLMAP axes.
        camera_to_world[:3, 1:3] *= -1.0
        world_to_camera = np.linalg.inv(camera_to_world)
        cameras.append(
            CameraInfo(
                uid=index,
                R=world_to_camera[:3, :3].T,
                T=world_to_camera[:3, 3],
                FovY=fov_y,
                FovX=fov_x,
                image=blank,
                image_path=str(path),
                image_name="camera_{:03d}".format(index),
                width=width,
                height=height,
                fid=0.0,
                flow=None,
            )
        )
    return cameras


class Scene:
    """A Gaussian object observed by deterministic orbit cameras."""

    gaussians: GaussianModel

    def __init__(self, args, gaussians, render_view=None, **_unused):
        del render_view  # Camera selection is performed by the caller.
        self.model_path = args.model_path
        self.gaussians = gaussians
        camera_path = Path(args.source_path) / "cameras.json"
        camera_infos = (
            _make_asset_cameras(camera_path, args)
            if camera_path.is_file()
            else _make_orbit_cameras(args)
        )
        self.train_cameras = {1.0: cameraList_from_camInfos(camera_infos, 1.0, args)}
        self.test_cameras = {1.0: self.train_cameras[1.0]}

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
