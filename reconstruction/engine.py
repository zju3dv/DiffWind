"""Full position/deformation adjoint over the public MPM and Gaussian renderer."""
import json
from types import SimpleNamespace

import numpy as np
import taichi as ti
import torch
from PIL import Image
from plyfile import PlyData

from simulator.mpm_simulator_force import MPMSimulator_Force
from scene import _orbit_poses
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render_phygs
from .config import scene_config


def points(path):
    vertex = PlyData.read(str(path))["vertex"]
    return np.stack([vertex[k] for k in ("x", "y", "z")], axis=1).astype(np.float32)


def image_tensor(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1).cuda()


def zero_force():
    return np.zeros((128, 128, 128, 3), dtype=np.float32)


@ti.data_oriented
class ReconstructionEngine:
    def __init__(self, name, data_root, cache_dir):
        self.spec = scene_config(name, data_root)
        self.image_domain = "clipped"
        self.config = self.spec["settings"]
        self.phys = SimpleNamespace(**self.config["physics"])
        if self.phys.rho != 1000 or self.phys.material != 10:
            raise ValueError(
                "Inherited mass/material implementation needs a separate adapter"
            )
        self.substeps = self.phys.mpm_iter_cnt
        ti.init(
            arch=ti.cuda,
            debug=False,
            fast_math=False,
            device_memory_fraction=0.35,
            offline_cache=True,
            offline_cache_file_path=str(cache_dir),
        )
        torch.manual_seed(0)
        np.random.seed(0)
        gp = points(self.spec["gs"])
        combined = (
            np.concatenate([gp, points(self.spec["fill"])]) if self.spec["fill"] else gp
        )
        vol = torch.from_numpy(combined).cuda()
        self.ng, self.n = len(gp), len(combined)
        self.center = vol.mean(0)
        centered = vol - self.center
        self.scale = centered.max() - centered.min()
        self.shift = -centered.min() + self.scale * 0.5
        self.rest = ((centered + self.shift) / self.scale).cpu().numpy()
        self.initial = dict(
            x=self.rest.copy(),
            v=np.tile(np.asarray(self.phys.init_vel, np.float32), (self.n, 1)),
            F=np.tile(np.eye(3, dtype=np.float32), (self.n, 1, 1)),
            C=np.zeros((self.n, 3, 3), np.float32),
        )
        self.num = ti.field(ti.i32, shape=())
        self.dx = ti.field(ti.f32, shape=())
        self.inv_dx = ti.field(ti.f32, shape=())
        self.particle = ti.root.dynamic(ti.i, 2**30, 2**14)
        self.sim = MPMSimulator_Force(
            dtype=ti.f32,
            dt=1.0 / self.phys.fps / self.substeps,
            frame_dt=1.0 / self.phys.fps,
            n_particles=self.num,
            particle_layout=self.particle,
            dx=self.dx,
            inv_dx=self.inv_dx,
            args=self.phys,
            gravity=self.phys.gravity,
            material=self.phys.material,
            cuda_chunk_size=self.spec["chunk"],
        )
        self.num[None] = self.n
        self.dx[None] = self.phys.voxel_size
        self.inv_dx[None] = 1.0 / self.phys.voxel_size
        for kind, entry in self.phys.bc.items():
            if not kind.startswith("ground"):
                raise ValueError("Unsupported collider: " + kind)
            self.sim.add_surface_collider(*entry)
        self.sim.add_force_field(
            point=[1.0, 1.0, 1.0], size=[1.0, 1.0, 2.0], num_frames=1e10
        )
        self.sim.set_static_part_on_cuboid(**self.phys.static_part)
        mu = []
        lam = []
        for e, nu, count in zip(
            self.phys.E_list, self.phys.poisson_list, self.phys.part_counts
        ):
            mu.extend([e / (2 * (1 + nu))] * count)
            lam.extend([e * nu / ((1 + nu) * (1 - 2 * nu))] * count)
        if len(mu) != self.n or len(lam) != self.n:
            raise ValueError(
                "Material partitions must exactly cover the particle ordering"
            )
        self.mu = np.asarray(mu[: self.n], np.float32)
        self.lam = np.asarray(lam[: self.n], np.float32)
        self.cloud = GaussianModel(3)
        self.cloud.load_ply(str(self.spec["gs"]))
        for value in vars(self.cloud).values():
            if isinstance(value, torch.Tensor) and value.is_leaf:
                value.requires_grad_(False)
        self.pipeline = SimpleNamespace(
            convert_SHs_python=False, compute_cov3D_python=False, debug=False
        )
        self.background = torch.ones(3, device="cuda")
        width, height = Image.open(self.spec["gt"] / "train_view0/00001.png").size
        self.resolution = (width, height)
        calibration = self.config["camera"]
        if [width, height] != calibration["resolution"]:
            raise ValueError("Training images must use the scene native resolution")
        fovx, fovy = calibration["fovx"], calibration["fovy"]
        if self.spec["camera_json"]:
            camera = json.loads(self.spec["camera_json"].read_text())
            poses = []
            for frame in camera["frames"]:
                c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
                c2w[:3, 1:3] *= -1
                poses.append(c2w)
        else:
            poses = _orbit_poses(
                calibration["count"], calibration["radius"], calibration["elevation"]
            )
        self.cameras = []
        for i, view in enumerate(self.spec["views"]):
            w2c = np.linalg.inv(poses[view])
            self.cameras.append(
                Camera(
                    view,
                    w2c[:3, :3].T,
                    w2c[:3, 3],
                    fovx,
                    fovy,
                    torch.ones(3, height, width),
                    None,
                    str(view),
                    i,
                    data_device="cuda",
                )
            )
        self.metadata = dict(
            scene=name,
            gaussians=self.ng,
            particles=self.n,
            configured_part_counts=self.phys.part_counts,
            material_array_count=len(mu),
            actual_material_count=self.n,
            substeps=self.substeps,
            cache_chunk=self.spec["chunk"],
            resolution=list(self.resolution),
            camera_views=self.spec["views"],
            fovx=fovx,
            fovy=fovy,
            physics=self.config["physics"],
            image_domain=self.image_domain,
            initial_state="normalized PLY; configured initial velocity; identity F; zero C",
            static_particle_count=int(
                np.all(
                    np.abs(self.rest - np.asarray(self.phys.static_part["point"]))
                    < np.asarray(self.phys.static_part["size"]),
                    axis=1,
                ).sum()
            ),
        )

    @ti.kernel
    def initialize_particles(
        self,
        x: ti.types.ndarray(),
        v: ti.types.ndarray(),
        f: ti.types.ndarray(),
        c: ti.types.ndarray(),
        mu: ti.types.ndarray(),
        lam: ti.types.ndarray(),
    ):
        for p in range(self.num[None]):
            self.sim.p_vol[p] = (self.dx[None] * 0.5) ** 3
            self.sim.p_mass[p] = self.sim.p_vol[p] * 1000.0
            self.sim.mu[p] = mu[p]
            self.sim.lam[p] = lam[p]
            for i in ti.static(range(3)):
                self.sim.x[p, 0][i] = x[p, i]
                self.sim.v[p, 0][i] = v[p, i]
                for j in ti.static(range(3)):
                    self.sim.F[p, 0][i, j] = f[p, i, j]
                    self.sim.C[p, 0][i, j] = c[p, i, j]

    @ti.kernel
    def seed_state_grad(self, gx: ti.types.ndarray(), gf: ti.types.ndarray()):
        slot = self.sim.n_substeps[None] % self.sim.cuda_chunk_size
        for p in range(self.num[None]):
            for i in ti.static(range(3)):
                self.sim.x.grad[p, slot][i] += gx[p, i]
                for j in ti.static(range(3)):
                    self.sim.F.grad[p, slot][i, j] += gf[p, i, j]

    def forward(self, state, force, clear_force_grad=True):
        force = np.asarray(force, dtype=np.float32)
        if force.shape != (128, 128, 128, 3) or not np.isfinite(force).all():
            raise ValueError(
                "Force must be finite and match the actual 128^3 simulator field."
            )
        self.sim.cached_states.clear()
        self.sim.reset_dt()
        self.sim.time[None] = 0.0
        self.sim.cfl_satisfy[None] = 1
        self.sim.clear_grads()
        self.sim.p_vol.grad.fill(0)
        if clear_force_grad:
            self.sim.force_field.grad.fill(0)
        self.initialize_particles(
            state["x"], state["v"], state["F"], state["C"], self.mu, self.lam
        )
        self.sim.from_torch_force_field(np.ascontiguousarray(force))
        self.sim.advance(0)
        ti.sync()
        if not self.sim.cfl_satisfy[None]:
            raise FloatingPointError("MPM CFL/domain guard failed")
        result = {k: np.empty_like(state[k]) for k in state}
        for k in result:
            getattr(self.sim, "get_" + k)(1, result[k])
        if not all(np.isfinite(v).all() for v in result.values()):
            raise FloatingPointError("Nonfinite MPM state")
        speed = np.linalg.norm(result["v"], axis=-1)
        self.last_state_diagnostics = dict(
            max_speed=float(speed.max()),
            minimum_det=float(np.linalg.det(result["F"]).min()),
            legacy_speed_threshold_exceeded=int((speed > 5).sum()),
            fast_gaussians=int((speed[: self.ng] > 5).sum()),
            fast_filling_particles=int((speed[self.ng :] > 5).sum()),
            final_advective_cfl=float(
                np.abs(result["v"]).max()
                / self.phys.fps
                / self.substeps
                / self.phys.voxel_size
            ),
        )
        return result

    def render(self, state, view, require_grad=False):
        x = (
            torch.from_numpy(state["x"][: self.ng].copy())
            .cuda()
            .requires_grad_(require_grad)
        )
        f = (
            torch.from_numpy(state["F"][: self.ng].copy())
            .cuda()
            .requires_grad_(require_grad)
        )
        delta = x * self.scale - self.shift + self.center - self.cloud.get_xyz
        result = render_phygs(
            self.cameras[view],
            self.cloud,
            self.pipeline,
            self.background,
            delta,
            0.0,
            0.0,
            f,
            False,
        )
        return result["render"], x, f

    def objective(self, state, frame, gradient=False, loss_mode="mse"):
        if not 1 <= frame <= 48:
            raise ValueError("Training frame outside reconstruction prefix")
        total = 0.0
        psnrs = []
        gx = np.zeros((self.n, 3), np.float32)
        gf = np.zeros((self.n, 3, 3), np.float32)
        for view in range(4):
            with torch.set_grad_enabled(gradient):
                image, x, f = self.render(state, view, gradient)
                if self.image_domain == "clipped":
                    image = image.clamp(0, 1)
                gt = image_tensor(
                    self.spec["gt"] / ("train_view%d" % view) / ("%05d.png" % frame)
                )
                mse = (image - gt).square().mean()
                if loss_mode == "mse":
                    term = mse
                else:
                    raise ValueError(loss_mode)
                loss = term / 4
                if gradient:
                    grads = torch.autograd.grad(loss, (x, f))
                    gx[: self.ng] += grads[0].detach().cpu().numpy()
                    gf[: self.ng] += grads[1].detach().cpu().numpy()
                total += float(loss.detach())
                psnrs.append(float(-10 * torch.log10(mse.detach())))
        return (
            dict(loss=total, train_psnr=float(np.mean(psnrs)), per_view_psnr=psnrs),
            gx,
            gf,
        )

    def backward(self, gx, gf):
        self.seed_state_grad(np.ascontiguousarray(gx), np.ascontiguousarray(gf))
        self.sim.advance_grad(0)
        ti.sync()
        grad = self.sim.force_field.grad.to_numpy()
        if not np.isfinite(grad).all():
            raise FloatingPointError("Nonfinite MPM force adjoint")
        return grad

    def evaluate(
        self,
        state,
        force,
        frame,
        gradient=False,
        full_gradient=True,
        clear_force_grad=True,
        loss_mode="mse",
    ):
        next_state = self.forward(state, force, clear_force_grad)
        metrics, gx, gf = self.objective(next_state, frame, gradient, loss_mode)
        metrics["state_diagnostics"] = self.last_state_diagnostics
        grad = None
        if gradient:
            metrics["image_position_gradient_norm"] = float(np.linalg.norm(gx))
            metrics["image_F_gradient_norm"] = float(np.linalg.norm(gf))
            grad = self.backward(gx, gf if full_gradient else np.zeros_like(gf))
        return metrics, grad, next_state
