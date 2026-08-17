"""Forward state management and gradient extraction for the MPM simulator."""

import numpy as np
import taichi as ti
import torch
from torch import nn

from simulator.mpm_simulator_force import MPMSimulator_Force as MPMSimulator


FORCE_GRID_SIZE = 128
FORCE_GRADIENT_SCALE = 0.01


@ti.data_oriented
class Estimator_Force(nn.Module):
    """Configure MPM inputs, advance states, and expose simulator gradients."""

    def __init__(self, phys_args, dtype, init_vol, cuda_chunk_size=100):
        super().__init__()
        if init_vol.ndim != 2 or init_vol.shape[1] != 3:
            raise ValueError("init_vol must have shape [num_particles, 3]")

        self.device = init_vol.device
        self.dtype = ti.f64 if dtype == "float64" else ti.f32
        self.voxel_size = phys_args.voxel_size
        self.init_vol = init_vol
        self.part_counts = list(phys_args.part_counts)
        self.E_list = list(phys_args.E_list)
        self.poisson_list = list(phys_args.poisson_list)
        self._validate_material_partitions()

        particle_count = init_vol.shape[0]
        self.num_particles = ti.field(ti.i32, shape=())
        self.num_particles[None] = particle_count
        self.dx = ti.field(self.dtype, shape=())
        self.inv_dx = ti.field(self.dtype, shape=())

        self.particle_rho = ti.field(dtype=self.dtype, needs_grad=True)
        particle_layout = ti.root.dynamic(ti.i, 2**30, 2**14)
        particle_layout.place(self.particle_rho, self.particle_rho.grad)

        self.global_rho = nn.Parameter(
            torch.as_tensor(phys_args.rho, dtype=torch.float32, device=self.device)
        )
        self.init_vel = nn.Parameter(
            torch.as_tensor(phys_args.init_vel, dtype=torch.float32, device=self.device)
        )
        self.yield_stress = nn.Parameter(
            torch.tensor(
                [getattr(phys_args, "init_yield_stress", 0.0)],
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.plastic_viscosity = nn.Parameter(
            torch.tensor(
                [getattr(phys_args, "init_plastic_viscosity", -1e6)],
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.friction_alpha = nn.Parameter(
            torch.tensor(
                [getattr(phys_args, "init_friction_alpha", 0.0)],
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.cohesion = nn.Parameter(
            torch.tensor(
                [getattr(phys_args, "init_cohesion", 0.0)],
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.global_force = nn.Parameter(
            torch.zeros(
                (FORCE_GRID_SIZE, FORCE_GRID_SIZE, FORCE_GRID_SIZE, 3),
                dtype=torch.float32,
                device=self.device,
            )
        )

        frame_dt = 1.0 / phys_args.fps
        self.simulator = MPMSimulator(
            dtype=self.dtype,
            dt=frame_dt / phys_args.mpm_iter_cnt,
            frame_dt=frame_dt,
            n_particles=self.num_particles,
            material=phys_args.material,
            dx=self.dx,
            inv_dx=self.inv_dx,
            particle_layout=particle_layout,
            args=phys_args,
            gravity=phys_args.gravity,
            cuda_chunk_size=cuda_chunk_size,
        )
        self._configure_colliders(phys_args.bc)
        self._coupling_configured = False

    def _validate_material_partitions(self):
        partition_count = len(self.part_counts)
        if (
            len(self.E_list) != partition_count
            or len(self.poisson_list) != partition_count
        ):
            raise ValueError(
                "part_counts, E_list, and poisson_list must have equal lengths"
            )
        if sum(self.part_counts) != self.init_vol.shape[0]:
            raise ValueError("Material partition counts must match the particle count")

    def _configure_colliders(self, colliders):
        for collider_type, collider in colliders.items():
            if "ground" in collider_type:
                point, normal, boundary_style = collider
                self.simulator.add_surface_collider(point, normal, boundary_style)
            elif "cylinder" in collider_type:
                start, end, radius, boundary_style = collider
                self.simulator.add_cylinder_collider(start, end, radius, boundary_style)

    def _material_parameters(self):
        mu_parts = []
        lam_parts = []
        for youngs_modulus, poisson_ratio, count in zip(
            self.E_list, self.poisson_list, self.part_counts
        ):
            youngs_modulus = torch.tensor(
                youngs_modulus, dtype=torch.float32, device=self.device
            )
            poisson_ratio = torch.tensor(
                poisson_ratio, dtype=torch.float32, device=self.device
            )
            mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
            lam = (
                youngs_modulus
                * poisson_ratio
                / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
            )
            mu_parts.append(mu.repeat(count))
            lam_parts.append(lam.repeat(count))
        return torch.cat(mu_parts), torch.cat(lam_parts)

    def set_force_field(self, force_field):
        """Copy a dense ``[128, 128, 128, 3]`` force field to the estimator."""
        force_field = torch.as_tensor(
            force_field, dtype=torch.float32, device=self.device
        )
        if tuple(force_field.shape) != tuple(self.global_force.shape):
            raise ValueError(
                "force_field must have shape [{0}, {0}, {0}, 3]".format(FORCE_GRID_SIZE)
            )
        with torch.no_grad():
            self.global_force.copy_(force_field)

    def configure_forward_coupling(self, static_part, force_duration_frames=1e10):
        """Reproduce the force/static callback sequence used by the reference runs."""
        if self._coupling_configured:
            return

        # The reference loop registered one pair for frame 0 and another pair
        # for frame 1 before Taichi first compiled grid_op. Register that exact
        # sequence once so later initialize() calls cannot change the kernel.
        for _ in range(2):
            self.simulator.add_force_field(
                point=[1.0, 1.0, 1.0],
                size=[1.0, 1.0, 2.0],
                num_frames=force_duration_frames,
            )
            self.simulator.set_static_part_on_cuboid(
                point=static_part["point"],
                size=static_part["size"],
            )
        self._coupling_configured = True

    def initialize(self):
        """Reset the initial MPM state and upload the current physical inputs."""
        torch.cuda.synchronize()
        ti.sync()

        particle_count = self.num_particles[None]
        velocities = self.init_vel.repeat(particle_count).reshape(particle_count, 3)
        particle_rho = self.global_rho.repeat(particle_count)
        particle_mu, particle_lam = self._material_parameters()

        self.dx[None] = self.voxel_size
        self.inv_dx[None] = 1.0 / self.voxel_size
        self.simulator.reset_dt()
        self.simulator.cached_states.clear()
        self.clear_gradients()
        self.compute_particle_volume()
        self.load_initial_state(
            self.init_vol.detach().cpu().numpy(),
            velocities.detach().cpu().numpy(),
            particle_rho.detach().cpu().numpy(),
            particle_mu.detach().cpu().numpy(),
            particle_lam.detach().cpu().numpy(),
        )
        self.compute_particle_mass()
        self._set_constitutive_parameters()
        self.simulator.cfl_satisfy[None] = True

        self.simulator.from_torch_force_field(self.global_force.detach().cpu().numpy())

    def _set_constitutive_parameters(self):
        yield_stress = torch.pow(10.0, self.yield_stress)
        plastic_viscosity = torch.pow(10.0, self.plastic_viscosity)
        sin_phi = torch.sin(torch.deg2rad(self.friction_alpha))
        friction_alpha = np.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
        self.simulator.yield_stress[None] = yield_stress.item()
        self.simulator.plastic_viscosity[None] = plastic_viscosity.item()
        self.simulator.friction_alpha[None] = friction_alpha.item()
        self.simulator.cohesion[None] = self.cohesion.item()

    def clear_gradients(self):
        self.particle_rho.grad.fill(0)
        self.simulator.clear_grads()

    @ti.kernel
    def compute_particle_volume(self):
        particle_volume = (self.dx[None] * 0.5) ** 3
        for particle in range(self.num_particles[None]):
            self.simulator.p_vol[particle] = particle_volume

    @ti.kernel
    def compute_particle_mass(self):
        for particle in range(self.num_particles[None]):
            self.simulator.p_mass[particle] = (
                self.particle_rho[particle] * self.simulator.p_vol[particle]
            )

    @ti.kernel
    def load_initial_state(
        self,
        positions: ti.types.ndarray(),
        velocities: ti.types.ndarray(),
        particle_rho: ti.types.ndarray(),
        particle_mu: ti.types.ndarray(),
        particle_lam: ti.types.ndarray(),
    ):
        for particle in range(self.num_particles[None]):
            self.particle_rho[particle] = particle_rho[particle]
            self.simulator.mu[particle] = particle_mu[particle]
            self.simulator.lam[particle] = particle_lam[particle]
            self.simulator.p_mass[particle] = 0.0
            self.simulator.F[particle, 0] = ti.Matrix.identity(self.dtype, 3)
            self.simulator.C[particle, 0] = ti.Matrix.zero(self.dtype, 3, 3)
            for axis in ti.static(range(3)):
                self.simulator.x[particle, 0][axis] = positions[particle, axis]
                self.simulator.v[particle, 0][axis] = velocities[particle, axis]

    def load_simulator_state(
        self, particle_pos, particle_velo, particle_F, particle_C, frame=0
    ):
        self.simulator.set_x(frame, self._to_numpy(particle_pos))
        self.simulator.set_v(frame, self._to_numpy(particle_velo))
        self.simulator.set_F(frame, self._to_numpy(particle_F))
        self.simulator.set_C(frame, self._to_numpy(particle_C))

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return value

    def forward_sim(self, frame):
        """Advance to ``frame`` and return position, velocity, F, and C tensors."""
        if frame > 0:
            self.simulator.advance(frame - 1)
        if not self.simulator.cfl_satisfy[None]:
            raise RuntimeError("MPM simulation failed its CFL stability check")

        state = self._read_state(frame)
        state = self._sanitize_state(state)
        self.load_simulator_state(*state, frame=frame)
        return state

    def _read_state(self, frame):
        particle_count = self.num_particles[None]
        position = np.zeros((particle_count, 3), dtype=np.float32)
        velocity = np.zeros((particle_count, 3), dtype=np.float32)
        deformation = np.zeros((particle_count, 3, 3), dtype=np.float32)
        affine = np.zeros((particle_count, 3, 3), dtype=np.float32)
        self.simulator.get_x(frame, position)
        self.simulator.get_v(frame, velocity)
        self.simulator.get_F(frame, deformation)
        self.simulator.get_C(frame, affine)
        return tuple(
            torch.from_numpy(value).to(self.device)
            for value in (position, velocity, deformation, affine)
        )

    def _sanitize_state(self, state):
        position, velocity, deformation, affine = state
        unstable = torch.linalg.norm(velocity, dim=1) > 5.0
        if unstable.any():
            velocity[unstable] = 0.0
            identity = torch.eye(3, dtype=deformation.dtype, device=self.device)
            deformation[unstable] = identity
            affine[unstable] = identity
        return position, velocity, deformation, affine

    @ti.kernel
    def set_position_gradient(
        self, frame: ti.i32, position_gradient: ti.types.ndarray()
    ):
        local_index = (
            frame * self.simulator.n_substeps[None]
        ) % self.simulator.cuda_chunk_size
        for particle in range(self.num_particles[None]):
            for axis in ti.static(range(3)):
                self.simulator.x.grad[particle, local_index][axis] += position_gradient[
                    particle, axis
                ]

    @ti.kernel
    def export_input_gradients(
        self,
        position_gradient: ti.types.ndarray(),
        velocity_gradient: ti.types.ndarray(),
        rho_gradient: ti.types.ndarray(),
        mu_gradient: ti.types.ndarray(),
        lam_gradient: ti.types.ndarray(),
    ):
        for particle in range(self.num_particles[None]):
            rho_gradient[particle] = self.particle_rho.grad[particle]
            mu_gradient[particle] = self.simulator.mu.grad[particle]
            lam_gradient[particle] = self.simulator.lam.grad[particle]
            for axis in ti.static(range(3)):
                position_gradient[particle, axis] = self.simulator.x.grad[particle, 0][
                    axis
                ]
                velocity_gradient[particle, axis] = self.simulator.v.grad[particle, 0][
                    axis
                ]

    @ti.kernel
    def export_force_field_gradient(self, gradient: ti.types.ndarray()):
        for i, j, k in ti.ndrange(FORCE_GRID_SIZE, FORCE_GRID_SIZE, FORCE_GRID_SIZE):
            for axis in ti.static(range(3)):
                gradient[i, j, k, axis] = self.simulator.force_field.grad[i, j, k][axis]

    def backward(self, frame, position_gradient=None):
        """Backpropagate one frame and return input gradients at frame zero."""
        if position_gradient is not None:
            position_gradient = self._to_numpy(position_gradient).astype(
                np.float32, copy=False
            )
            expected_shape = (self.num_particles[None], 3)
            if position_gradient.shape != expected_shape:
                raise ValueError(
                    "position_gradient must have shape {}".format(expected_shape)
                )
            self.set_position_gradient(frame, position_gradient)

        if frame > 0:
            self.simulator.advance_grad(frame - 1)
            return None

        self.compute_particle_mass.grad()
        return self._collect_input_gradients()

    def _collect_input_gradients(self):
        particle_count = self.num_particles[None]
        position_gradient = np.zeros((particle_count, 3), dtype=np.float32)
        velocity_gradient = np.zeros((particle_count, 3), dtype=np.float32)
        rho_gradient = np.zeros(particle_count, dtype=np.float32)
        mu_gradient = np.zeros(particle_count, dtype=np.float32)
        lam_gradient = np.zeros(particle_count, dtype=np.float32)
        self.export_input_gradients(
            position_gradient,
            velocity_gradient,
            rho_gradient,
            mu_gradient,
            lam_gradient,
        )

        yield_stress_gradient = np.array(
            [self.simulator.yield_stress.grad[None]], dtype=np.float32
        )
        viscosity_gradient = np.array(
            [self.simulator.plastic_viscosity.grad[None]], dtype=np.float32
        )
        friction_gradient = np.array(
            [self.simulator.friction_alpha.grad[None]], dtype=np.float32
        )
        cohesion_gradient = np.array(
            [self.simulator.cohesion.grad[None]], dtype=np.float32
        )
        force_field_gradient = np.zeros(
            (FORCE_GRID_SIZE, FORCE_GRID_SIZE, FORCE_GRID_SIZE, 3),
            dtype=np.float32,
        )
        self.export_force_field_gradient(force_field_gradient)
        force_field_gradient = np.nan_to_num(force_field_gradient, nan=0.0)
        force_field_gradient *= FORCE_GRADIENT_SCALE

        arrays = (
            position_gradient,
            velocity_gradient,
            mu_gradient,
            lam_gradient,
            yield_stress_gradient,
            viscosity_gradient,
            friction_gradient,
            cohesion_gradient,
            rho_gradient,
            force_field_gradient,
        )
        return tuple(torch.from_numpy(array).to(self.device) for array in arrays)
