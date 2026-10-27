"""GPU regression for zero-stiffness particle stress."""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import taichi as ti
from simulator.mpm_simulator_force import MPMSimulator_Force


@ti.data_oriented
class Fixture:
    def __init__(self):
        self.n = ti.field(ti.i32, shape=())
        self.dx = ti.field(ti.f32, shape=())
        self.inv_dx = ti.field(ti.f32, shape=())
        layout = ti.root.dynamic(ti.i, 16, 16)
        self.sim = MPMSimulator_Force(
            ti.f32, 1e-4, 1e-4, layout, self.dx, self.inv_dx, self.n,
            SimpleNamespace(damping=0.0), material=10, cuda_chunk_size=1,
        )
        self.n[None] = 1
        self.dx[None] = 0.03125
        self.inv_dx[None] = 32.0

    @ti.kernel
    def setup(self, f: ti.types.ndarray(), mu: ti.f32, lam: ti.f32):
        self.sim.x[0, 0] = ti.Vector([0.91, 0.93, 0.95])
        self.sim.v[0, 0] = ti.Vector.zero(ti.f32, 3)
        self.sim.C[0, 0] = ti.Matrix.zero(ti.f32, 3, 3)
        self.sim.p_vol[0] = (self.dx[None]*0.5)**3
        self.sim.p_mass[0] = self.sim.p_vol[0]*1000
        self.sim.mu[0] = mu
        self.sim.lam[0] = lam
        for i, j in ti.static(ti.ndrange(3, 3)):
            self.sim.F[0, 1][i, j] = f[i, j]

    @ti.kernel
    def seed(self):
        for I in ti.grouped(self.sim.grid_m):
            self.sim.grid_v_in.grad[I] = I.cast(ti.f32)

    def evaluate(self, f, mu, lam):
        self.sim.grid.deactivate_all()
        self.sim.clear_grads()
        self.setup(np.asarray(f, np.float32), mu, lam)
        self.sim.p2g(0)
        momentum = self.sim.grid_v_in.to_numpy()
        self.seed()
        self.sim.p2g.grad(0)
        grad = self.sim.F.grad[0, 1].to_numpy()
        return momentum, grad


class ZeroStiffnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cuda, fast_math=False, device_memory_fraction=0.04,
                offline_cache=False)
        cls.fixture = Fixture()

    def test_passive_inverted_deformation_has_zero_stress_and_adjoint(self):
        momentum, grad = self.fixture.evaluate(np.diag([-1.0, 1.0, 1.0]), 0.0, 0.0)
        self.assertTrue(np.isfinite(momentum).all())
        np.testing.assert_array_equal(momentum, 0)
        self.assertTrue(np.isfinite(grad).all())
        np.testing.assert_array_equal(grad, 0)

    def test_elastic_deformation_keeps_nonzero_stress_and_adjoint(self):
        momentum, grad = self.fixture.evaluate(np.diag([1.05, 0.97, 1.0]), 1e6, 1e6)
        self.assertTrue(np.isfinite(momentum).all())
        self.assertGreater(float(np.linalg.norm(momentum)), 0)
        self.assertTrue(np.isfinite(grad).all())
        self.assertGreater(float(np.linalg.norm(grad)), 0)


if __name__ == '__main__':
    unittest.main()
