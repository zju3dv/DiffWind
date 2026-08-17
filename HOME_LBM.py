import taichi as ti

float_type = ti.f32
mat3 = ti.types.matrix(3, 3, float_type)
vec3 = ti.types.vector(3, float_type)


@ti.data_oriented
class HOME_LBM:
    def __init__(self, nx, ny, nz, nu):
        self.nu = nu
        self.tau = 3 * nu + 0.5
        self.nx, self.ny, self.nz = nx, ny, nz
        self.fx, self.fy, self.fz = 0.0, 0.0, 0.0
        self.cs = 1 / ti.sqrt(3.0)
        self.C_mat = [
            [
                0,
                1,
                -1,
                0,
                0,
                0,
                0,
                1,
                -1,
                1,
                -1,
                0,
                0,
                1,
                -1,
                1,
                -1,
                0,
                0,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
            ],
            [
                0,
                0,
                0,
                1,
                -1,
                0,
                0,
                1,
                -1,
                0,
                0,
                1,
                -1,
                -1,
                1,
                0,
                0,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                1,
                -1,
                0,
                0,
                1,
                -1,
                1,
                -1,
                0,
                0,
                -1,
                1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
            ],
        ]
        self.H2 = ti.Matrix.field(3, 3, float_type, shape=27)
        self.H3_xxy = ti.field(float_type, shape=27)
        self.H3_xyy = ti.field(float_type, shape=27)
        self.H3_xxz = ti.field(float_type, shape=27)
        self.H3_xzz = ti.field(float_type, shape=27)
        self.H3_yzz = ti.field(float_type, shape=27)
        self.H3_yyz = ti.field(float_type, shape=27)
        self.H3_xyz = ti.field(float_type, shape=27)
        self.f = ti.Vector.field(27, float_type, shape=(nx, ny, nz))
        self.F = ti.Vector.field(27, float_type, shape=(nx, ny, nz))
        self.rho = ti.field(float_type, shape=(nx, ny, nz))
        self.v = ti.Vector.field(3, float_type, shape=(nx, ny, nz))
        self.e = ti.Vector.field(3, ti.i32, shape=27)
        self.e_f = ti.Vector.field(3, float_type, shape=27)
        self.S = ti.Matrix.field(3, 3, float_type, shape=(nx, ny, nz))
        self.w = ti.field(float_type, shape=27)
        self.force = ti.Vector.field(3, float_type, shape=(nx, ny, nz))
        self.vel_rb = ti.Vector.field(3, float_type, shape=())
        self.external_force = ti.Vector.field(3, float_type, shape=())
        self.LR = [
            0,
            2,
            1,
            4,
            3,
            6,
            5,
            8,
            7,
            10,
            9,
            12,
            11,
            14,
            13,
            16,
            15,
            18,
            17,
            20,
            19,
            22,
            21,
            24,
            23,
            26,
            25,
        ]
        self.solid = ti.field(ti.i8, shape=(nx, ny, nz))
        self.wind_force = ti.Vector.field(3, float_type, shape=(nx, ny, nz))
        self.mpm_velo = ti.Vector.field(3, float_type, shape=(nx, ny, nz))

        # Boundary modes: 0 periodic, 1 fixed pressure, 2 fixed velocity.
        self.bc_x_left, self.rho_bcxl, self.vx_bcxl, self.vy_bcxl, self.vz_bcxl = (
            0,
            1.0,
            0.0,
            0.0,
            0.0,
        )
        self.bc_x_right, self.rho_bcxr, self.vx_bcxr, self.vy_bcxr, self.vz_bcxr = (
            0,
            1.0,
            0.0,
            0.0,
            0.0,
        )
        self.bc_y_left, self.rho_bcyl, self.vx_bcyl, self.vy_bcyl, self.vz_bcyl = (
            0,
            1.0,
            0.0,
            0.0,
            0.0,
        )
        self.bc_y_right, self.rho_bcyr, self.vx_bcyr, self.vy_bcyr, self.vz_bcyr = (
            0,
            1.0,
            0.0,
            0.0,
            0.0,
        )
        self.bc_z_left, self.rho_bczl, self.vx_bczl, self.vy_bczl, self.vz_bczl = (
            0,
            1.0,
            0.0,
            0.0,
            0.0,
        )
        self.bc_z_right, self.rho_bczr, self.vx_bczr, self.vy_bczr, self.vz_bczr = (
            0,
            1.0,
            0.0,
            0.0,
            0.0,
        )

        for i in range(27):
            e_i = ti.Vector(
                [self.C_mat[0][i], self.C_mat[1][i], self.C_mat[2][i]], float_type
            )
            e_ii = ti.Vector(
                [self.C_mat[0][i], self.C_mat[1][i], self.C_mat[2][i]], ti.i32
            )
            self.e_f[i] = e_i
            self.e[i] = e_ii
            if i == 0:
                self.w[i] = 8.0 / 27.0
            elif 1 <= i <= 6:
                self.w[i] = 2.0 / 27.0
            elif 7 <= i <= 18:
                self.w[i] = 1.0 / 54.0
            elif 19 <= i <= 26:
                self.w[i] = 1.0 / 216.0
            self.H2[i] = e_i.outer_product(e_i) - 1.0 / 3.0 * mat3(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            )
            self.H3_xxy[i] = e_i[0] * e_i[0] * e_i[1] - 1.0 / 3.0 * e_i[1]
            self.H3_xyy[i] = e_i[0] * e_i[1] * e_i[1] - 1.0 / 3.0 * e_i[0]
            self.H3_xxz[i] = e_i[0] * e_i[0] * e_i[2] - 1.0 / 3.0 * e_i[2]
            self.H3_xzz[i] = e_i[0] * e_i[2] * e_i[2] - 1.0 / 3.0 * e_i[0]
            self.H3_yzz[i] = e_i[1] * e_i[2] * e_i[2] - 1.0 / 3.0 * e_i[1]
            self.H3_yyz[i] = e_i[1] * e_i[1] * e_i[2] - 1.0 / 3.0 * e_i[2]
            self.H3_xyz[i] = e_i[0] * e_i[1] * e_i[2]

    def init_simulation(self):
        self.bc_vel_x_left = [self.vx_bcxl, self.vy_bcxl, self.vz_bcxl]
        self.bc_vel_x_right = [self.vx_bcxr, self.vy_bcxr, self.vz_bcxr]
        self.bc_vel_y_left = [self.vx_bcyl, self.vy_bcyl, self.vz_bcyl]
        self.bc_vel_y_right = [self.vx_bcyr, self.vy_bcyr, self.vz_bcyr]
        self.bc_vel_z_left = [self.vx_bczl, self.vy_bczl, self.vz_bczl]
        self.bc_vel_z_right = [self.vx_bczr, self.vy_bczr, self.vz_bczr]

        self.init_fields()

    def load_geo(self, solid_np):
        self.solid.fill(0)
        self.solid.from_numpy(solid_np)

    def load_mpm_velo(self, mpm_velo_np):
        self.mpm_velo.fill(0)
        self.mpm_velo.from_numpy(mpm_velo_np)

    @ti.kernel
    def init_fields(self):
        self.vel_rb[None] = vec3(0)
        self.external_force[None] = vec3(0)
        for i in ti.grouped(self.rho):
            self.rho[i] = 1.0
            self.v[i] = ti.Vector([0, 0, 0])
            self.force[i] = self.external_force[None]
            self.S[i] = mat3(0)
        for i in ti.grouped(self.rho):
            rho = self.rho[i]
            v = self.v[i]
            S = self.S[i]
            for s in ti.static(range(27)):
                self.f[i][s] = self.reconstruct_F_local(v, rho, S, s)
                self.F[i][s] = self.reconstruct_F_local(v, rho, S, s)

    @ti.kernel
    def collision(self):
        tau = self.tau
        for i in ti.grouped(self.rho):
            v = self.v[i]
            rho = self.rho[i]
            f = self.force[i]
            self.v[i] += (f / 2) / rho
            S_new = S = self.S[i]
            Sxx = S[0, 0]
            Syy = S[1, 1]
            Szz = S[2, 2]
            Sxy = S[0, 1]
            Syz = S[1, 2]
            Sxz = S[0, 2]
            S_new[1, 0] = S_new[0, 1] = (
                (1.0 - 1.0 / tau) * Sxy
                + 1.0 / tau * v.x * v.y
                + (2.0 * tau - 1.0) / (2.0 * tau * rho) * (f.x * v.y + f.y * v.x)
            )
            S_new[1, 2] = S_new[2, 1] = (
                (1.0 - 1.0 / tau) * Syz
                + 1.0 / tau * v.y * v.z
                + (2.0 * tau - 1.0) / (2.0 * tau * rho) * (f.z * v.y + f.y * v.z)
            )
            S_new[2, 0] = S_new[0, 2] = (
                (1.0 - 1.0 / tau) * Sxz
                + 1.0 / tau * v.x * v.z
                + (2.0 * tau - 1.0) / (2.0 * tau * rho) * (f.x * v.z + f.z * v.x)
            )
            S_new[0, 0] = (
                (tau - 1.0) / (3.0 * tau) * (2.0 * Sxx - Syy - Szz)
                + (v.x**2.0 + v.y**2.0 + v.z**2.0) / 3
                + (2.0 * v.x**2.0 - v.y**2.0 - v.z**2.0) / (3.0 * tau)
                + f.x * v.x / rho
                + (tau - 1.0)
                / (3.0 * tau * rho)
                * (2.0 * f.x * v.x - f.y * v.y - f.z * v.z)
            )
            S_new[1, 1] = (
                (tau - 1.0) / (3.0 * tau) * (2.0 * Syy - Sxx - Szz)
                + 1.0 / 3.0 * (v.x**2.0 + v.y**2.0 + v.z**2.0)
                + (2.0 * v.y**2.0 - v.x**2.0 - v.z**2.0) / (3.0 * tau)
                + f.y * v.y / rho
                + (tau - 1.0)
                / (3.0 * tau * rho)
                * (2.0 * f.y * v.y - f.x * v.x - f.z * v.z)
            )
            S_new[2, 2] = (
                (tau - 1.0) / (3.0 * tau) * (2.0 * Szz - Syy - Sxx)
                + 1.0 / 3.0 * (v.x**2.0 + v.y**2.0 + v.z**2.0)
                + (2.0 * v.z**2.0 - v.y**2.0 - v.x**2.0) / (3.0 * tau)
                + f.z * v.z / rho
                + (tau - 1.0)
                / (3.0 * tau * rho)
                * (2.0 * f.z * v.z - f.y * v.y - f.x * v.x)
            )
            self.S[i] = S_new

    @ti.kernel
    def HOME_streaming(self):
        for i in ti.grouped(self.rho):
            for s in ti.static(range(27)):
                fp = 0.0
                ip = (
                    i - self.e[s]
                )  # pre dt ,detect collision with solid object. [ip] streaming to [i]
                if (
                    (ip[0] >= 0)
                    and (ip[0] < self.nx)
                    and (ip[1] >= 0)
                    and (ip[1] < self.ny)
                    and (ip[2] >= 0)
                    and (ip[2] < self.nz)
                ):
                    if self.solid[i] == 0 and self.solid[ip] == 1:  # from solid to air
                        vp = self.mpm_velo[i]
                        vx = self.v[i]
                        rhop = self.rho[i]
                        Sp = vp.outer_product(vp) + self.S[i] - vx.outer_product(vx)
                        fp = self.reconstruct_F_local(vp, rhop, Sp, s)
                    else:
                        fp = self.f[ip][s]
                else:
                    fp = self.feq(s, 1.0, -self.vel_rb[None])

                self.F[i][s] = fp

    @ti.func
    def feq(self, k, rho_local, u):
        eu = self.e_f[k].dot(u)
        uu = u.dot(u)
        feq = (
            self.w[k]
            * rho_local
            * (
                1.0
                + eu / (self.cs**2)
                + eu * eu / (2 * self.cs**4)
                - uu / (2 * self.cs**2)
            )
        )
        return feq

    @ti.func
    def reconstruct_F_local(self, v, rho, S, s):
        e_f = self.e_f[s]
        part1 = 1 + e_f.dot(v) / self.cs**2
        part2 = (self.H2[s] * S).sum() / (2 * self.cs**4)
        part3_0 = self.H3_xxy[s] * (
            S[0, 0] * v.y + 2 * S[0, 1] * v.x - 2 * v.x * v.x * v.y
        )
        part3_1 = self.H3_xyy[s] * (
            S[1, 1] * v.x + 2 * S[0, 1] * v.y - 2 * v.x * v.y * v.y
        )
        part3_2 = self.H3_xxz[s] * (
            S[0, 0] * v.z + 2 * S[0, 2] * v.x - 2 * v.x * v.x * v.z
        )
        part3_3 = self.H3_xzz[s] * (
            S[2, 2] * v.x + 2 * S[0, 2] * v.z - 2 * v.x * v.z * v.z
        )
        part3_4 = self.H3_yzz[s] * (
            S[2, 2] * v.y + 2 * S[1, 2] * v.z - 2 * v.y * v.z * v.z
        )
        part3_5 = self.H3_yyz[s] * (
            S[1, 1] * v.z + 2 * S[1, 2] * v.z - 2 * v.y * v.y * v.z
        )
        part3_6 = self.H3_xyz[s] * (
            S[0, 2] * v.y + S[1, 2] * v.x + S[0, 1] * v.z - 2 * v.x * v.y * v.z
        )
        part3 = (
            part3_0 + part3_1 + part3_2 + part3_3 + part3_4 + part3_5 + part3_6
        ) / (2 * self.cs**6)
        return rho * self.w[s] * (part1 + part2 + part3)

    @ti.kernel
    def construct_distribution(self):
        for i in ti.grouped(self.rho):
            rho = self.rho[i]
            v = self.v[i]
            S = self.S[i]
            for s in ti.static(range(27)):
                self.f[i][s] = self.reconstruct_F_local(v, rho, S, s)

    @ti.kernel
    def compute_moments(self):
        for i in ti.grouped(self.rho):
            rho = 0.0
            S = mat3(0)
            v = vec3(0)
            for s in ti.static(range(27)):
                F = self.F[i][s]
                rho = rho + F
                v = v + self.e_f[s] * F
                S = S + self.H2[s] * F
            self.rho[i] = rho
            self.S[i] = S / rho
            self.v[i] = (v + self.force[i] / 2) / rho

    @ti.kernel
    def cal_wind_force_field(self):
        self.wind_force.fill(0.0)
        for i in ti.grouped(self.rho):
            v_lbm = self.v[i]
            rho_lbm = self.rho[i]
            v_mpm = self.mpm_velo[i]
            v_rel = v_lbm - v_mpm
            local_force = vec3(0)
            if v_rel.norm() > 1e-6:
                local_force = rho_lbm * v_rel.norm() ** 2 * v_rel.normalized()

            self.wind_force[i] = local_force

    def get_wind_force_field(self):
        return self.wind_force

    def set_bc_vel_x1(self, vel):
        self.bc_x_right = 2
        self.vx_bcxr = vel[0]
        self.vy_bcxr = vel[1]
        self.vz_bcxr = vel[2]

    def set_bc_vel_x0(self, vel):
        self.bc_x_left = 2
        self.vx_bcxl = vel[0]
        self.vy_bcxl = vel[1]
        self.vz_bcxl = vel[2]

    def set_bc_vel_y1(self, vel):
        self.bc_y_right = 2
        self.vx_bcyr = vel[0]
        self.vy_bcyr = vel[1]
        self.vz_bcyr = vel[2]

    def set_bc_vel_y0(self, vel):
        self.bc_y_left = 2
        self.vx_bcyl = vel[0]
        self.vy_bcyl = vel[1]
        self.vz_bcyl = vel[2]

    def set_bc_vel_z1(self, vel):
        self.bc_z_right = 2
        self.vx_bczr = vel[0]
        self.vy_bczr = vel[1]
        self.vz_bczr = vel[2]

    def set_bc_vel_z0(self, vel):
        self.bc_z_left = 2
        self.vx_bczl = vel[0]
        self.vy_bczl = vel[1]
        self.vz_bczl = vel[2]

    @ti.kernel
    def Boundary_condition(self):
        if ti.static(self.bc_x_left == 1):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                if self.solid[0, j, k] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[1, j, k] > 0:
                            self.F[0, j, k][s] = self.feq(
                                s, self.rho_bcxl, self.v[1, j, k]
                            )
                        else:
                            self.F[0, j, k][s] = self.feq(
                                s, self.rho_bcxl, self.v[0, j, k]
                            )

        if ti.static(self.bc_x_left == 2):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                if self.solid[0, j, k] == 0:
                    for s in ti.static(range(19)):
                        self.F[0, j, k][s] = self.feq(
                            s, 1.0, ti.Vector(self.bc_vel_x_left)
                        )

        if ti.static(self.bc_x_right == 1):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                if self.solid[self.nx - 1, j, k] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[self.nx - 2, j, k] > 0:
                            self.F[self.nx - 1, j, k][s] = self.feq(
                                s, self.rho_bcxr, self.v[self.nx - 2, j, k]
                            )
                        else:
                            self.F[self.nx - 1, j, k][s] = self.feq(
                                s, self.rho_bcxr, self.v[self.nx - 1, j, k]
                            )

        if ti.static(self.bc_x_right == 2):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                if self.solid[self.nx - 1, j, k] == 0:
                    for s in ti.static(range(19)):
                        self.F[self.nx - 1, j, k][s] = self.feq(
                            s, 1.0, ti.Vector(self.bc_vel_x_right)
                        )

        # Direction Y
        if ti.static(self.bc_y_left == 1):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                if self.solid[i, 0, k] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[i, 1, k] > 0:
                            self.F[i, 0, k][s] = self.feq(
                                s, self.rho_bcyl, self.v[i, 1, k]
                            )
                        else:
                            self.F[i, 0, k][s] = self.feq(
                                s, self.rho_bcyl, self.v[i, 0, k]
                            )

        if ti.static(self.bc_y_left == 2):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                if self.solid[i, 0, k] == 0:
                    for s in ti.static(range(19)):
                        self.F[i, 0, k][s] = self.feq(
                            s, 1.0, ti.Vector(self.bc_vel_y_left)
                        )

        if ti.static(self.bc_y_right == 1):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                if self.solid[i, self.ny - 1, k] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[i, self.ny - 2, k] > 0:
                            self.F[i, self.ny - 1, k][s] = self.feq(
                                s, self.rho_bcyr, self.v[i, self.ny - 2, k]
                            )
                        else:
                            self.F[i, self.ny - 1, k][s] = self.feq(
                                s, self.rho_bcyr, self.v[i, self.ny - 1, k]
                            )

        if ti.static(self.bc_y_right == 2):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                if self.solid[i, self.ny - 1, k] == 0:
                    for s in ti.static(range(19)):
                        self.F[i, self.ny - 1, k][s] = self.feq(
                            s, 1.0, ti.Vector(self.bc_vel_y_right)
                        )

        # Z direction
        if ti.static(self.bc_z_left == 1):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                if self.solid[i, j, 0] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[i, j, 1] > 0:
                            self.F[i, j, 0][s] = self.feq(
                                s, self.rho_bczl, self.v[i, j, 1]
                            )
                        else:
                            self.F[i, j, 0][s] = self.feq(
                                s, self.rho_bczl, self.v[i, j, 0]
                            )

        if ti.static(self.bc_z_left == 2):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                if self.solid[i, j, 0] == 0:
                    for s in ti.static(range(19)):
                        self.F[i, j, 0][s] = self.feq(
                            s, 1.0, ti.Vector(self.bc_vel_z_left)
                        )

        if ti.static(self.bc_z_right == 1):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                if self.solid[i, j, self.nz - 1] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[i, j, self.nz - 2] > 0:
                            self.F[i, j, self.nz - 1][s] = self.feq(
                                s, self.rho_bczr, self.v[i, j, self.nz - 2]
                            )
                        else:
                            self.F[i, j, self.nz - 1][s] = self.feq(
                                s, self.rho_bczr, self.v[i, j, self.nz - 1]
                            )

        if ti.static(self.bc_z_right == 2):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                if self.solid[i, j, self.nz - 1] == 0:
                    for s in ti.static(range(19)):
                        self.F[i, j, self.nz - 1][s] = self.feq(
                            s, 1.0, ti.Vector(self.bc_vel_z_right)
                        )

    def step(self):
        self.construct_distribution()
        self.HOME_streaming()
        self.Boundary_condition()
        self.compute_moments()
        self.collision()
