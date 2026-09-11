
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from skimage import measure


class ViscousFlowAroundCube:
    

    def __init__(self, nx=35, ny=35, nz=35, Re=50, dt=None, L=4.0, cube_size=0.8):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.Re = Re
        self.nu = 1.0 / Re
        self.U0 = 1.0

        self.x = np.linspace(0, L, nx)
        self.y = np.linspace(0, L, ny)
        self.z = np.linspace(0, L, nz)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]

        # dt 
        if dt is None:
            h = min(self.dx, self.dy, self.dz)
            dt_conv = 0.5 * h / self.U0
            dt_diff = 0.5 * h ** 2 / (6 * self.nu)
            dt = 0.4 * min(dt_conv, dt_diff)
        self.dt = dt

        self.u = np.zeros((nx, ny, nz))
        self.v = np.zeros((nx, ny, nz))
        self.w = np.zeros((nx, ny, nz))
        self.p = np.zeros((nx, ny, nz))

        self.cube_size = cube_size
        self.cube_center = (L / 2, L / 2, L / 2)
        self.cube_mask = self.create_cube_mask()
        self.set_initial_conditions()

    def create_cube_mask(self):
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        cx, cy, cz = self.cube_center
        half = self.cube_size / 2
        return (np.abs(X - cx) <= half) & (np.abs(Y - cy) <= half) & (np.abs(Z - cz) <= half)

    def set_initial_conditions(self):
        self.u[:, :, :] = self.U0
        self.u[self.cube_mask] = 0
        self.v[self.cube_mask] = 0
        self.w[self.cube_mask] = 0

    def apply_boundary_conditions(self):
        self.u[self.cube_mask] = 0
        self.v[self.cube_mask] = 0
        self.w[self.cube_mask] = 0

        self.u[0, :, :] = self.U0
        self.v[0, :, :], self.w[0, :, :] = 0, 0

        self.u[-1, :, :] = self.u[-2, :, :]
        self.v[-1, :, :] = self.v[-2, :, :]
        self.w[-1, :, :] = self.w[-2, :, :]

        self.u[:, 0, :] = self.u[:, -1, :] = self.u[:, :, 0] = self.u[:, :, -1] = 0
        self.v[:, 0, :] = self.v[:, -1, :] = self.v[:, :, 0] = self.v[:, :, -1] = 0
        self.w[:, 0, :] = self.w[:, -1, :] = self.w[:, :, 0] = self.w[:, :, -1] = 0

    def _grad_upwind(self, f, vel, axis, h):
        if axis == 0:
            f_ip1, f_i, f_im1 = f[2:, 1:-1, 1:-1], f[1:-1, 1:-1, 1:-1], f[:-2, 1:-1, 1:-1]
        elif axis == 1:
            f_ip1, f_i, f_im1 = f[1:-1, 2:, 1:-1], f[1:-1, 1:-1, 1:-1], f[1:-1, :-2, 1:-1]
        else:
            f_ip1, f_i, f_im1 = f[1:-1, 1:-1, 2:], f[1:-1, 1:-1, 1:-1], f[1:-1, 1:-1, :-2]
        back = (f_i - f_im1) / h
        fwd = (f_ip1 - f_i) / h
        return np.where(vel >= 0, back, fwd)

    def solve_step(self):
        u_old, v_old, w_old = self.u.copy(), self.v.copy(), self.w.copy()
        nu = self.nu
        sl = (slice(1, -1), slice(1, -1), slice(1, -1))
        uc, vc, wc = u_old[sl], v_old[sl], w_old[sl]

        # Diffussion
        lap_u = (
            (u_old[2:, 1:-1, 1:-1] - 2*uc + u_old[:-2, 1:-1, 1:-1]) / self.dx**2 +
            (u_old[1:-1, 2:, 1:-1] - 2*uc + u_old[1:-1, :-2, 1:-1]) / self.dy**2 +
            (u_old[1:-1, 1:-1, 2:] - 2*uc + u_old[1:-1, 1:-1, :-2]) / self.dz**2
        )
        lap_v = (
            (v_old[2:, 1:-1, 1:-1] - 2*vc + v_old[:-2, 1:-1, 1:-1]) / self.dx**2 +
            (v_old[1:-1, 2:, 1:-1] - 2*vc + v_old[1:-1, :-2, 1:-1]) / self.dy**2 +
            (v_old[1:-1, 1:-1, 2:] - 2*vc + v_old[1:-1, 1:-1, :-2]) / self.dz**2
        )
        lap_w = (
            (w_old[2:, 1:-1, 1:-1] - 2*wc + w_old[:-2, 1:-1, 1:-1]) / self.dx**2 +
            (w_old[1:-1, 2:, 1:-1] - 2*wc + w_old[1:-1, :-2, 1:-1]) / self.dy**2 +
            (w_old[1:-1, 1:-1, 2:] - 2*wc + w_old[1:-1, 1:-1, :-2]) / self.dz**2
        )

        #  Convection UPWIND  Pe > 2 
        conv_u = -(uc * self._grad_upwind(u_old, uc, 0, self.dx) +
                   vc * self._grad_upwind(u_old, vc, 1, self.dy) +
                   wc * self._grad_upwind(u_old, wc, 2, self.dz))
        conv_v = -(uc * self._grad_upwind(v_old, uc, 0, self.dx) +
                   vc * self._grad_upwind(v_old, vc, 1, self.dy) +
                   wc * self._grad_upwind(v_old, wc, 2, self.dz))
        conv_w = -(uc * self._grad_upwind(w_old, uc, 0, self.dx) +
                   vc * self._grad_upwind(w_old, vc, 1, self.dy) +
                   wc * self._grad_upwind(w_old, wc, 2, self.dz))

        self.u[sl] = uc + self.dt * (nu * lap_u + conv_u)
        self.v[sl] = vc + self.dt * (nu * lap_v + conv_v)
        self.w[sl] = wc + self.dt * (nu * lap_w + conv_w)

        self.apply_boundary_conditions()

        # Divergence
        div = np.zeros_like(self.u)
        div[sl] = (
            (self.u[2:, 1:-1, 1:-1] - self.u[:-2, 1:-1, 1:-1]) / (2*self.dx) +
            (self.v[1:-1, 2:, 1:-1] - self.v[1:-1, :-2, 1:-1]) / (2*self.dy) +
            (self.w[1:-1, 1:-1, 2:] - self.w[1:-1, 1:-1, :-2]) / (2*self.dz)
        )

        # Pressure
        for _ in range(40):
            p_old = self.p.copy()
            self.p[sl] = (
                (p_old[2:, 1:-1, 1:-1] + p_old[:-2, 1:-1, 1:-1]) * self.dy**2 * self.dz**2 +
                (p_old[1:-1, 2:, 1:-1] + p_old[1:-1, :-2, 1:-1]) * self.dx**2 * self.dz**2 +
                (p_old[1:-1, 1:-1, 2:] + p_old[1:-1, 1:-1, :-2]) * self.dx**2 * self.dy**2 -
                (div[sl] / self.dt) * self.dx**2 * self.dy**2 * self.dz**2
            ) / (2 * (self.dx**2 * self.dy**2 + self.dx**2 * self.dz**2 + self.dy**2 * self.dz**2))

            self.p[:, 0, :] = self.p[:, 1, :]
            self.p[:, -1, :] = self.p[:, -2, :]
            self.p[:, :, 0] = self.p[:, :, 1]
            self.p[:, :, -1] = self.p[:, :, -2]
            self.p[0, :, :] = self.p[1, :, :]
            self.p[-1, :, :] = 0.0  #  (p_inf = 0) 

        # Velocity correction
        self.u[sl] -= self.dt * (self.p[2:, 1:-1, 1:-1] - self.p[:-2, 1:-1, 1:-1]) / (2*self.dx)
        self.v[sl] -= self.dt * (self.p[1:-1, 2:, 1:-1] - self.p[1:-1, :-2, 1:-1]) / (2*self.dy)
        self.w[sl] -= self.dt * (self.p[1:-1, 1:-1, 2:] - self.p[1:-1, 1:-1, :-2]) / (2*self.dz)

        self.u = np.nan_to_num(self.u)
        self.v = np.nan_to_num(self.v)
        self.w = np.nan_to_num(self.w)

        self.apply_boundary_conditions()

    def solve(self, num_steps=600):
        print(f"Simulando {num_steps} pasos (dt={self.dt:.5f}, t_final={num_steps*self.dt:.2f})...")
        for step in range(num_steps):
            self.solve_step()
            if step % 100 == 0:
                print(f"  Paso {step}/{num_steps}  max|u|={np.max(np.abs(self.u)):.3f}  max|p|={np.max(np.abs(self.p)):.3f}")
        print("finished")

    def compute_cp(self):
        """Cp = (p - p_inf)/(0.5*rho*U0^2), rho=1, p_inf=0."""
        return 2.0 * self.p / (self.U0 ** 2)

  
    # Graphs
   
    def plot_velocity_2d_slice(self, save_path=None):
        """Figura 1: mapa de velocidad + líneas de corriente 2D (corte Z central)."""
        mid = self.nz // 2
        u2 = self.u[:, :, mid].T
        v2 = self.v[:, :, mid].T
        speed = np.sqrt(u2**2 + v2**2)

        fig, ax = plt.subplots(figsize=(7.5, 6.3))
        im = ax.pcolormesh(self.x, self.y, speed, shading='gouraud', cmap='viridis')
        ax.streamplot(self.x, self.y, u2, v2, color='white', density=1.3,
                      linewidth=0.8, arrowsize=1.0)
        half = self.cube_size / 2
        cx, cy, _ = self.cube_center
        ax.add_patch(plt.Rectangle((cx - half, cy - half), self.cube_size, self.cube_size,
                                    facecolor='red', alpha=0.9, edgecolor='black', lw=1.5))
        ax.set_title(f"Velocity field + 2D streamlines - mid Z-plane\n"
                     f"Viscous flow around a cube, Re={self.Re}", fontsize=12)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_aspect('equal')
        cbar = plt.colorbar(im, ax=ax); cbar.set_label("|V|")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_pressure_coefficient_2d(self, save_path=None):
        """Figura 2: mapa de Cp 2D (corte Z central)."""
        mid = self.nz // 2
        Cp = self.compute_cp()[:, :, mid].T
        Cp = gaussian_filter(Cp, sigma=0.6)  # smoothing

        fig, ax = plt.subplots(figsize=(7.5, 6.3))
        vmax = np.max(np.abs(Cp)) * 0.8
        im = ax.pcolormesh(self.x, self.y, Cp, shading='gouraud', cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax)
        half = self.cube_size / 2
        cx, cy, _ = self.cube_center
        ax.add_patch(plt.Rectangle((cx - half, cy - half), self.cube_size, self.cube_size,
                                    facecolor='gray', alpha=0.9, edgecolor='black', lw=1.5))
        ax.set_title(f"Pressure coefficient (Cp) - mid Z-plane\n"
                     f"Viscous flow around a cube, Re={self.Re}", fontsize=12)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_aspect('equal')
        cbar = plt.colorbar(im, ax=ax); cbar.set_label("Cp")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_pressure_coefficient_3d(self, save_path=None):
        """Figura 3: isosuperficies de Cp en 3D (marching cubes, scikit-image)."""
        Cp = gaussian_filter(self.compute_cp(), sigma=0.8)
        level_pos = 0.55 * Cp.max()
        level_neg = 0.55 * Cp.min()

        fig = plt.figure(figsize=(9, 7.5))
        ax = fig.add_subplot(111, projection='3d')

        def add_isosurface(field, level, color, alpha):
            verts, faces, _, _ = measure.marching_cubes(
                field, level=level, spacing=(self.dx, self.dy, self.dz)
            )
            verts = verts + np.array([self.x[0], self.y[0], self.z[0]])
            mesh = Poly3DCollection(verts[faces], alpha=alpha)
            mesh.set_facecolor(color)
            mesh.set_edgecolor('none')
            ax.add_collection3d(mesh)

        add_isosurface(Cp, level_pos, 'royalblue', 0.4)
        add_isosurface(Cp, level_neg, 'firebrick', 0.4)

        half = self.cube_size / 2
        cx, cy, cz = self.cube_center
        ax.bar3d(cx - half, cy - half, cz - half,
                  self.cube_size, self.cube_size, self.cube_size,
                  color='gray', alpha=0.95, shade=True)

        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(self.y[0], self.y[-1])
        ax.set_zlim(self.z[0], self.z[-1])
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f"3D pressure coefficient (Cp) isosurfaces\n"
                     f"blue: Cp~{level_pos:.2f} (stagnation)  ·  red: Cp~{level_neg:.2f} (suction)")
        ax.view_init(elev=22, azim=-55)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()



# Execution

if __name__ == "__main__":
    sim = ViscousFlowAroundCube(nx=35, ny=35, nz=35, Re=50)  # dt
    sim.solve(num_steps=600)

    print("\nGenerating graphs...")
    sim.plot_velocity_2d_slice(save_path="fig1_velocity_streamlines.png")
    sim.plot_pressure_coefficient_2d(save_path="fig2_cp_2d.png")
    sim.plot_pressure_coefficient_3d(save_path="fig3_cp_3d.png")
