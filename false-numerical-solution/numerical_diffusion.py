"""
Script creates a plot of numerical diffusion. The number of mesh points and / or domain size control for how strong
  the false numerical diffusion is.
The code is designed for Linux Ubuntu systems with the Ubuntu Light font type installed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Set global font properties
glob_font_size = 18
plt.rcParams['font.family'] = 'Ubuntu'
plt.rcParams['font.weight'] = 'light'
plt.rcParams['font.size'] = glob_font_size
plt.rcParams['axes.titlesize'] = glob_font_size
plt.rcParams['axes.labelsize'] = glob_font_size
plt.rcParams['xtick.labelsize'] = glob_font_size
plt.rcParams['ytick.labelsize'] = glob_font_size
plt.rcParams['legend.fontsize'] = glob_font_size
plt.rcParams['figure.titlesize'] = glob_font_size


def run_fdm_simulation(nx, ny, nt):
    Lx, Ly = 10.0, 10.0  # domain size
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # grid spacing
    dt = 0.01  # timestep

    # Generate the initial condition: a step function
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    u = np.ones((ny, nx))
    u[:, int(nx / 4):int(nx / 2)] = 2.0

    # Flatten the grid for triangulation
    x_flat = X.flatten()
    y_flat = Y.flatten()
    triang = tri.Triangulation(x_flat, y_flat)

    # Coefficients for advection (for simplicity, use constant advection speed)
    vx, vy = 1.0, 1.0

    # Perform the numerical simulation
    for n in range(nt):
        un = u.copy()
        u[1:, 1:] = (un[1:, 1:] - vx * dt / dx * (un[1:, 1:] - un[:-1, 1:])
                     - vy * dt / dy * (un[1:, 1:] - un[1:, :-1]))

        # Applying boundary conditions (Dirichlet)
        u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 1.0

    # Flatten the solution for plotting
    u_flat = u.flatten()

    return triang, u_flat


def run_fem_simulation(nx, ny, nt, dt, vx, vy):
    Lx, Ly = 10.0, 10.0
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # grid spacing
    X, Y = np.meshgrid(x, y)
    u = np.ones((ny, nx))
    u[:, int(nx / 4):int(nx / 2)] = 2.0

    x_flat = X.flatten()
    y_flat = Y.flatten()
    triang = tri.Triangulation(x_flat, y_flat)

    # Placeholder for global stiffness matrix and force vector assembly
    # Assuming a very simplified and symbolic form of FEM assembly
    # A proper FEM implementation would involve integration over elements,
    # shape functions, and their derivatives.

    for n in range(nt):
        u_new = u.copy()
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                u_new[i, j] = u[i, j] - vx * dt * (u[i, j] - u[i - 1, j]) / dx - vy * dt * (u[i, j] - u[i, j - 1]) / dy

        u = u_new

    u_flat = u.flatten()
    return triang, u_flat

# Parameters for the simulation
nt = 100  # number of timesteps

# Run FDM simulations with different mesh sizes
# triang1, u_flat1 = run_fdm_simulation(nx=10, ny=10, nt=nt)
# triang2, u_flat2 = run_fdm_simulation(nx=100, ny=100, nt=nt)
# Run FEM simulation
nx, ny = 10, 10
vx, vy = 1.0, 1.0
dt = 0.01
triang1, u_flat1 = run_fem_simulation(nx, ny, nt, dt, vx, vy)
triang2, u_flat2 = run_fem_simulation(nx * 10, ny * 10, nt, dt, vx, vy)

# Create the 2D plots with triangular mesh and varying color patterns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Plotting the triangular mesh with varying colors for the first subplot
tpc1 = ax1.tripcolor(triang1, u_flat1, shading='flat', cmap='seismic')
# cbar1 = plt.colorbar(tpc1, ax=ax1)
# cbar1.set_label('Concentration')
ax1.set_xlabel('Longitudinal domain width (m)')
ax1.set_ylabel('Lateral domain length (m)')
ax1.set_title('a) 10 x 10 nodes')
# ax1.grid(True)
ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True)

# Plotting the triangular mesh with varying colors for the second subplot
tpc2 = ax2.tripcolor(triang2, u_flat2, shading='flat', cmap='seismic')
cbar2 = plt.colorbar(tpc2, ax=ax2)
cbar2.set_label('Arbitrary tracer concentration')
ax2.set_xlabel('Longitudinal domain width (m)')
ax2.set_ylabel('Lateral domain length (m)')
ax2.set_title('b) 100 x 100 nodes')
# ax2.grid(True)
ax2.tick_params(axis='both', which='both', direction='in', top=True, right=True)

# Adjust layout
plt.tight_layout()

# Save the plot with 300 dpi resolution
plt.savefig('false-numerical-diffusion.png', dpi=300)

plt.show()
