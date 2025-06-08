import numpy as np
import math
import pyvista as pv

# === Model parameters ===
P_v = 1.92e-9              # Base permeability [cm/s]
k = 1e-4             # Sensitivity to stiffness
t_leaflet = 0.0694         # Leaflet thickness (cm)
R = 1.0                    # Radius of leaflet (cm)


# Radial and vertical bounds
r_min, r_max = 0.0, R

# Layer fractions
f_fib = 0.44
f_vent = 0.25
f_spong = 1.0 - f_fib - f_vent

# Layer z cutoffs
z_ventricularis = f_vent * t_leaflet                        #z_vent = 0 - .017 cm
z_spongiosa     = f_spong * t_leaflet + z_ventricularis     # z_spon = .017 - .022 cm
                                                            #z_fib = .022 - .069 cm

[0.1, 0.2, 0.3]
[0.1, 0.3, 0.5]

# Stiffness in kPa
E_base   = 150
E_free   = 42.6
E_vent   = 26.9
E_spong  = 15.4
E_fib    = 37.1

# Exponential decay (precompute)
P_base  = P_v * np.exp(-k * E_base)
P_free  = P_v * np.exp(-k * E_free)
P_vent  = P_v * np.exp(-k * E_vent)
P_spong = P_v * np.exp(-k * E_spong)
P_fib   = P_v * np.exp(-k * E_fib)

# === Discretize space ===
Nx, Ny, Nz = 100, 100, 100  # theta, radial, z
theta = np.linspace(0, np.pi, Nx)
r = np.linspace(0, R, Ny)
z = np.linspace(0, t_leaflet, Nz)

# Create meshgrid
Theta_grid, R_grid, Z_grid = np.meshgrid(theta, r, z, indexing='ij')

X = R_grid * np.cos(Theta_grid)
Y = R_grid * np.sin(Theta_grid)
Z = Z_grid

# Define P field with old model (constant P)
P_field_old = np.full_like(Z, P_v)

# Update dimensions and reshape logic
points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = [Nx, Ny, Nz]  # [θ, r, z]

# === Compute radial and layered permeability ===
# Compute radial distance
radial_distance = R_grid

# Flip the interpolation so r = 0 is free edge, r = R is base
p_r = P_free + (P_base - P_free) * (radial_distance - r_min) / (r_max - r_min)

# Compute p_z based on z-layers
p_z = np.zeros_like(Z_grid)
p_z[Z_grid < z_ventricularis] = P_vent
p_z[(Z_grid >= z_ventricularis) & (Z_grid < z_spongiosa)] = P_spong
p_z[Z_grid >= z_spongiosa] = P_fib

# Equal weights (can be adjusted)
w_r = 1.0
w_z = 1.0

# Combine with Pythagorean formula
P_field_new = np.sqrt(w_r * p_r**2 + w_z * p_z**2)

# Final permeability field
# P_field = math.sqrt(p_r * p_r + p_z * p_z)


# === Scale for better visualization ===
# scaling_factor = 1e18
# P_field_scaled = P_field * scaling_factor


# === PyVista 3D structured grid ===
points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = [Ny, Nx, Nz]  # [r, θ, z]

# === Plot 3D heatmap of old model ===

plotter = pv.Plotter()
pmin, pmax = P_field_old.min(), P_field_old.max()
grid["Permeability"] = P_field_old.ravel()
plotter.add_mesh(grid, scalars="Permeability", cmap="gist_rainbow", clim=[pmin, pmax], scalar_bar_args={"title": "Permeability (cm/s)"})
plotter.view_xy()
plotter.add_title("3D LAD LDL Permeability Heatmap")
plotter.show()

# === Plot 3D heatmap of new model ===
plotter = pv.Plotter()
pmin, pmax = P_field_new.min(), P_field_new.max()
grid["Permeability"] = P_field_new.ravel()
plotter.add_mesh(grid, scalars="Permeability", cmap="gist_rainbow", clim=[pmin, pmax], scalar_bar_args={"title": "Permeability (cm/s)"})
plotter.view_xy()
plotter.add_title("3D AVS LDL Permeability Heatmap")
plotter.show()

# === Stretch factor for better z-visualization ===
z_scale = 10

# Update points with stretched Z
Z_stretched = Z * z_scale
points_stretched = np.c_[X.ravel(), Y.ravel(), Z_stretched.ravel()]
grid.points = points_stretched


# === Plot 3D heatmap ===
plotter = pv.Plotter()
pmin, pmax = P_field_new.min(), P_field_new.max()
plotter.add_mesh(grid, scalars="Permeability", cmap="gist_rainbow", clim=[pmin, pmax], scalar_bar_args={"title": "Permeability (cm/s)"})
plotter.view_xy()
plotter.add_title("3D AVS LDL Permeability Heatmap (Stretched)")
plotter.show()


# === Plot XY slices at different z-levels with stretch ===

# Choose which z-slices to visualize (by index)
slices_to_plot = [5, 50, 95]  # You can change these values to pick different depths

# Determine anatomical layer names for each selected z-slice
layer_names = []
for z_idx in slices_to_plot:
    z_val = z[z_idx]
    if z_val < z_ventricularis:
        layer_names.append("Ventricularis")
    elif z_val < z_spongiosa:
        layer_names.append("Spongiosa")
    else:
        layer_names.append("Fibrosa")

plotter = pv.Plotter(shape=(1, 3), window_size=(1200, 400))

for i, z_idx in enumerate(slices_to_plot):
    slice_data = P_field_new[:, :, z_idx]
    x_slice = X[:, :, z_idx]
    y_slice = Y[:, :, z_idx]
    z_value = z[z_idx] * z_scale

    # Create a 2D grid with a fixed Z
    grid2d = pv.StructuredGrid()
    grid2d.points = np.c_[x_slice.ravel(), y_slice.ravel(), np.full_like(x_slice.ravel(), z_value)]
    grid2d.dimensions = [Ny, Nx, 1]  # r, theta, 1 slice
    grid2d["P_slice"] = slice_data.ravel()  # Flatten to 1D

    plotter.subplot(0, i)
    plotter.add_mesh(
    grid2d,
    scalars="P_slice",
    cmap="gist_rainbow",
    show_edges=False,
    clim=[pmin, pmax],
    scalar_bar_args={"title": "Permeability (cm/s)"}
)  
    plotter.view_xy()
    plotter.add_title(f"{layer_names[i]}")

plotter.show()
