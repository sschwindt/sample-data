"""
This script exemplifies the calculation of the workflow for estimating Manning's n for a channel
with ripple and dune bedforms
"""
import math

# Constants
g = 9.81  # gravitational acceleration in m/s^2
s = 2.65  # sediment density ratio (typical for sand in water)
nu = 1.0e-6  # kinematic viscosity of water in m^2/s

# User parameters
U = 1.0  # m/s, flow velocity
h = 0.5  # m, water depth (approximates hydraulic radius Rh)
S_0 = 0.0001  # slope
d_90 = 0.00001  # m

# Dimensionless Particle Diameter d_*
d_star = ((s - 1) * g / nu**2) ** (1/3) * d_90

# Critical Bed Shear Velocity u_{x, cr}
# Using simplified empirical equation based on grain size and slope
Re_star = (math.sqrt(g * h * S_0) * d_90) / nu
u_x_cr = math.sqrt(g * d_90 * s * 0.5 * (0.22 * Re_star ** -0.6 + 0.06 * 10 ** (-7.7 * Re_star ** -0.6)))

# Hydraulic Radius R and k_s
R = h  # Approximating hydraulic radius with depth
k_s = 3 * d_90  # Approximate bed roughness height based on d_50

# Iterative calculation for R_h_b based on the equation R_h_b = U^2 / (g * S_0 * [18 * log(12 * R_h_b / k_s)])
# Initialize according to recommendation
R_h_b = 1
delta_R_h_b = 1.0
tolerance = 1e-4
max_iterations = 10000
it = 0

while delta_R_h_b > tolerance and it < max_iterations:
    last_R_h_b = R_h_b
    # Update R_h_b using the current estimate
    R_h_b = (U**2) / (g * S_0 * 18 * math.log(12 * (last_R_h_b / k_s), 10))
    print(R_h_b)
    # Compute the relative change for convergence check
    delta_R_h_b = abs(R_h_b - last_R_h_b) / (last_R_h_b if last_R_h_b != 0 else 1)
    it += 1

# Convergence feedback
if delta_R_h_b > tolerance:
    print("Warning: Bed hydraulic radius iterations did not converge.")
else:
    print(f"Calculated bed hydraulic radius of R_h_b = {R_h_b:.6f} m with precision of {delta_R_h_b:.6e} after {it} iterations.")

# Bed-shear related grain velocity
u_x_gr = g**0.5 / (18 * math.log(12 * R_h_b / k_s, 10)) * U

# Transport stage parameter
if u_x_gr > u_x_cr:
    T_x = (u_x_gr**2 - u_x_cr**2) / u_x_cr**2
else:
    T_x = 0.  # set zero transport stage to avoid negative values

# Determine bedform type based on T_x and d_*
if T_x <= 1 and d_star <= 1:
    bedform_type = "Ripples"
elif (1 < T_x <= 15) or (d_star > 1 and T_x <= 15):
    bedform_type = "Dunes"
elif 15 < T_x <= 25:
    bedform_type = "Transitional (dunes-planed-bed)"
elif T_x > 25:
    bedform_type = "Plane-bed"
else:
    bedform_type = "Undefined"

# Bedform height
nu_h_b = 0.11 * h * (d_90 / h) ** 0.3 * (1 - math.exp(-0.5 * T_x)) * (25 - T_x)

# Bedform wavelength
lambda_b = 7.3 * h

# Total equivalent friction height of bedforms
k_s_b = k_s + 1.1 * nu_h_b * (1 - math.exp(-25 * nu_h_b / lambda_b))

# Chezy coefficient for beforms
C_bedform = 18 * math.log(12 * R_h_b / k_s_b, 10)

# Manning's n with modified hydraulic radius and total bedform roughness
n_bedform = R**(1/6) / C_bedform

# Display results
results = {
    "Dimensionless Particle Diameter d_*": d_star,
    "Critical Bed Shear Velocity u_{x, cr}": u_x_cr,
    "Iterative Hydraulic Radius for Bed R_{h, b}": R_h_b,
    "Transport stage parameter T_x": T_x,
    "Bedform type": bedform_type,
    "Equivalent roughness height k_s": k_s,
    "Total equivalent roughness height k_s:b": k_s_b,
    "Manning's n": n_bedform,
    "According Strickler k_st": 1/n_bedform
}

for key, value in results.items():
    print(f"{key}: {value}")
