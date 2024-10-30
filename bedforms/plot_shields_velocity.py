"""
This script plots the critical bed-shear velocity u_x_cr as a function of the dimensionless grain size Re_x
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81                 # gravitational acceleration (m/s^2)
d50 = 0.1e-3             # median grain diameter (d50) in meters
s = 2.68                 # specific gravity of sediment
S = 0.0001               # slope
Rh = 0.5                 # hydraulic radius approximation (m)
nu = 1e-6                # kinematic viscosity (m^2/s)
Re_x_min = 10            # minimum for Re_x - also for plot limits
Re_x_max = 10**5         # maximum for Re_x - also for plot limits

# Calculate Re_x
u_x = np.sqrt(g * Rh * S)  # shear velocity
Re_x_values = np.linspace(start=Re_x_min, stop=Re_x_max, num=Re_x_max - Re_x_min)  # Range for Re_* values

# Calculate u_{x,cr} based on Re_x
u_x_cr = np.sqrt(g * d50 * s * 0.5 * (0.22 * Re_x_values ** -0.6 + 0.06 * 10 ** (-7.7 * Re_x_values ** -0.6)))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(Re_x_values, u_x_cr, label=r"$u_{x,cr}$ as a function of $Re_*$")
plt.xscale("log")
plt.yscale("log")
plt.xlim(Re_x_min, Re_x_max)
plt.ylim(4*0.001, 0.01)
plt.xlabel(r"$Re_*$ (Reynolds number based on shear velocity)")
plt.ylabel(r"$u_{x,cr}$ (m/s)")
plt.title(r"Plot of $u_{x,cr}$ as a function of $Re_*$")
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.show()
