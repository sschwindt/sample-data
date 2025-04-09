#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt

# USER INPUT PARAMETERS (defaults)
USER_FLOW_VELOCITY = 2.1   # m/s
USER_WATER_DEPTH   = 1.0   # m
USER_D50           = 200e-6 # m
USER_REF_HEIGHT    = 0.05  # m
USER_REF_CONC      = 0.90  # kg/m³

# CONSTANTS
RHO_F = 1000.0  # water density (kg/m³)
RHO_S = 2650.0  # sediment grain density (kg/m³)
MU    = 0.001   # dynamic viscosity (Pa·s)
G     = 9.81    # gravitational acceleration (m/s²)
KAPPA = 0.41    # von Kármán constant (-)

# Optional style settings
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['font.family'] = 'Open Sans'
plt.rcParams['font.size'] = 12


def get_particle_settling_velocity(U, h, d, print_results=True):
    """
    Calculate particle settling velocity w_s using a simplified 
    van Rijn (1984) approach.
    """
    # Dimensionless particle diameter
    D_star = d * ((RHO_S - RHO_F) * G / (RHO_F * MU**2))**(1/3)
    
    # Calculate settling velocity based on D_star
    if D_star <= 10:
        # Stokes
        ws = ((RHO_S - RHO_F) * G * d**2) / (18 * MU)
    elif D_star <= 1000:
        # Transitional
        ws = (MU / RHO_F) * (10.0 / d) * ((1 + 0.01 * D_star**3)**0.5 - 1)
    else:
        # Turbulent
        ws = 1.1 * ((RHO_S - RHO_F) * G * d / RHO_F)**0.5
    
    # Optional correction factor for turbulence intensity (semi-empirical)
    Fr = U / math.sqrt(G * h)  # Froude number
    correction_factor = 1 / (1 + 0.2 * Fr**2)
    ws_effective = ws * correction_factor

    if print_results:
        print(f"Settling velocity (w_s): {ws_effective:.4f} m/s")
        print(f"Dimensionless D*:       {D_star:.2f}")
    return ws_effective


def get_u_star(U, h, d):
    """
    Estimate friction velocity using a Colebrook-White friction factor
    with Nikuradse roughness ~ 2.5*d_50.
    """
    ks = 2.5 * d
    # Colebrook-White friction factor (rough, simplified)
    f = 0.24 / (math.log10(12 * h / ks))**2
    # Friction velocity
    u_star = U * math.sqrt(f/8)
    return u_star


def plot_rouse_profile(w_s,
                       u_star,
                       h,
                       a=USER_REF_HEIGHT,
                       c_a=USER_REF_CONC,
                       show_plot=False,
                       filename="ssc-Rouse-profile.png"):
    """
    Plot a single Rouse-type suspended sediment concentration profile 
    for a given settling velocity w_s, friction velocity u_star, 
    water depth h, reference height a, and reference concentration c_a.

    Saves output as 'ssc-Rouse-profile.png' unless 'filename' is changed.
    """
    # Compute Rouse number
    if u_star == 0:
        # Avoid division by zero
        beta = 0.0
    else:
        beta = w_s / (KAPPA * u_star)
    print(f"Rouse number (beta) = {beta:.3f}")
    
    # Create a vertical array from near-bed 'a' up to 'h'
    z = np.linspace(a, h, 100)
    # Rouse formula: C(z) = c_a * [ (a*(h - z)) / (z*(h - a)) ] ^ beta
    c = c_a * ((a * (h - z)) / (z * (h - a)))**beta

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(c, z, 'k-', linewidth=2.0, label='SSC profile')

    ax.set_xlabel("Suspended sediment concentration (kg/m³)")
    ax.set_ylabel("Height above bed (m)")
    ax.set_ylim([0, h])
    ax.grid(True)

    # Show a text box with relevant parameters
    textstr = '\n'.join((
        r'$h=%.2f$ m'    % (h, ),
        r'$a=%.2f$ m'    % (a, ),
        r'$c_a=%.2f$'    % (c_a, ),
        r'$u_* = %.3f$ m/s'  % (u_star,),
        r'$w_s = %.4f$ m/s'  % (w_s,),
        r'$Z = \frac{w_s}{\kappa\,u_*} = %.3f$' % (beta,),
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.legend(loc='upper right')

    plt.tight_layout()

    # Save figure
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_rouse_sweep(a=USER_REF_HEIGHT,
                     c_a=USER_REF_CONC,
                     h=USER_WATER_DEPTH,
                     z_points=100,
                     show_plot=True,
                     filename=None):
    """
    Sweep a range of Rouse numbers from 1/32 to 4, 
    and plot a family of curves in a single figure.
    """
    rouse_values = np.geomspace(1/32, 4, num=20)
    z = np.linspace(a, h, z_points)

    fig, ax = plt.subplots(figsize=(10,5))
    cmap = plt.cm.cool
    norm = plt.Normalize(vmin=rouse_values[0], vmax=rouse_values[-1])

    for Z in rouse_values:
        c = c_a * ((a*(h - z)) / (z*(h - a)))**Z
        color = cmap(norm(Z))
        ax.plot(c, z, color=color)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)  # attach colorbar to current axes
    cbar.set_label("Rouse number (Z)")

    ax.set_xlabel("Suspended sediment concentration (kg/m³)")
    ax.set_ylabel("Height above bed (m)")
    ax.set_xlim(left=0)
    ax.set_ylim([0, h])
    ax.set_title("Suspended Sediment Concentration Profiles\n(Rouse Number Sweep)")

    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()


# If you run this file directly for a quick test:
if __name__ == "__main__":
    U = USER_FLOW_VELOCITY
    h = USER_WATER_DEPTH
    d = USER_D50

    w_s = get_particle_settling_velocity(U, h, d)
    u_star = get_u_star(U, h, d)

    # Example single-profile plot
    plot_rouse_profile(w_s, u_star, h, 
                       a=USER_REF_HEIGHT, c_a=USER_REF_CONC,
                       show_plot=True, 
                       filename="ssc-Rouse-profile.png")

    # Example sweep
    plot_rouse_sweep(a=USER_REF_HEIGHT, 
                     c_a=USER_REF_CONC, 
                     h=USER_WATER_DEPTH,
                     show_plot=True, 
                     filename="ssc-Rouse-sweep.png")
