#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt

# USER INPUT PARAMETERS
USER_FLOW_VELOCITY = 2.1  # m/s
USER_WATER_DEPTH = 1.0    # m 
USER_D50 = 200e-6        # m (200 micro m, typical suspended sediment size)
USER_REF_HEIGHT = 0.05    # m (reference height above bed)
USER_REF_CONC = 0.9      # kg/m³ (typical reference concentration)

# CONSTANTS
RHO_F = 1000.0  # water density (kg/m³)
RHO_S = 2650.0  # sediment grain density (kg/m³)
MU = 0.001      # dynamic viscosity (Pa*s)
G = 9.81        # gravitational acceleration (m/s²)
KAPPA = 0.41    # von Karman constant (-)

# Set global font and style
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['font.family'] = 'Open Sans'
plt.rcParams['font.size'] = 12


def get_particle_settling_velocity(U, h, d, print_results=True):
    """Calculate particle settling velocity w_s using van Rijn (1984) method"""
    # Dimensionless particle diameter
    D_star = d * ((RHO_S - RHO_F) * G / (RHO_F * MU**2))**(1/3)
    
    # Calculate settling velocity based on D_star (van Rijn, 1984)
    if D_star <= 10:
        ws = ((RHO_S - RHO_F) * G * d**2) / (18 * MU)  # Stokes
    elif D_star <= 1000:
        ws = (MU/RHO_F) * (10.0/d) * (((1 + 0.01 * D_star**3)**0.5) - 1)
    else:
        ws = 1.1 * ((RHO_S - RHO_F) * G * d / RHO_F)**0.5
        
    # Apply turbulent correction
    Fr = U / math.sqrt(G * h)
    correction_factor = 1 / (1 + 0.2 * Fr**2)  # Modified correction factor
    ws_effective = ws * correction_factor

    if print_results:
        print(f"Settling velocity: {ws_effective:.3f} m/s")
        print(f"Dimensionless particle diameter: {D_star:.1f}")
        
    return ws_effective


def get_u_star(U, h, d):
    """Calculate friction velocity using Nikuradse roughness"""
    # Nikuradse roughness height
    ks = 2.5 * d
    
    # Friction coefficient using Colebrook-White
    f = 0.24 / (math.log10(12 * h / ks))**2
    
    # Compute u_star
    u_star = U * math.sqrt(f/8)
    return u_star


def plot_Rouse_profile(a=DEFAULT_REF_HEIGHT, c_a=DEFAULT_REF_CONC, 
                      w_s=None, u_star=None, h=DEFAULT_WATER_DEPTH,
                      show_plot=False):
    """
    Plot the Rouse profile for suspended sediment concentration
    """
    if w_s is None or u_star is None:
        U = USER_FLOW_VELOCITY
        d = DEFAULT_D50
        w_s = get_particle_settling_velocity(U, h, d, print_results=False)
        u_star = get_u_star(U, h, d)

    # Compute Rouse number
    beta = w_s / (KAPPA * u_star)
    print(f"Rouse number: {beta:.2f}")

    # Generate vertical positions
    z = np.linspace(a, h, 100)
    c = c_a * ((a * (h - z)) / (z * (h - a)))**beta

    # Create plot with wider aspect ratio
    fig, ax = plt.subplots(figsize=(10, 5))  # Changed to landscape format
    
    # Plot profile with thicker line
    ax.plot(c, z, 'k-', linewidth=2.5, label='Sediment concentration')
    
    # Labels and title with adjusted font sizes
    ax.set_xlabel('Suspended sediment concentration (kg/m³)')
    ax.set_ylabel('Height above bed (m)')
    
    # Grid and legend
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Set axis limits
    ax.set_xlim(0, round(max(c), 2))  # Slightly reduced right margin
    ax.set_ylim(0, h)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add parameter text box in the upper left
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    textstr = '\n'.join((
        r'Parameters:',
        r'$h=%.1f$ m' % (h, ),
        r'$u_*=%.2f$ m/s' % (u_star, ),
        r'$w_s=%.3f$ m/s' % (w_s, ),
        r'$\beta=%.2f$' % (beta, )))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Adjust layout
    plt.tight_layout()
    
    # Save with higher quality and exact size for A4 width
    plt.savefig('ssc-Rouse-profile.png', dpi=300, bbox_inches='tight',
                pad_inches=0.1)
    
    if show_plot:  # Only show if explicitly requested
        plt.show()
    else:
        plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    # Use default values for Dutch Rhine
    U = USER_FLOW_VELOCITY
    h = DEFAULT_WATER_DEPTH
    d = DEFAULT_D50
    
    print(f"Flow velocity: {U} m/s")
    print(f"Water depth: {h} m")
    print(f"Median grain size: {d*1e6:.0f} μm")
    
    # Calculate parameters
    w_s = get_particle_settling_velocity(U, h, d)
    u_star = get_u_star(U, h, d)
    
    # Plot profile with show_plot=True for command line usage
    plot_Rouse_profile(w_s=w_s, u_star=u_star, h=h, show_plot=True)

