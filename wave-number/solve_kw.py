import numpy as np

def wave_number(omega, h, g=9.81, tol=1e-12, max_iter=20, k0=None):
    """
    Solve omega^2 = g k tanh(k h) for k > 0 via Newton's method.

    Parameters
    ----------
    omega : float or ndarray
        Angular frequency [rad/s] (omega = 2*pi / T).
    h : float or ndarray
        Water depth [m].
    g : float, optional
        Gravity acceleration [m/s^2]. Default 9.81.
    tol : float, optional
        Relative tolerance for Newton updates.
    max_iter : int, optional
        Maximum iterations.
    k0 : float or ndarray, optional
        Optional initial guess for k [1/m]. If None, an automatic guess is used.

    Returns
    -------
    k : float or ndarray
        Wave number [1/m].
    """
    omega = np.asarray(omega, dtype=float)
    h = np.asarray(h, dtype=float)

    # Automatic initial guess if not provided
    if k0 is None:
        # Non-dimensional indicator x ~ (omega^2 h / g); small => shallow, large => deep
        x = (omega**2) * h / g
        k_shallow = omega / np.sqrt(g * h + 1e-300)  # avoid divide-by-zero
        k_deep = (omega**2) / g
        k = np.where(x < 1.0, k_shallow, k_deep)
    else:
        k = np.asarray(k0, dtype=float)

    # Newton iterations
    for _ in range(max_iter):
        kh = k * h
        tanh_kh = np.tanh(kh)
        sech2_kh = 1.0 / np.cosh(kh)**2

        f = omega**2 - g * k * tanh_kh
        # df/dk = -g [tanh(kh) + kh * sech^2(kh)]
        df = -g * (tanh_kh + kh * sech2_kh)

        # Newton update (guard against zero derivative)
        step = f / (df + 1e-300)
        k_new = k - step

        # keep k positive
        k_new = np.maximum(k_new, 1e-12)

        # convergence check (relative)
        if np.all(np.abs((k_new - k) / k_new) < tol):
            k = k_new
            break
        k = k_new

    return k

# --- Example usage ---
if __name__ == "__main__":
    T = 8.0           # peak (or monochromatic) period [s]
    h = 10.0          # depth [m]
    omega = 2*np.pi/T
    k = wave_number(omega, h)
    L = 2*np.pi / k   # wavelength [m]
    print(f"k = {k:.8f} 1/m,  L = {L:.3f} m,  kh = {k*h:.3f}")

