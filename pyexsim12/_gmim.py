"""

"""
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import _spectra
plt.interactive(True)


def acc2vel(acc, dt):
    """
    Integrates acceleration record to obtain the velocity record. If the acceleration record is in units of g, it is
    recommended to convert to cm/s**2 or m/s**2 to obtain consistent units.
    Args:
        acc: Acceleration record.
        dt: Time step (s).

    Returns:
        vel: Velocity record.
    """
    vel = np.append([0], integrate.cumtrapz(acc, dx=dt))
    return vel


def acc2disp(acc, dt):
    """
    Integrates acceleration record twice to obtain the displacement record. If the acceleration record is in units of g,
    it is recommended to convert to cm/s**2 or m/s**2 to obtain consistent units.
    Args:
        acc: Acceleration record.
        dt: Time step (s).

    Returns:
        disp: Displacement record.
    """
    vel = acc2vel(acc, dt)
    disp = np.append([0], integrate.cumtrapz(vel, dx=dt))
    return disp


def pga(acc):
    """
    Calculates the peak ground acceleration for the input ground acceleration record.
    Args:
        acc: Acceleration record.

    Returns:
        Peak ground acceleration.
    """
    return np.max(np.abs(acc))


def pgv(acc, dt):
    """
    Calculates the peak ground velocity for the input ground acceleration record.
    Args:
        acc: Acceleration record.
        dt: Time step (s).

    Returns:
        Peak ground velocity.
    """
    return np.max(np.abs(acc2vel(acc, dt)))


def pgd(acc, dt):
    """
    Calculates the peak ground displacement for the input ground acceleration record.
    Args:
        acc: Acceleration record.
        dt: Time step (s).

    Returns:
        Peak ground displacement.
    """
    return np.max(np.abs(acc2disp(acc, dt)))


def pgv_over_pga(acc, dt):
    """
    Ratio of peak ground velocity to peak ground acceleration in units of s.
    Args:
        acc: Acceleration record.
        dt: Time step (s).

    Returns:
        Peak ground velocity / Peak ground acceleration
    """
    return pgv(acc, dt) / pga(acc)


def arias_intensity(acc, dt, g_const=981):
    """
    Calculates the Arias Intensity for the input ground acceleration record, which is defined as the time integral of
    the squared acceleration, multiplied by pi over two times gravitational acceleration.
    Args:
        acc: Acceleration record.
        dt: Time step (s).
        g_const: Conversion constant for gravitational acceleration. Default is 981, for input acceleration units of
                  cm/s**2.
    Returns:
        Arias intensity.
    """
    return np.trapz(acc ** 2, dx=dt) * np.pi / 2 / g_const


def significant_duration(acc, dt, g_const=981):
    """
    Calculates the significant duration for the input ground acceleration record, which is defined as the time between
    the exceedance of 5% and 95% of Arias Intensity
    Args:
        acc: Acceleration record.
        dt: Time step (s).
        g_const: Conversion constant for gravitational acceleration. Default is 981, for input acceleration units of
                  cm/s**2.
    Returns:
        t_sign: Significant duration (s).
    """
    ia_vec = integrate.cumtrapz(acc ** 2, dx=dt) * np.pi / 2 / g_const
    ia_vec = np.append([0], ia_vec)
    ia_perc = ia_vec / max(ia_vec) * 100
    idx_5 = (np.abs(ia_perc - 5)).argmin()  # Index of closest value to 5 %
    idx_95 = (np.abs(ia_perc - 95)).argmin()  # Index of closest value to 95 %
    t_sign = (idx_95 - idx_5) * dt
    return t_sign


def cav(acc, dt):
    """
    Calculates the cumulative absolute velocity for the input ground acceleration record, which is defined as the time
    integral of the absolute value of the input ground acceleration.
    Args:
        acc: Acceleration record.
        dt: Time step (s).

    Returns:
        Cumulative absolute velocity.
    """
    return np.trapz(np.abs(acc), dx=dt)


def response_spec(acc: np.ndarray, dt: float, xi: float = 0.05, periods: np.ndarray = None):
    """
    Calculates the response spectrum for the input ground acceleration record, using Newmark's beta method for numerical
    integration. Uses numba.jit decorator to significantly increase computing performance. In order to take advantage of
    this performance increase, input acceleration and period arrays must be numpy arrays (np.ndarray).
    Args:
        acc (np.ndarray): Acceleration record.
        dt (float): Time step (s).
        xi (float): Ratio to critical damping. Default value is 0.05.
        periods (np.ndarray): (optional) Period array for response spectrum calculation.

    Returns:
        periods: Vibration periods for spectral acceleration calculation.
        spec_acc: Spectral acceleration values in cm/s**2.
    """
    return spectra.response_spec(acc, dt, xi, periods)


def predominant_period(acc, dt):
    """
    Predominant period for the input ground acceleration record, defined as the period at the maximum spectral
    acceleration.
    Args:
        acc: Acceleration record.
        dt: Time step (s).

    Returns:
        tp: Predominant period (s).
    """
    period, sa = response_spec(acc, dt)
    tp = period[np.argmax(sa)]
    return tp


def mean_period(acc: np.ndarray, dt):
    """
    Calculates the mean period for the input ground acceleration record, which is defined by Rathje et al. (1998)

    Reference:
    Rathje, E. M., Abrahamson, N. A., & Bray, J. D. (1998). Simplified frequency content estimates of earthquake ground
    motions. Journal of geotechnical and geoenvironmental engineering, 124(2), 150-159.

    Args:
        acc (np.ndarray): Acceleration record.
        dt: Time step (s).

    Returns:
        t_m: Mean period (s).
    """
    freq, spec = spectra.fourier_spec(acc, dt)
    cond = np.logical_and(freq < 20, freq > 0.25)
    freq = freq[cond]
    spec = spec[cond]
    t_m = np.sum(spec ** 2 / freq) / np.sum(spec ** 2)
    return t_m


def max_sa(acc: np.ndarray, dt):
    """
    Calculates the maximum spectral acceleration for the input ground acceleration.
    Args:
        acc (np.ndarray): Acceleration record.
        dt: Time step (s).

    Returns:
        Maximum spectral acceleration.
    """
    _, sa = response_spec(acc, dt)
    return max(sa)


def max_sa_over1(acc: np.ndarray, dt):
    """
    Calculates the ratio of maximum spectral acceleration to the spectral acceleration at 1-seconds, for the input
    ground acceleration.
    Args:
        acc (np.ndarray): Acceleration record.
        dt: Time step (s).

    Returns:
        Ratio of maximum spectral accelerations to the spectral acceleration at 1-seconds.
    """
    t, sa = response_spec(acc, dt)
    sa1 = sa[t == 1][0]
    max_sa_ = max(sa)
    return max_sa_ / sa1
