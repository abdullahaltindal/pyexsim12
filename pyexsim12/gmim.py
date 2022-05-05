"""
A module for calculating ground-motion intensity measures of pyexsim12.simulation.Simulation objects.
"""
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from pyexsim12 import _spectra
plt.interactive(True)


def _acc2vel(acc, dt):
    """
    Integrates acceleration record to obtain the velocity record.
    Args:
        acc: Acceleration record.
        dt: Time step (s).

    Returns:
        vel: Velocity series.
    """
    vel = np.append([0], integrate.cumtrapz(acc, dx=dt))
    return vel


def _acc2disp(acc, dt):
    """
    Integrates acceleration record twice to obtain the displacement record.
    Args:
        acc: Acceleration record.
        dt: Time step (s).

    Returns:
        disp: Displacement series.
    """
    vel = _acc2vel(acc, dt)
    disp = np.append([0], integrate.cumtrapz(vel, dx=dt))
    return disp


def pga(sim, site, filt_dict=None):
    """
    Peak ground acceleration at a given site for the Simulation object.
    Args:
        sim (Simulation): Simulation object.
        site (int): Site number.
        filt_dict (dict): Dictionary containing filter properties. If False, no filtering operations will be applied
            Missing keys will be replaced with default values. Filtering is applied with scipy.signal module. Keys are:
                "N": The order of the filter. Default is 4.
                "Wn": The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for
                       bandpass and bandstop filters, Wn is a length-2 sequence.
                "btype": btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}. The type of filter.
                                Default is 'bandpass'.
                "tukey": Shape parameter of the Tukey window, representing the fraction of the window inside the cosine
                tapered region.

    Returns:
        pga: Peak ground acceleration (cm/s/s)
    """

    _, acc = sim.get_acc(site=site, filt_dict=filt_dict)
    return np.max(np.abs(acc))


def pgv(sim, site, filt_dict=None):
    """
    Peak ground velocity at a given site for the Simulation object.
    sim (Simulation): Simulation object.
        site (int): Site number.
        filt_dict (dict): Dictionary containing filter properties. If False, no filtering operations will be applied
            Missing keys will be replaced with default values. Filtering is applied with scipy.signal module. Keys are:
                "N": The order of the filter. Default is 4.
                "Wn": The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for
                       bandpass and bandstop filters, Wn is a length-2 sequence.
                "btype": btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}. The type of filter.
                                Default is 'bandpass'.
                "tukey": Shape parameter of the Tukey window, representing the fraction of the window inside the cosine
                tapered region.

    Returns:
        pgv: Peak ground velocity (cm/s).
    """
    _, acc = sim.get_acc(site=site, filt_dict=filt_dict)
    dt = sim.path.time_pads.delta_t
    return np.max(np.abs(_acc2vel(acc, dt)))


def pgd(sim, site, filt_dict=None):
    """
    Peak ground displacement at a given site for the Simulation object.
    sim (Simulation): Simulation object.
        site (int): Site number.
        filt_dict (dict): Dictionary containing filter properties. If False, no filtering operations will be applied
            Missing keys will be replaced with default values. Filtering is applied with scipy.signal module. Keys are:
                "N": The order of the filter. Default is 4.
                "Wn": The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for
                       bandpass and bandstop filters, Wn is a length-2 sequence.
                "btype": btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}. The type of filter.
                                Default is 'bandpass'.
                "tukey": Shape parameter of the Tukey window, representing the fraction of the window inside the cosine
                tapered region.

    Returns:
        pgd: Peak ground displacement (cm).
    """
    _, acc = sim.get_acc(site=site, filt_dict=filt_dict)
    dt = sim.path.time_pads.delta_t
    return np.max(np.abs(_acc2disp(acc, dt)))


def pgv_over_pga(sim, site, filt_dict=None):
    """
    Ratio of peak ground velocity to peak ground acceleration.
    sim (Simulation): Simulation object.
        site (int): Site number.
        filt_dict (dict): Dictionary containing filter properties. If False, no filtering operations will be applied
            Missing keys will be replaced with default values. Filtering is applied with scipy.signal module. Keys are:
                "N": The order of the filter. Default is 4.
                "Wn": The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for
                       bandpass and bandstop filters, Wn is a length-2 sequence.
                "btype": btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}. The type of filter.
                                Default is 'bandpass'.
                "tukey": Shape parameter of the Tukey window, representing the fraction of the window inside the cosine
                tapered region.

    Returns:
        PGV / PGA (s).
    """
    return pgv(sim, site, filt_dict) / pga(sim, site, filt_dict)


def arias_intensity(sim, site, filt_dict=None):
    """
    Arias Intensity for the input ground acceleration record, which is defined as the time integral of
    the squared acceleration, multiplied by pi over two times gravitational acceleration.
    sim (Simulation): Simulation object.
        site (int): Site number.
        filt_dict (dict): Dictionary containing filter properties. If False, no filtering operations will be applied
            Missing keys will be replaced with default values. Filtering is applied with scipy.signal module. Keys are:
                "N": The order of the filter. Default is 4.
                "Wn": The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for
                       bandpass and bandstop filters, Wn is a length-2 sequence.
                "btype": btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}. The type of filter.
                                Default is 'bandpass'.
                "tukey": Shape parameter of the Tukey window, representing the fraction of the window inside the cosine
                tapered region.

    Returns:
        Arias intensity (cm/s).
    """
    g_const = 981
    _, acc = sim.get_acc(site=site, filt_dict=filt_dict)
    dt = sim.path.time_pads.delta_t
    return np.trapz(acc ** 2, dx=dt) * np.pi / 2 / g_const


def significant_duration(sim, site, filt_dict=None):
    """
    Significant duration for the input ground acceleration record, which is defined as the time between the
     exceedance of 5% and 95% of cumulative Arias Intensity.
    sim (Simulation): Simulation object.
        site (int): Site number.
        filt_dict (dict): Dictionary containing filter properties. If False, no filtering operations will be applied
            Missing keys will be replaced with default values. Filtering is applied with scipy.signal module. Keys are:
                "N": The order of the filter. Default is 4.
                "Wn": The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for
                       bandpass and bandstop filters, Wn is a length-2 sequence.
                "btype": btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}. The type of filter.
                                Default is 'bandpass'.
                "tukey": Shape parameter of the Tukey window, representing the fraction of the window inside the cosine
                tapered region.

    Returns:
        t_sign: Significant duration (s).
    """
    _, acc = sim.get_acc(site=site, filt_dict=filt_dict)
    dt = sim.path.time_pads.delta_t
    g_const = 981
    ia_vec = integrate.cumtrapz(acc ** 2, dx=dt) * np.pi / 2 / g_const
    ia_vec = np.append([0], ia_vec)
    ia_perc = ia_vec / max(ia_vec) * 100
    idx_5 = (np.abs(ia_perc - 5)).argmin()  # Index of closest value to 5 %
    idx_95 = (np.abs(ia_perc - 95)).argmin()  # Index of closest value to 95 %
    t_sign = (idx_95 - idx_5) * dt
    return t_sign


def cav(sim, site, filt_dict=None):
    """
    Cumulative absolute velocity for the input ground acceleration record, which is defined as the time
    integral of the absolute value of the input ground acceleration.
    sim (Simulation): Simulation object.
        site (int): Site number.
        filt_dict (dict): Dictionary containing filter properties. If False, no filtering operations will be applied
            Missing keys will be replaced with default values. Filtering is applied with scipy.signal module. Keys are:
                "N": The order of the filter. Default is 4.
                "Wn": The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for
                       bandpass and bandstop filters, Wn is a length-2 sequence.
                "btype": btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}. The type of filter.
                                Default is 'bandpass'.
                "tukey": Shape parameter of the Tukey window, representing the fraction of the window inside the cosine
                tapered region.

    Returns:
        Cumulative absolute velocity (cm/s).
    """
    _, acc = sim.get_acc(site=site, filt_dict=filt_dict)
    dt = sim.path.time_pads.delta_t
    return np.trapz(np.abs(acc), dx=dt)


def _response_spec(acc, dt, xi=0.05, periods=None):
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

    return _spectra.response_spec(acc, dt, xi, periods)


def predominant_period(sim, site, filt_dict=None):
    """
    Predominant period for the input ground acceleration record, defined as the period at the maximum spectral
    acceleration.
    sim (Simulation): Simulation object.
        site (int): Site number.
        filt_dict (dict): Dictionary containing filter properties. If False, no filtering operations will be applied
            Missing keys will be replaced with default values. Filtering is applied with scipy.signal module. Keys are:
                "N": The order of the filter. Default is 4.
                "Wn": The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for
                       bandpass and bandstop filters, Wn is a length-2 sequence.
                "btype": btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}. The type of filter.
                                Default is 'bandpass'.
                "tukey": Shape parameter of the Tukey window, representing the fraction of the window inside the cosine
                tapered region.

    Returns:
        tp: Predominant period (s).
    """
    _, acc = sim.get_acc(site=site, filt_dict=filt_dict)
    dt = sim.path.time_pads.delta_t
    period, sa = _response_spec(acc, dt)
    tp = period[np.argmax(sa)]
    return tp


def mean_period(sim, site, filt_dict=None):
    """
    Mean period for the input ground acceleration record as defined by Rathje et al. (1998).
    Reference:
    Rathje, E. M., Abrahamson, N. A., & Bray, J. D. (1998). Simplified frequency content estimates of earthquake ground
    motions. Journal of geotechnical and geoenvironmental engineering, 124(2), 150-159.
    sim (Simulation): Simulation object.
        site (int): Site number.
        filt_dict (dict): Dictionary containing filter properties. If False, no filtering operations will be applied
            Missing keys will be replaced with default values. Filtering is applied with scipy.signal module. Keys are:
                "N": The order of the filter. Default is 4.
                "Wn": The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for
                       bandpass and bandstop filters, Wn is a length-2 sequence.
                "btype": btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}. The type of filter.
                                Default is 'bandpass'.
                "tukey": Shape parameter of the Tukey window, representing the fraction of the window inside the cosine
                tapered region.

    Returns:
        t_m: Mean period (s).
    """
    _, acc = sim.get_acc(site=site, filt_dict=filt_dict)
    dt = sim.path.time_pads.delta_t
    freq, spec = _spectra.fourier_spec(acc, dt)
    cond = np.logical_and(freq < 20, freq > 0.25)
    freq = freq[cond]
    spec = spec[cond]
    t_m = np.sum(spec ** 2 / freq) / np.sum(spec ** 2)
    return t_m
