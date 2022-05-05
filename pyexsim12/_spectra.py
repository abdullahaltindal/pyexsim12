from numba import jit
import numpy as np


def response_spec(acc_g, dt, xi=0.05, periods=None):
    """
    Calculates the response spectrum for the Simulation object at the given site.
    Returns the periods and spectral acceleration values.
    Args:
        acc_g: Ground acceleration series
        dt: Time step (s)
        xi: Damping ratio. Default value is 0.05.
        periods: (optional) Vibration periods for spectral acceleration calculation.
    Returns:
        periods: Vibration periods for spectral acceleration calculation.
        spec_acc: Spectral acceleration values in cm/s/s
    """
    if periods is None:
        periods = np.concatenate(
            (np.linspace(0, 0.5, 501),
             np.linspace(0.51, 2, 150),
             np.linspace(2.1, 4, 20))
        )
    spec_acc = [spectral_acc(acc_g, dt, period, xi) for period in periods]
    return np.array(periods), np.array(spec_acc)


@jit()
def newmark(acc_g, dt, period, ksi=0.05, beta=0.25, gamma=0.5):
    """
    Solves the equation of motion for a SDOF system under ground excitation, with Newmark's Beta method.

    Args:
        acc_g: Ground acceleration
        dt: Time step
        period: Period of the SDOF system.
        ksi: Damping ratio. Default value is 0.05.
        beta: Beta parameter for the numerical calculation. Default value is 0.25.
        gamma: Gamma parameter for the numerical calculation. Default value is 0.5.

    Returns:
        disp: Ground displacement.
        vel: Ground velocity.
        acc_total: Total acceleration.
    """
    if period == 0:
        period = 0.001
    m = 1
    k = 4 * np.pi ** 2 * m / period ** 2
    c = ksi * 2 * np.sqrt(m * k)
    acc = np.zeros(len(acc_g))
    vel = np.zeros(len(acc_g))
    disp = np.zeros(len(acc_g))
    p_hat = np.zeros(len(acc_g))

    # Initial calculations
    acc[0] = (-acc_g[0]) / m
    a1 = m / beta / dt ** 2 + gamma * c / beta / dt
    a2 = m / beta / dt + (gamma / beta - 1) * c
    a3 = m * (1 / 2 / beta - 1) + dt * (gamma / 2 / beta - 1)
    k_hat = k + a1
    p = -m * acc_g

    for i, ag in enumerate(acc_g[:-1]):
        p_hat[i + 1] = p[i + 1] + a1 * disp[i] + a2 * vel[i] + a3 * acc[i]
        disp[i + 1] = p_hat[i + 1] / k_hat
        vel[i + 1] = gamma / beta / dt * (disp[i + 1] - disp[i]) + (1 - gamma / beta) * vel[i] + dt * (
                1 - gamma / 2 / beta) * acc[i]
        acc[i + 1] = 1 / beta / dt ** 2 * (disp[i + 1] - disp[i]) - vel[i] / beta / dt - (1 / 2 / beta - 1) * acc[i]
    acc_total = (acc + acc_g)
    return disp, vel, acc_total


@jit()
def spectral_acc(acc_g, dt, period, ksi=0.05, beta=0.25, gamma=0.5):
    """
    Calculates spectral acceleration corresponding to a given period, under ground excitation.

    Args:
        acc_g: Ground acceleration.
        dt: Time step
        period: Period of the SDOF system.
        ksi: Damping ratio. Default value is 0.05.
        beta: Beta parameter for the numerical calculation. Default value is 0.25.
        gamma: Gamma parameter for the numerical calculation. Default value is 0.5.

    Returns:
        sa: Spectral acceleration.
    """
    _, _, acc = newmark(acc_g, dt, period, ksi=ksi, beta=beta, gamma=gamma)
    sa = max(np.abs(acc))
    return sa


def fourier_spec(acc, dt):
    """
    Calculate the Fourier amplitude spectrum of the acceleration record using the fast-Fourier transformation algorithm
    Args:
        acc: Ground acceleration series.
        dt: Time step (s).

    Returns:
        freq: Frequency values in Hz.
        fas: Fourier amplitudes in cm/s.
    """
    length = len(acc)
    fas = np.fft.fft(acc)
    freq = np.linspace(0.0, 1 / (2 * dt), length // 2)
    fas = np.abs(fas[:length // 2]) * dt
    return freq, fas
