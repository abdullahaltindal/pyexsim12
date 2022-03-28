"""
Module for the implementation of the ground motion model proposed by Boore et al. (2014).
Reference:
Boore, D. M., Stewart, J. P., Seyhan, E., & Atkinson, G. M. (2014). NGA-West2 equations for predicting PGA, PGV, and
5% damped PSA for shallow crustal earthquakes. Earthquake Spectra, 30(3), 1057-1085.

Code is written by Abdullah Altindal, METU.
"""

import numpy as np
import pandas as pd

try:
    coeffs_table = pd.read_csv("BSSA14_coeffs.csv", index_col=0)
except FileNotFoundError:
    coeffs_table = pd.read_csv("./gmm/BSSA14_coeffs.csv", index_col=0)

periods = np.array([0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
                    4.0, 5.0, 7.5, 10])


def bssa14_vectorized(mw, r_jb, vs30, mech, region, *, z1=None, pga_ref=None, unit="cm"):
    """
    Vectorized version of the ground motion model of Boore et al. (2014), or BSSA14.
    Args:
        mw: Moment magnitude
        r_jb: Joyner-Boore distance (km)
        vs30: (m/s)
        mech: Faulting mechanism (0: Unspecified, SS: Strike-slip, N: Normal, R: Reverse)
        region: (0: Global, 1: Turkey & China, 2: Italy & Japan)
        z1: Depth from the ground surface to the 1.0 km∕s shear-wave horizon
        pga_ref: True when function is called for reference PGA calculation. Sets the site function to be equal to zero.
        unit: "cm" for cm/s/s, "g", for g.

    Returns:
        sa: Mean spectral acceleration array in input units.
        sa_plus_sigma: Mean + standard deviation acceleration array in input units.
        sa_minus_sigma: Mean - standard deviation acceleration array in input units.
    """

    ln_sa = np.array([bssa14_ln(mw, r_jb, vs30, t, mech, region, z1, pga_ref) for t in periods])
    sgm_ln_sa = np.array([bssa14_sigma(mw, r_jb, vs30, t) for t in periods])
    sa_g = np.exp(ln_sa)
    sa_g_plus_sigma = np.exp(ln_sa + sgm_ln_sa)
    sa_g_minus_sigma = np.exp(ln_sa - sgm_ln_sa)

    if unit == "cm":
        const = 981
    elif unit == "g":
        const = 1.0
    else:
        raise Exception("Unit not understood. Please enter 'cm' or 'g'.")

    sa = sa_g * const
    sa_plus_sigma = sa_g_plus_sigma * const
    sa_minus_sigma = sa_g_minus_sigma * const
    return sa, sa_plus_sigma, sa_minus_sigma


def bssa14_ln(mw, r_jb, vs30, period, mech, region, z1=None, pga_ref=None):
    """
    Ground motion model of Boore et al. (2014), also referred to as BSSA14.
    Args:
        mw: Moment magnitude
        r_jb: Joyner-Boore distance(km)
        vs30: (m/s)
        period: Vibration period (s)
        mech: Faulting mechanism (0: Unspecified, SS: Strike-slip, N: Normal, R: Reverse)
        region: (0: Global, 1: Turkey & China, 2: Italy & Japan)
        z1: Depth from the ground surface to the 1.0 km∕s shear-wave horizon
        pga_ref: True when function is called for reference PGA calculation. Sets the site function to be equal to zero.
    Returns:
        ln_y: Spectral acceleration in ln units of g.
    """

    coeffs = coeffs_table.loc[period]
    event = f_e(mw, mech, coeffs)
    path = f_p(r_jb, mw, region, coeffs)
    site = f_site(vs30, r_jb, mw, region, z1, coeffs, mech, period, pga_ref=pga_ref)
    ln_y = event + path + site

    return ln_y


def f_e(mw, mech, coeffs):
    """ Source function for BSSA14 ground motion model"""
    (u, ss, ns, rs) = (0, 0, 0, 0)
    if mech == 0:
        u = 1
    elif mech == "SS":
        ss = 1
    elif mech == "N":
        ns = 1
    elif mech == "R":
        rs = 1
    else:
        raise Exception("Please enter a valid faulting mechanism:\n "
                        "(0: Normal, 1: Reverse, 2: Strike-Slip)")

    if mw <= coeffs.m_h:
        return float(coeffs.e0 * u + coeffs.e1 * ss + coeffs.e2 * ns + coeffs.e3 * rs + coeffs.e4 * (
                mw - coeffs.m_h) + coeffs.e5 * (mw - coeffs.m_h) ** 2)
    else:
        return float(coeffs.e0 * u + coeffs.e1 * ss + coeffs.e2 * ns + coeffs.e3 * rs + coeffs.e6 * (mw - coeffs.m_h))


def f_p(r_jb, mw, region, coeffs):
    """ Path function for BSSA14 ground motion model """
    if region == 0:
        dc3 = coeffs.dc3_global
    elif region == 1:
        dc3 = coeffs.dc3_china_turkey
    elif region == 2:
        dc3 = coeffs.dc3_italy_japan
    else:
        raise Exception("Please enter a valid region")

    r = np.sqrt(r_jb ** 2 + coeffs.h ** 2)
    return float((coeffs.c1 + coeffs.c2 * (mw - coeffs.m_ref)) * np.log(r / coeffs.r_ref) + (coeffs.c3 + dc3) * (
            r - coeffs.r_ref))


def f_site(vs30, r_jb, mw, region, z1, coeffs, mech, period, pga_ref=True):
    """ Site function for BSSA14 ground motion model """
    if pga_ref:
        return 0

    if vs30 <= coeffs.v_c:
        f_lin = coeffs.c * np.log(vs30 / coeffs.v_ref)
    else:
        f_lin = coeffs.c * np.log(coeffs.v_c / coeffs.v_ref)
    # Nonlinear response
    ln_pga_r = bssa14_ln(mw, r_jb, 760, 0, mech, region, z1, pga_ref=True)
    pga_r = np.exp(ln_pga_r)

    f2 = coeffs.f4 * (np.exp(coeffs.f5 * (min(vs30, 760) - 360)) - np.exp(coeffs.f5 * (760 - 360)))

    f_nl = coeffs.f1 + f2 * np.log((pga_r + coeffs.f3) / coeffs.f3)

    # Basin effects
    if z1 is None:
        f_dz1 = 0
    else:
        ln_mu_z1 = -7.15 / 4 * np.log((vs30 ** 4 + 570.94 ** 4) / (1360 ** 4 + 570.94 ** 4)) - np.log(1000)
        mu_z1 = np.exp(ln_mu_z1)
        dz1 = z1 - mu_z1
        if period < 0.65:
            f_dz1 = 0
        elif period >= 0.65 and np.abs(dz1) <= coeffs.f7 / coeffs.f6:
            f_dz1 = coeffs.f6 * dz1
        elif period >= 0.65 and np.abs(dz1) > coeffs.f7 / coeffs.f6:
            f_dz1 = coeffs.f7
        else:
            raise Exception("Problem calculating basin effects.")

    f_s = float(f_lin + f_nl + f_dz1)
    return f_s


def bssa14_sigma(mw, r_jb, vs30, period):
    """ Variability function for BSSA14 ground motion model """
    # Magnitude dependent phi (within-event standard deviation)
    coeffs = coeffs_table.loc[period]
    if mw <= 4.5:
        phi_m = coeffs.phi1
    elif mw <= 5.5:
        phi_m = coeffs.phi1 + (coeffs.phi2 - coeffs.phi1) * (mw - 4.5)
    else:
        phi_m = coeffs.phi2

    # Magnitude and Rjb dependent phi
    if r_jb <= coeffs.r1:
        phi_m_rjb = phi_m
    elif r_jb <= coeffs.r2:
        phi_m_rjb = phi_m + coeffs.d_fr * (np.log(r_jb / coeffs.r1)) / (np.log(coeffs.r2 / coeffs.r1))
    else:
        phi_m_rjb = phi_m + coeffs.d_fr

    # Magnitude, Rjb and Vs30 dependent phi
    if vs30 >= coeffs.v_2:
        phi_m_rjb_vs = phi_m_rjb
    elif coeffs.v_2 >= vs30 >= coeffs.v_1:
        phi_m_rjb_vs = phi_m_rjb - coeffs.d_fv * (np.log(coeffs.v_2 / vs30)) / (np.log(coeffs.v_2 / coeffs.v_1))
    elif vs30 <= coeffs.v_1:
        phi_m_rjb_vs = phi_m_rjb - coeffs.d_fv
    else:
        raise Exception("Problem calculating standard deviation.")

    # Tao (between-event standard deviation)
    if mw <= 4.5:
        tao = coeffs.tao1
    elif mw <= 5.5:
        tao = coeffs.tao1 + (coeffs.tao2 - coeffs.tao1) * (mw - 4.5)
    else:
        tao = coeffs.tao2

    return np.sqrt(phi_m_rjb_vs ** 2 + tao ** 2)
