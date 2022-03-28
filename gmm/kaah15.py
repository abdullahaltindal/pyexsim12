"""
Module for the implementation of the ground motion model proposed by Kale et al. (2015).
Coefficients for this version are for Turkey only.
Delta terms can be added to give the option to include Iran as wel in the future.
Reference:
Kale, Ö., Akkar, S., Ansari, A., & Hamzehloo, H. (2015). A ground‐motion predictive model for Iran and Turkey for
horizontal PGA, PGV, and 5% damped response spectrum: Investigation of possible regional effects. Bulletin of the
Seismological Society of America, 105(2A), 963-980.

Code is written by Abdullah Altindal, METU.
"""

import numpy as np
import pandas as pd

try:
    coeffs_table = pd.read_csv("KAAH15_coeffs.csv", index_col=0)
except FileNotFoundError:
    coeffs_table = pd.read_csv("./gmm/KAAH15_coeffs.csv", index_col=0)

periods = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
                    0.18, 0.19, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44,
                    0.46, 0.48, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.1, 1.2,
                    1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4])


def kaah15_vectorized(mw, r_jb, vs30, mech, pga_ref=None, unit="cm"):
    """
    Vectorized version of the ground motion model of Kale et al. (2015), or KAAH15.
    Args:
        mw: Moment magnitude
        r_jb: Joyner-Boore distance (km)
        vs30: (m/s)
        mech: Style of faulting. (N: Normal, R: Reverse, SS: Strike-Slip)
        pga_ref: True when function is called for reference PGA calculation. Sets the site function to be equal to zero.
        unit: "cm" for cm/s/s, "g", for g.

    Returns:
        sa: Mean spectral acceleration array in input units.
        sa_plus_sigma: Mean + standard deviation acceleration array in input units.
        sa_minus_sigma: Mean - standard deviation acceleration array in input units.
    """
    ln_sa = np.array([kaah15_ln(mw, r_jb, vs30, t, mech, pga_ref=pga_ref) for t in periods])
    sgm_ln_sa = np.array([kaah15_sigma(mw, t) for t in periods])
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


def kaah15_ln(mw, r_jb, vs30, period, mech, pga_ref=None):
    """
    Ground motion model of Kale et al. (2015), also referred to as KAAH15.
    Args:
        mw: Moment magnitude
        r_jb: Joyner-Boore distance (km)
        vs30: (m/s)
        period: Vibration period (s)
        mech: Style of faulting. (N: Normal, R: Reverse, SS: Strike-Slip)
        pga_ref: True when function is called for reference PGA calculation. Sets the site function to be equal to zero.

    Returns:
        ln_y: Spectral acceleration in ln units of g.
    """
    # """
    # Ground motion model defined in Kale et al. (2015)
    # Parameters in this script are for Turkey only.
    #
    # :param mw: Moment magnitude
    # :param r_jb: Joyner-Boore distance (km)
    # :param vs30: (m/s)
    # :param period: Period (s)
    # :param mech: Style of faulting. (N: Normal, R: Reverse, SS: Strike-Slip)
    # :param pga_ref: Reference PGA
    # :return: lnY, sigma (in units of g)
    # """

    # Unpacking coefficients and initializing constants
    coeffs = coeffs_table.loc[period]
    v_con = 1000
    v_ref = 750
    c = 2.5
    n = 3.2
    c1 = 6.75  # Hinging magnitude

    # Magnitude scaling
    if mw <= c1:
        f_mag = coeffs.b1 + coeffs.b2 * (mw - c1) + coeffs.b3 * (8.5 - mw) ** 2
    else:
        f_mag = coeffs.b1 + coeffs.b7 * (mw - c1) + coeffs.b3 * (8.5 - mw) ** 2

    # Distance scaling
    f_dis = (coeffs.b4 + coeffs.b5 * (mw - c1)) * np.log(np.sqrt(r_jb ** 2 + coeffs.b6 ** 2))

    # Assigning dummy variables for style of faulting
    if mech == "N":
        f_nm = 1
        f_rv = 0
    elif mech == "R":
        f_nm = 0
        f_rv = 1
    else:
        f_nm = 0
        f_rv = 0

    # Style of Faulting
    f_sof = coeffs.b8 * f_nm + coeffs.b9 * f_rv

    # Anelastic attenuation
    if r_jb <= 80:
        f_aat = 0
    else:
        f_aat = coeffs.b10 * (r_jb - 80)

    # Calculating reference PGA for ref. conditions
    if not pga_ref:
        pga_ref = np.exp(kaah15_ln(mw, r_jb, 750, 0, mech, pga_ref=True))

    # Site amplification
    if vs30 < v_ref:
        f_site = coeffs.sb1 * np.log(vs30 / v_ref) + coeffs.sb2 * np.log(
            (pga_ref + c * (vs30 / v_ref) ** n) / ((pga_ref + c) * (vs30 / v_ref) ** n))
    elif vs30 > v_ref:
        f_site = coeffs.sb1 * np.log((min(vs30, v_con)) / v_ref)
    else:
        f_site = 0

    # Median amplitude
    ln_y = f_mag + f_dis + f_sof + f_aat + f_site
    return ln_y


def kaah15_sigma(mw, period):
    """
    Variability function for the ground motion model of Kale et al. (2015)
    Args:
        mw: Moment magnitude.
        period: Vibration period (s)

    Returns:
        sigma: Total aleatory variability
    """
    coeffs = coeffs_table.loc[period]
    if mw < 6.0:
        w = coeffs.a1
    elif mw < 6.5:
        w = coeffs.a1 + coeffs.a2 - coeffs.a1 * (mw - 6) * 0.5
    else:
        w = coeffs.a2

    phi = w * coeffs.sd1  # intra-event variability (within-event)
    tao = w * coeffs.sd2  # inter-event variability (between-event)
    sigma = np.sqrt(phi ** 2 + tao ** 2)  # Total aleatory variability
    return sigma
