"""Analytical density and anisotropy profiles for dwarf spheroidal galaxies."""

import numpy as np
import scipy.special
from numpy.typing import NDArray


# =============================================================================
# Plummer profiles (stellar tracer)
# =============================================================================

def ln_Plummer2d(
    lnR: NDArray[np.floating],
    lnL: float,
    ln_a: float
) -> NDArray[np.floating]:
    """
    2D projected Plummer density profile (surface brightness).

    Equation: Sigma_star = L / (pi * a^2) * (1 + R^2/a^2)^{-2}

    Parameters
    ----------
    lnR : NDArray[np.floating]
        Log of projected radius.
    lnL : float
        Log of total luminosity.
    ln_a : float
        Log of Plummer scale radius.

    Returns
    -------
    NDArray[np.floating]
        Log of surface density Sigma(R).
    """
    lnSigma = lnL - np.log(np.pi) - 2 * ln_a - 2 * np.log1p(np.exp(2 * (lnR - ln_a)))
    return lnSigma


def ln_Plummer3d(
    lnr: NDArray[np.floating],
    lnL: float,
    ln_a: float
) -> NDArray[np.floating]:
    """
    3D Plummer density profile (deprojected).

    Equation: rho_star = 3L / (4 * pi * a^3) * (1 + r^2/a^2)^{-5/2}

    Parameters
    ----------
    lnr : NDArray[np.floating]
        Log of 3D radius.
    lnL : float
        Log of total luminosity.
    ln_a : float
        Log of Plummer scale radius.

    Returns
    -------
    NDArray[np.floating]
        Log of 3D density nu(r).
    """
    ln_nu = lnL - 3 * ln_a - (5 / 2) * np.log1p(np.exp(2 * (lnr - ln_a))) - np.log(4 * np.pi / 3)
    return ln_nu


# =============================================================================
# Generalized NFW profiles (dark matter)
# =============================================================================

# def ln_rho_gnfw(
#     ln_r: NDArray[np.floating],
#     ln_rho0: float,
#     ln_rdm: float,
#     gamma: float,
# ) -> NDArray[np.floating]:
#     """
#     Generalized NFW density profile.

#     Equation: rho(r) = rho0 / (r/rs)^gamma / (1 + r/rs)^(3-gamma)

#     Parameters
#     ----------
#     ln_r : NDArray[np.floating]
#         Log of 3D radius.
#     ln_rho0 : float
#         Log of characteristic density.
#     ln_rdm : float
#         Log of dark matter scale radius.
#     gamma : float
#         Inner slope parameter (gamma=1 for standard NFW).

#     Returns
#     -------
#     NDArray[np.floating]
#         Log of density rho(r).
#     """
#     x = ln_r - ln_rdm
#     ln_rho = ln_rho0 - gamma * x + (gamma - 3) * np.log1p(np.exp(x))
#     return ln_rho


# def ln_mass_gnfw(
#     lnr: NDArray[np.floating],
#     ln_rho0: float,
#     ln_rdm: float,
#     gamma: float,
#     alpha: float = 1.0,
#     beta: float = 3.0
# ) -> NDArray[np.floating]:
#     """
#     Enclosed mass for a generalized NFW profile.

#     Parameters
#     ----------
#     lnr : NDArray[np.floating]
#         Log of 3D radius.
#     ln_rho0 : float
#         Log of characteristic density.
#     ln_rdm : float
#         Log of dark matter scale radius.
#     gamma : float
#         Inner slope parameter.

#     Returns
#     -------
#     NDArray[np.floating]
#         Log of enclosed mass M(<r).
#     """
#     # lnx = lnr - ln_rdm
#     # prefactor = np.log(4 * np.pi) + ln_rho0 + 3 * ln_rdm - np.log(3 - gamma)
#     # exponent = 3 - gamma
#     # hypergeom = scipy.special.hyp2f1(exponent, exponent, 1 + exponent, -np.exp(lnx))
#     # ln_mass = prefactor + exponent * lnx + np.log(hypergeom)
#     # return ln_mass

def ln_rho_gnfw(
    ln_r: NDArray[np.floating],
    ln_rho0: float,
    ln_rdm: float,
    gamma: float,
    alpha: float = 1.0,
    beta: float = 3.0
) -> NDArray[np.floating]:
    """
    Generalized NFW density profile.

    Equation: rho(r) = rho0 / [(r/rs)^gamma * (1 + (r/rs)^alpha)^((beta-gamma)/alpha)]

    Parameters
    ----------
    ln_r : NDArray[np.floating]
        Log of 3D radius.
    ln_rho0 : float
        Log of characteristic density.
    ln_rdm : float
        Log of dark matter scale radius.
    gamma : float
        Inner slope parameter (gamma=1 for standard NFW).
    alpha : float
        Transition sharpness parameter (default: 1.0).
    beta : float
        Outer slope parameter (default: 3.0).

    Returns
    -------
    NDArray[np.floating]
        Log of density rho(r).
    """
    lnx = ln_r - ln_rdm  # ln(r/rs)

    # ln[rho(r)] = ln(rho0) - gamma*ln(r/rs) - ((beta-gamma)/alpha)*ln(1 + (r/rs)^alpha)
    ln_rho = ln_rho0 - gamma * lnx - ((beta - gamma) / alpha) * np.log1p(np.exp(alpha * lnx))

    return ln_rho


def ln_mass_gnfw(
    lnr: NDArray[np.floating],
    ln_rho0: float,
    ln_rdm: float,
    gamma: float,
    alpha: float = 1.0,
    beta: float = 3.0
) -> NDArray[np.floating]:
    """
    Enclosed mass for a generalized NFW profile.

    Parameters
    ----------
    lnr : NDArray[np.floating]
        Log of 3D radius.
    ln_rho0 : float
        Log of characteristic density.
    ln_rdm : float
        Log of dark matter scale radius.
    gamma : float
        Inner slope parameter.
    alpha : float
        Transition sharpness parameter (default: 1.0).
    beta : float
        Outer slope parameter (default: 3.0).

    Returns
    -------
    NDArray[np.floating]
        Log of enclosed mass M(<r).
    """
    lnx = lnr - ln_rdm  # ln(r/r_s)

    # Hypergeometric function arguments
    a1 = (3.0 - gamma) / alpha
    a2 = (beta - gamma) / alpha
    a3 = 1.0 + (3.0 - gamma) / alpha
    a4 = -np.exp(alpha * lnx)  # -(r/r_s)^alpha

    # Prefactor: ln[4pi rho_s r_s^3 / (3-gamma)]
    ln_prefactor = np.log(4 * np.pi) + ln_rho0 + 3 * ln_rdm - np.log(3.0 - gamma)

    # Power term: ln[(r/r_s)^(3-gamma)]
    ln_power = (3.0 - gamma) * lnx

    # Hypergeometric function (stays in linear space)
    hypergeom = scipy.special.hyp2f1(a1, a2, a3, a4)

    ln_mass = ln_prefactor + ln_power + np.log(hypergeom)

    return ln_mass


def ln_rhobar_gnfw(
    lnr: NDArray[np.floating],
    ln_rho0: float,
    ln_rdm: float,
    gamma: float
) -> NDArray[np.floating]:
    """
    Mean enclosed density for a generalized NFW profile.

    Parameters
    ----------
    lnr : NDArray[np.floating]
        Log of 3D radius.
    ln_rho0 : float
        Log of characteristic density.
    ln_rdm : float
        Log of dark matter scale radius.
    gamma : float
        Inner slope parameter.

    Returns
    -------
    NDArray[np.floating]
        Log of mean enclosed density rho_bar(<r).
    """
    ln_mass = ln_mass_gnfw(lnr, ln_rho0, ln_rdm, gamma)
    ln_rhobar = ln_mass - 3 * lnr - np.log(4 * np.pi / 3)
    return ln_rhobar


# =============================================================================
# Velocity anisotropy profiles
# =============================================================================

def beta_osipkov_merritt(
    lnr: NDArray[np.floating],
    beta_0: float,
    ln_ra: float
) -> NDArray[np.floating]:
    """
    Osipkov-Merritt velocity anisotropy parameter beta(r).

    Equation: beta(r) = (beta_0 + x^2) / (1 + x^2), where x = r/ra

    Parameters
    ----------
    lnr : NDArray[np.floating]
        Log of 3D radius.
    beta_0 : float
        Central anisotropy parameter (beta_0=0 for isotropic center).
    ln_ra : float
        Log of anisotropy radius.

    Returns
    -------
    NDArray[np.floating]
        Anisotropy parameter beta(r).
    """
    lnx2 = 2 * (lnr - ln_ra)
    x2 = np.exp(lnx2)
    # When x^2 >> 1: beta -> 1, when x^2 << 1: beta -> beta_0
    return np.where(lnx2 > 10, 1.0, (beta_0 + x2) / (1 + x2))


def ln_g_osipkov_merritt(
    lnr: NDArray[np.floating],
    beta_0: float,
    ln_ra: float
) -> NDArray[np.floating]:
    """
    Anisotropy integral g(r) for Osipkov-Merritt profile.

    Equation: g(r) = r^{2*beta_0} * (r^2 + ra^2)^{-(beta_0-1)}

    Parameters
    ----------
    lnr : NDArray[np.floating]
        Log of 3D radius.
    beta_0 : float
        Central anisotropy parameter.
    ln_ra : float
        Log of anisotropy radius.

    Returns
    -------
    NDArray[np.floating]
        Log of anisotropy integral g(r).
    """
    ra = np.exp(ln_ra)
    r = np.exp(lnr)

    # Compute in log space for stability
    # ln(g) = 2*beta_0*ln(r) - (beta_0-1)*ln(r^2 + ra^2)
    exponent = beta_0 * lnr - 0.5 * (beta_0 - 1) * np.log(r**2 + ra**2)
    return 2 * exponent
