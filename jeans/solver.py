"""Jeans equation solver for spherical systems."""

import numpy as np
from numpy.typing import NDArray
from scipy import integrate
import astropy.units as u
import astropy.constants as const

from ..profiles import (
    ln_Plummer2d,
    ln_Plummer3d,
    ln_mass_gnfw,
    beta_osipkov_merritt,
    ln_g_osipkov_merritt,
)


def calc_ln_sigma2_nu(
    lnr: NDArray[np.floating],
    ln_mass: NDArray[np.floating],
    ln_nu: NDArray[np.floating],
    ln_gbeta: NDArray[np.floating],
    method: str = 'trapezoid'
) -> NDArray[np.floating]:
    """
    Calculate 3D velocity dispersion profile from Jeans equation.

    Parameters
    ----------
    lnr : NDArray[np.floating]
        Log of 3D radius grid.
    ln_mass : NDArray[np.floating]
        Log of enclosed mass M(<r) at each radius.
    ln_nu : NDArray[np.floating]
        Log of 3D tracer density nu(r) at each radius.
    ln_gbeta : NDArray[np.floating]
        Log of anisotropy integral g(r) at each radius.
    method : str, optional
        Integration method: 'trapezoid' or 'simpson'. Default is 'trapezoid'.

    Returns
    -------
    NDArray[np.floating]
        Log of sigma^2(r) * nu(r).
    """
    # Compute integrand in log space: ln(M * nu * g / r^2)
    ln_inte = ln_mass + ln_nu + ln_gbeta - 2 * lnr

    # Subtract maximum for numerical stability
    ln_inte_max = np.nanmax(ln_inte)
    ln_inte_safe = ln_inte - ln_inte_max

    # Clip extreme values
    ln_inte_safe = np.clip(ln_inte_safe, -700, 700)
    inte = np.exp(ln_inte_safe)

    r = np.exp(lnr)

    if method == 'simpson' and len(r) > 2:
        # Simpson's rule (more accurate for smooth functions)
        result_cumsum = np.zeros_like(inte)

        # Compute cumulative integral using Simpson's rule
        for i in range(len(r)):
            if i < len(r) - 1:
                result_cumsum[i] = integrate.simpson(inte[i:], x=r[i:])
            else:
                result_cumsum[i] = 0.0
    else:
        # Trapezoidal rule (more stable)
        result = integrate.cumulative_trapezoid(inte, x=r, initial=0.0)

        # Integral from r to infinity: I(∞) - I(r)
        result_cumsum = result[-1] - result

    # Handle edge cases
    result_cumsum = np.maximum(result_cumsum, 1e-300)

    # Return to log space
    ln_sigma2_nu = np.log(result_cumsum) + ln_inte_max - ln_gbeta

    return ln_sigma2_nu


def calc_ln_sigma2p_Sigma(
    lnR: NDArray[np.floating],
    lnr: NDArray[np.floating],
    ln_sigma2_nu: NDArray[np.floating],
    beta: NDArray[np.floating],
    min_rminR2: float = 1e-6,
    method: str = 'trapezoid'
) -> NDArray[np.floating]:
    """
    Calculate projected velocity dispersion profile.

    Parameters
    ----------
    lnR : NDArray[np.floating]
        Log of projected radius grid (1D array).
    lnr : NDArray[np.floating]
        Log of 3D radius grid (1D array).
    ln_sigma2_nu : NDArray[np.floating]
        Log of sigma^2 * nu from 3D Jeans solution.
    beta : NDArray[np.floating]
        Anisotropy parameter at each 3D radius.
    min_rminR2 : float, optional
        Minimum value for r^2 - R^2 to avoid singularity. Default is 1e-6.
    method : str, optional
        Integration method: 'trapezoid' or 'simpson'. Default is 'trapezoid'.

    Returns
    -------
    NDArray[np.floating]
        Log of sigma_p^2(R) * Sigma(R).
    """
    lnR = lnR[:, None]
    lnr = lnr[None, :]
    R = np.exp(lnR)
    r = np.exp(lnr)

    # Handle r^2 - R^2 term carefully near r = R
    rminR2 = r**2 - R**2

    # Set minimum separation to avoid singularity
    rminR2 = np.where((rminR2 > 0) & (rminR2 < min_rminR2), min_rminR2, rminR2)

    beta_grid = beta[None, :]

    # Compute integrand: (1 - beta*R^2/r^2) * sigma^2 * nu * r / sqrt(r^2 - R^2)
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = (1 - beta_grid * (R / r)**2)

        # For numerical stability, clip the factor
        factor = np.clip(factor, -10, 10)

        inte = factor * np.exp(ln_sigma2_nu + lnr)

        # Only integrate where r > R
        sqrt_term = np.sqrt(np.maximum(rminR2, 0))
        inte = np.where(rminR2 > 0, inte / sqrt_term, 0)

    # Replace NaN and Inf with zeros
    inte = np.nan_to_num(inte, nan=0.0, posinf=0.0, neginf=0.0)

    # Perform integration for each R
    if method == 'simpson' and r.shape[1] > 2:
        sigma2p_Sigma = 2 * integrate.simpson(inte, x=r, axis=1)
    else:
        sigma2p_Sigma = 2 * integrate.trapezoid(inte, x=r, axis=1)

    # Ensure positive values
    sigma2p_Sigma = np.maximum(sigma2p_Sigma, 1e-300)

    ln_sigma2p_Sigma = np.log(sigma2p_Sigma)

    return ln_sigma2p_Sigma


def calc_sigma2_los(
    lnR: NDArray[np.floating],
    gamma: float,
    ln_rdm: float,
    ln_rho0: float,
    beta_0: float,
    ln_ra: float,
    ln_a: float,
    alpha: float = 1.0,
    beta: float = 3.0,
    rint: NDArray[np.floating] | None = None,
    rint_min: float = 1e-2,
    rint_max: float = 10,
    nint: int = 1000,
    integration_method: str = 'trapezoid',
    use_log_spacing: bool = True
) -> NDArray[np.floating]:
    """
    Calculate line-of-sight velocity dispersion squared.

    Solves the spherical Jeans equation for a Plummer stellar tracer
    in a generalized NFW dark matter halo with Osipkov-Merritt anisotropy.

    Parameters
    ----------
    lnR : NDArray[np.floating]
        Log of projected radius at which to compute sigma_los^2.
    gamma : float
        Inner slope of gNFW profile.
    beta : float
        Outer slope of gNFW profile.
    alpha : float
        Transition slope of gNFW profile.
    ln_rdm : float
        Log of dark matter scale radius.
    ln_rho0 : float
        Log of dark matter characteristic density.
    beta_0 : float
        Central velocity anisotropy.
    ln_ra : float
        Log of anisotropy radius.
    ln_a : float
        Log of Plummer scale radius.
    rint : NDArray[np.floating] | None, optional
        3D radius integration grid. If None, generated from rint_min/max.
    rint_min : float, optional
        Minimum integration radius. Default is 1e-2.
    rint_max : float, optional
        Maximum integration radius. Default is 10.
    nint : int, optional
        Number of integration points. Default is 1000.
    integration_method : str, optional
        Integration method: 'trapezoid' or 'simpson'. Default is 'trapezoid'.
    use_log_spacing : bool, optional
        Use logarithmic spacing for integration grid. Default is True.

    Returns
    -------
    NDArray[np.floating]
        Line-of-sight velocity dispersion squared in (km/s)^2.
    """
    if rint is None:
        if use_log_spacing:
            # Log spacing is better for wide dynamic ranges
            ln_rint = np.linspace(np.log(rint_min), np.log(rint_max), nint)
            rint = np.exp(ln_rint)
        else:
            # Linear spacing
            rint = np.linspace(rint_min, rint_max, nint)

    ln_rint = np.log(rint)

    # Calculate the velocity anisotropy and anisotropy integral g(r)
    beta_ani = beta_osipkov_merritt(ln_rint, beta_0, ln_ra)
    ln_gbeta_ani = ln_g_osipkov_merritt(ln_rint, beta_0, ln_ra)

    # 2D and 3D tracer density profiles
    # set ln_L to zero since it cancels out anyway in the final sigma2p
    ln_Sigma = ln_Plummer2d(lnR, 0, ln_a)
    ln_nu = ln_Plummer3d(ln_rint, 0, ln_a)

    # Enclosed mass profile
    ln_mass = ln_mass_gnfw(ln_rint, ln_rho0, ln_rdm, gamma, alpha, beta)

    # 3D velocity dispersion profile
    ln_sigma2_nu = calc_ln_sigma2_nu(
        ln_rint, ln_mass, ln_nu, ln_gbeta_ani, method=integration_method)

    # Projected velocity dispersion
    ln_sigma2p_Sigma = calc_ln_sigma2p_Sigma(
        lnR, ln_rint, ln_sigma2_nu, beta_ani, method=integration_method)
    ln_sigma2p = ln_sigma2p_Sigma - ln_Sigma

    # Unit rescaling: convert to (km/s)^2
    G = const.G.to_value(u.kpc**3 / u.Msun / u.s**2)
    kpc_to_km = (u.kpc).to(u.km)
    ln_sigma2p = ln_sigma2p + np.log(G) + 2 * np.log(kpc_to_km)

    sigma2p = np.exp(ln_sigma2p)

    # Final sanity check
    sigma2p = np.nan_to_num(sigma2p, nan=0.0, posinf=1e10, neginf=0.0)

    return sigma2p

def calc_vsp1(
    gamma: float,
    ln_rdm: float,
    ln_rho0: float,
    beta_0: float,
    ln_ra: float,
    ln_a: float,
    alpha: float = 1.0,
    beta: float = 3.0,
):
    rint = np.logspace(np.log10(0.001), np.log10(50), 1000)
    ln_rint = np.log(rint)

    # Calculate the velocity anisotropy and anisotropy integral g(r)
    beta_ani = beta_osipkov_merritt(ln_rint, beta_0, ln_ra)
    ln_gbeta_ani = ln_g_osipkov_merritt(ln_rint, beta_0, ln_ra)

    # 2D and 3D tracer density profiles
    # set ln_L to zero since it cancels out anyway in the final sigma2p
    ln_nu = ln_Plummer3d(ln_rint, 0, ln_a)

    # Enclosed mass profile
    ln_mass = ln_mass_gnfw(ln_rint, ln_rho0, ln_rdm, gamma, alpha, beta)
    ln_sigma2_nu = calc_ln_sigma2_nu(
        ln_rint, ln_mass, ln_nu, ln_gbeta_ani, method='trapezoid')

    # vsp1 integrand
    # NOTE: ln_sigma2_nu ignore a factor of G, so this needs to be readded
    G = const.G.to_value(u.kpc**3 / u.Msun / u.s**2)
    kpc_to_km = (u.kpc).to(u.km)
    ln_vsp1_integrand = 2 * np.log(G) + ln_mass + ln_sigma2_nu + np.log(5 - 2 * beta_ani) + np.log(2 / 5) + ln_rint
    vsp1 = np.trapz(np.exp(ln_vsp1_integrand), rint)
    vsp1 = vsp1 * (kpc_to_km)**4

    return vsp1

def calc_vsp2(
    gamma: float,
    ln_rdm: float,
    ln_rho0: float,
    beta_0: float,
    ln_ra: float,
    ln_a: float,
    alpha: float = 1.0,
    beta: float = 3.0,
):
    rint = np.logspace(np.log10(0.001), np.log10(100), 1000)
    ln_rint = np.log(rint)

    # Calculate the velocity anisotropy and anisotropy integral g(r)
    beta_ani = beta_osipkov_merritt(ln_rint, beta_0, ln_ra)
    ln_gbeta_ani = ln_g_osipkov_merritt(ln_rint, beta_0, ln_ra)

    # 2D and 3D tracer density profiles
    # set ln_L to zero since it cancels out anyway in the final sigma2p
    ln_nu = ln_Plummer3d(ln_rint, 0, ln_a)

    # Enclosed mass profile
    ln_mass = ln_mass_gnfw(ln_rint, ln_rho0, ln_rdm, gamma, alpha, beta)
    ln_sigma2_nu = calc_ln_sigma2_nu(
        ln_rint, ln_mass, ln_nu, ln_gbeta_ani, method='trapezoid')

    # vsp2 integrand
    # NOTE: ln_sigma2_nu ignore a factor of G, so this needs to be readded
    G = const.G.to_value(u.kpc**3 / u.Msun / u.s**2)
    kpc_to_km = (u.kpc).to(u.km)
    ln_vsp2_integrand = 2 * np.log(G) + ln_mass + ln_sigma2_nu + np.log(7 - 6 * beta_ani) + np.log(4 / 35) + 3 * ln_rint
    vsp2 = np.trapz(np.exp(ln_vsp2_integrand), rint)
    vsp2 = vsp2 * (kpc_to_km)**4

    return vsp2