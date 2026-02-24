"""Velocity dispersion analysis for dwarf galaxies."""

from typing import Tuple, Optional

import astropy.coordinates as acoo
import astropy.units as auni
import emcee
import numpy as np
from numpy.typing import NDArray

def log_gauss_1d(
    x: NDArray[np.floating], mu: float, sigma: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Compute the log of 1D Gaussian probability density function.

    Parameters
    ----------
    x : NDArray[np.floating]
        Data points.
    mu : float
        Mean of the Gaussian.
    sigma : NDArray[np.floating]
        Standard deviation of the Gaussian.

    Returns
    -------
    NDArray[np.floating]
        Log probability density at each point.
    """
    return -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((x - mu) / sigma) ** 2


def log_prior(params: Tuple[float, float]) -> float:
    """
    Compute log-prior for velocity dispersion model parameters.

    Uses flat priors: mu in (-100, 100) km/s, log_sigma in (-5, 5).

    Parameters
    ----------
    params : Tuple[float, float]
        Model parameters (mu, log_sigma).

    Returns
    -------
    float
        Log-prior probability (0 if within bounds, -inf otherwise).
    """
    mu, log_sigma = params
    if -100.0 < mu < 100.0 and -5.0 < log_sigma < 5.0:
        return 0.0
    return -np.inf


def log_likelihood(
    params: Tuple[float, float],
    data: Tuple[NDArray[np.floating], NDArray[np.floating]],
) -> float:
    """
    Compute log-likelihood for velocity dispersion model.

    Parameters
    ----------
    params : Tuple[float, float]
        Model parameters (mu, log_sigma) where mu is systemic velocity
        and log_sigma is log of intrinsic velocity dispersion.
    data : Tuple[NDArray[np.floating], NDArray[np.floating]]
        Tuple of (velocities, velocity_errors) in km/s.

    Returns
    -------
    float
        Log-likelihood value.
    """
    mu, log_sigma = params
    vlos, vlos_error = data

    var = np.exp(log_sigma) ** 2 + vlos_error**2
    return np.sum(log_gauss_1d(vlos, mu, np.sqrt(var)))


def log_posterior(
    params: Tuple[float, float],
    data: Tuple[NDArray[np.floating], NDArray[np.floating]],
) -> float:
    """
    Compute log-posterior for velocity dispersion model.

    Parameters
    ----------
    params : Tuple[float, float]
        Model parameters (mu, log_sigma).
    data : Tuple[NDArray[np.floating], NDArray[np.floating]]
        Tuple of (velocities, velocity_errors) in km/s.

    Returns
    -------
    float
        Log-posterior probability.
    """
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, data)


def fit_vdisp_los(
    vr: NDArray[np.floating],
    vr_err: NDArray[np.floating],
    nwalkers: int = 8,
    nsteps: int = 1000,
    auto_extend: bool = True,
    max_steps: int = 10000,
    convergence_factor: float = 50.0,
    verbose: bool = True,
) -> NDArray[np.floating]:
    """
    Fit velocity dispersion profile using MCMC.

    Parameters
    ----------
    vr : NDArray[np.floating]
        Radial velocities in km/s.
    vr_err : NDArray[np.floating]
        Velocity uncertainties in km/s.
    nwalkers : int, optional
        Number of MCMC walkers, by default 8.
    nsteps : int, optional
        Number of MCMC steps per walker, by default 1000.
    auto_extend : bool, optional
        If True, automatically extend the chain if autocorrelation time
        indicates insufficient samples, by default True.
    max_steps : int, optional
        Maximum total steps when auto-extending, by default 10000.
    convergence_factor : float, optional
        Chain is considered converged when n_steps > convergence_factor * tau,
        by default 50.0.

    Returns
    -------
    NDArray[np.floating]
        Flattened MCMC samples of shape (n_samples, 2) with columns [mu, log_sigma].
    """
    ndim = 2
    p0 = np.random.rand(nwalkers, ndim)
    p0 = p0 * np.array([200.0, 10.0]) - np.array([100.0, 5.0])

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, args=[(vr, vr_err)]
    )
    sampler.run_mcmc(p0, nsteps, progress=True)

    # Check convergence with autocorrelation time
    total_steps = nsteps
    converged = False

    while not converged:
        try:
            tau = sampler.get_autocorr_time()
            converged = np.all(total_steps > convergence_factor * tau)

            if not converged and auto_extend:
                if total_steps >= max_steps:
                    print(
                        f"Warning: Reached max_steps={max_steps} without full convergence. "
                        f"Current tau={tau}, need {convergence_factor}*tau steps."
                    )
                    break

                # Extend by the amount needed to reach convergence
                steps_needed = int(convergence_factor * np.max(tau)) - total_steps
                extend_steps = min(steps_needed, max_steps - total_steps)
                extend_steps = max(extend_steps, nsteps)  # At least nsteps more

                print(
                    f"Chain not converged (n={total_steps}, tau={np.max(tau):.1f}). "
                    f"Extending by {extend_steps} steps..."
                )
                sampler.run_mcmc(None, extend_steps, progress=verbose)
                total_steps += extend_steps
            elif not converged:
                print(
                    f"Warning: Chain may not be converged. "
                    f"n_steps={total_steps}, tau={tau}. Consider increasing nsteps."
                )
                break

        except emcee.autocorr.AutocorrError as e:
            if auto_extend and total_steps < max_steps:
                extend_steps = min(nsteps, max_steps - total_steps)
                print(
                    f"Autocorrelation time estimation failed: {e}. "
                    f"Extending chain by {extend_steps} steps..."
                )
                sampler.run_mcmc(None, extend_steps, progress=verbose)
                total_steps += extend_steps
            else:
                print(f"Warning: Could not estimate autocorrelation time: {e}")
                tau = sampler.get_autocorr_time(quiet=True)
                break

    # Discard burn-in (at least 2-3 times max tau) and thin
    burnin = int(3 * np.max(tau))
    thin = max(1, int(0.5 * np.min(tau)))
    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

    if verbose:
        print(f"Autocorrelation times: {tau}")
        print(f"Mean tau: {np.mean(tau):.1f} steps (total chain: {total_steps} steps)")
        print(f"Discarded {burnin} steps, thinned by {thin}")
        print(f"Final sample size: {samples.shape[0]}")

    return samples


def calc_vdisp_los_binned(
    R_proj: NDArray[np.floating],
    vlos: NDArray[np.floating],
    vlos_err: NDArray[np.floating],
    bins: Optional[NDArray[np.floating]] = None,
    ntracer_per_bin: int = 50,
    nbins_min: int = 4,
    nbins_max: int = 8,
    nsteps: int = 2000,
    max_steps: int = 10000,
    auto_extend: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Calculate observed velocity dispersion profile in radial bins.

    Parameters
    ----------
    R_proj : NDArray[np.floating]
        Projected radius of tracers in kpc.
    vlos : NDArray[np.floating]
        Line-of-sight velocities in km/s.
    vlos_err : NDArray[np.floating]
        Velocity errors in km/s.
    bins : NDArray[np.floating] | None, optional
        Bin edges in kpc. If None, uses equal-count binning.
    ntracer_per_bin : int, optional
        Target number of tracers per bin (for equal-count binning).
    nbins_min : int, optional
        Minimum number of bins.
    nbins_max : int, optional
        Maximum number of bins.
    nsteps : int, optional
        Number of MCMC steps for velocity dispersion fitting.
    max_steps : int, optional
        Maximum steps for MCMC fitting if auto-extending.
    auto_extend : bool, optional
        Whether to auto-extend MCMC chains for convergence.
    verbose : bool, optional
        Whether to print verbose output during fitting.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'R_mid': median radius of each bin
        - 'R_lo': lower edge of each bin
        - 'R_hi': upper edge of each bin
        - 'velsig': median velocity dispersion
        - 'velsig_lo': 16th percentile
        - 'velsig_hi': 84th percentile
    """
    if bins is None:
        # Equal-count binning
        num_tracers = len(R_proj)
        nbins = int(np.ceil(num_tracers / ntracer_per_bin))
        nbins = np.clip(nbins, nbins_min, nbins_max)

        sorted_R = np.sort(R_proj)
        bin_indices = np.array_split(np.arange(num_tracers), nbins)
        bins = np.array(
            [sorted_R[idx[0]] for idx in bin_indices] + [sorted_R[-1] * 1.001]
        )

    nbins = len(bins) - 1
    R_mid, R_lo, R_hi = [], [], []
    sigma, sigma_lo, sigma_hi = [], [], []

    for i in range(nbins):
        bin_mask = (R_proj >= bins[i]) & (R_proj < bins[i + 1])
        if np.sum(bin_mask) < 3:
            continue

        vr_bin = vlos[bin_mask]
        vr_err_bin = vlos_err[bin_mask]
        R_bin = R_proj[bin_mask]

        R_mid.append(np.median(R_bin))
        R_lo.append(R_bin.min())
        R_hi.append(R_bin.max())

        samples = fit_vdisp_los(vr_bin, vr_err_bin, nsteps=nsteps, verbose=verbose,
                                auto_extend=auto_extend, max_steps=max_steps)
        sigma_samples = np.exp(samples[..., 1])

        sigma.append(np.median(sigma_samples))
        sigma_lo.append(np.percentile(sigma_samples, 16))
        sigma_hi.append(np.percentile(sigma_samples, 84))

    R_mid, R_lo, R_hi = np.array(R_mid), np.array(R_lo), np.array(R_hi)
    sigma, sigma_lo, sigma_hi = np.array(sigma), np.array(sigma_lo), np.array(sigma_hi)

    R_em, R_ep = R_mid - R_lo, R_hi - R_mid
    sigma_em, sigma_ep = sigma - sigma_lo, sigma_hi - sigma

    return {
        'R_mid': R_mid,
        'R_em': R_em,
        'R_ep': R_ep,
        'sigma': sigma,
        'sigma_em': sigma_em,
        'sigma_ep': sigma_ep,
    }