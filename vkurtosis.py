"""Non-Gaussian velocity kurtosis models for dwarf galaxies."""

from typing import Tuple, Optional

import emcee
import pocomc
import multiprocessing as mp
import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.legendre import leggauss
from scipy.special import gamma as Gamma
from scipy.stats import uniform as sp_uniform

from gh_alternative.line_profiles import (
    ln_uniform_kernel_pdf,
    ln_laplace_kernel_pdf,
    uniform_kernel_variance_kurtosis,
    laplace_kernel_variance_kurtosis,
)


def log_prior_se(params: Tuple[float, float, float]) -> float:
    """Log-prior for Sanders & Evans (2020) PDF parameters (mu, log_sigma, h4).

    Priors:
    - mu: flat in (-100, 100) km/s
    - log_sigma: flat in (-5, 5)
    - h4: flat in valid range (-0.1877, 0.1454)
      (slightly inside limits to avoid numerical issues at boundaries)
    """
    mu, log_sigma, h4 = params
    if (-100.0 < mu < 100.0) and (-5.0 < log_sigma < 5.0) and (-0.187 < h4 < 0.145):
        return 0.0
    return -np.inf


def log_likelihood_se(
    params: Tuple[float, float, float],
    data: Tuple[NDArray[np.floating], NDArray[np.floating]],
) -> float:
    """Log-likelihood using Sanders & Evans (2020) PDF.

    Parameters
    ----------
    params : Tuple[float, float, float]
        (mu, log_sigma, h4) where h4 encodes the branch:
        h4 < 0 -> uniform kernel (platykurtic, kappa < 3)
        h4 = 0 -> Gaussian (kappa = 3)
        h4 > 0 -> Laplace kernel (leptokurtic, kappa > 3)
    data : Tuple
        (velocities, velocity_errors) in km/s.
    """
    mu, log_sigma, h4 = params
    vlos, vlos_err = data

    sigma = np.exp(log_sigma)
    h3 = 0.0  # assume symmetric

    # select branch based on sign of h4
    eps = 1e-4
    if h4 < -eps:
        ln_pdf = ln_uniform_kernel_pdf(vlos, vlos_err, mu, sigma, h3, h4)
    elif h4 > eps:
        ln_pdf = ln_laplace_kernel_pdf(vlos, vlos_err, mu, sigma, h3, h4)
    else:
        # Gaussian limit
        var = sigma**2 + vlos_err**2
        ln_pdf = -0.5 * np.log(2 * np.pi * var) - 0.5 * (vlos - mu)**2 / var

    if np.any(~np.isfinite(ln_pdf)):
        return -np.inf

    return np.sum(ln_pdf)


def log_posterior_se(
    params: Tuple[float, float, float],
    data: Tuple[NDArray[np.floating], NDArray[np.floating]],
) -> float:
    """Log-posterior for Sanders & Evans (2020) PDF."""
    lp = log_prior_se(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_se(params, data)


def log_prior_genGauss(params: Tuple[float, float, float]) -> float:
    """Log-prior for generalised Gaussian parameters (mu, log_sigma, log_bet)."""
    mu, log_alp, log_bet = params
    alp = np.exp(log_alp)
    bet = np.exp(log_bet)
    if -100.0 < mu < 100.0 and 0.1 < alp < 100.0 and 0.5 < bet < 4.0:
        return 0.0
    return -np.inf


def log_likelihood_genGauss(
    params: Tuple[float, float, float],
    data: Tuple[NDArray[np.floating], NDArray[np.floating]],
) -> float:
    """Log-likelihood for generalised Gaussian PDF with fast error approximation."""
    mu, log_alp, log_bet = params
    vlos, vlos_err = data

    alp = np.exp(log_alp)
    bet = np.exp(log_bet)

    # fold errors into effective width
    alp_eff = np.sqrt(alp**2 + vlos_err**2 * Gamma(1.0/bet) / Gamma(3.0/bet))
    pdf = bet / (2.0 * alp_eff * Gamma(1.0/bet)) * \
          np.exp(-(np.abs(vlos - mu) / alp_eff)**bet)

    if np.any(~np.isfinite(pdf)) or np.any(pdf <= 0):
        return -np.inf
    return np.sum(np.log(pdf))


def log_posterior_genGauss(
    params: Tuple[float, float, float],
    data: Tuple[NDArray[np.floating], NDArray[np.floating]],
) -> float:
    """Log-posterior for generalised Gaussian fitting."""
    lp = log_prior_genGauss(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_genGauss(params, data)


def log_likelihood_genGauss_full(
    params: Tuple[float, float, float],
    data: Tuple[NDArray[np.floating], NDArray[np.floating]],
    n_quad: int = 54,
) -> float:
    """Log-likelihood for generalised Gaussian PDF with full error convolution.

    Convolves the generalised Gaussian with per-star Gaussian errors using
    vectorised Gauss-Legendre quadrature (all stars evaluated in one batch).

    Parameters
    ----------
    params : Tuple[float, float, float]
        (mu, log_alp, log_bet).
    data : Tuple
        (velocities, velocity_errors) in km/s.
    n_quad : int
        Number of Gauss-Legendre quadrature points. Default 64 gives ~1e-8
        accuracy; reduce to 32 for speed or raise for very broad errors.
    """
    mu, log_alp, log_bet = params
    vlos, vlos_err = data

    alp = np.exp(log_alp)
    bet = np.exp(log_bet)
    sig = alp * np.sqrt(Gamma(3.0/bet) / Gamma(1.0/bet))

    vzlow = mu - 10.0 * sig
    vzhigh = mu + 10.0 * sig

    # Gauss-Legendre nodes/weights on [-1,1], mapped to [vzlow, vzhigh]
    xi, wi = leggauss(n_quad)
    half = 0.5 * (vzhigh - vzlow)
    vzint = half * xi + 0.5 * (vzlow + vzhigh)  # (n_quad,)

    # genGauss is the same for every star: (n_quad,)
    genGauss = (bet / (2.0 * alp * Gamma(1.0/bet))
                * np.exp(-(np.abs(vzint - mu) / alp)**bet))

    # per-star Gaussian error kernel: (n_stars, n_quad)
    dv = vlos[:, None] - vzint[None, :]
    gauss_err = (np.exp(-0.5 * (dv / vlos_err[:, None])**2)
                 / (np.sqrt(2.0 * np.pi) * vlos_err[:, None]))

    # integrate via weighted sum: (n_stars,)
    pdf = gauss_err @ (genGauss * wi * half)

    if np.any(pdf <= 0) or np.any(~np.isfinite(pdf)):
        return -np.inf

    return np.sum(np.log(pdf))


def log_posterior_genGauss_full(
    params: Tuple[float, float, float],
    data: Tuple[NDArray[np.floating], NDArray[np.floating]],
) -> float:
    """Log-posterior for full generalised Gaussian fitting."""
    lp = log_prior_genGauss(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_genGauss_full(params, data)


def _run_pocomc(
    log_likelihood_fn,
    prior,
    data,
    n_walkers: int,
    verbose: bool=True,
) -> NDArray[np.floating]:
    """Run pocoMC preconditioned Monte Carlo sampler.

    Returns posterior samples of shape (n_samples, n_dim).
    """
    with mp.Pool(n_walkers) as pool:
        sampler = pocomc.Sampler(
            prior=prior,
            likelihood=log_likelihood_fn,
            likelihood_kwargs=dict(data=data),
            pool=pool,
            vectorize=False,
        )
        sampler.run(n_total=500, progress=verbose)
        samples, logl, logp = sampler.posterior(resample=True)
    return samples


def _run_mcmc(
    sampler: emcee.EnsembleSampler,
    p0: NDArray[np.floating],
    nsteps: int,
    auto_extend: bool,
    max_steps: int,
    convergence_factor: float,
    verbose: bool,
) -> NDArray[np.floating]:
    """Run MCMC with optional auto-extension until convergence.

    Returns flat samples after burn-in and thinning.
    """
    sampler.run_mcmc(p0, nsteps, progress=verbose)

    total_steps = nsteps
    converged = False
    tau = None

    while not converged:
        try:
            tau = sampler.get_autocorr_time()
            converged = np.all(total_steps > convergence_factor * tau)

            if not converged and auto_extend:
                if total_steps >= max_steps:
                    if verbose:
                        print(
                            f"Warning: Reached max_steps={max_steps} without full convergence. "
                            f"Current tau={tau}, need {convergence_factor}*tau steps."
                        )
                    break

                try:
                    steps_needed = int(convergence_factor * np.nanmax(tau)) - total_steps
                except:
                    print(tau)
                extend_steps = min(steps_needed, max_steps - total_steps)
                extend_steps = max(extend_steps, nsteps)

                if verbose:
                    print(
                        f"Chain not converged (n={total_steps}, tau={np.nanmax(tau):.1f}). "
                        f"Extending by {extend_steps} steps..."
                    )
                sampler.run_mcmc(None, extend_steps, progress=verbose)
                total_steps += extend_steps
            elif not converged:
                if verbose:
                    print(
                        f"Warning: Chain may not be converged. "
                        f"n_steps={total_steps}, tau={tau}. Consider increasing nsteps."
                    )
                break

        except emcee.autocorr.AutocorrError as e:
            if auto_extend and total_steps < max_steps:
                extend_steps = min(nsteps, max_steps - total_steps)
                if verbose:
                    print(
                        f"Autocorrelation time estimation failed: {e}. "
                        f"Extending chain by {extend_steps} steps..."
                    )
                sampler.run_mcmc(None, extend_steps, progress=verbose)
                total_steps += extend_steps
            else:
                if verbose:
                    print(f"Warning: Could not estimate autocorrelation time: {e}")
                tau = sampler.get_autocorr_time(quiet=True)
                break

    tau_max = np.nanmax(tau) if np.any(np.isfinite(tau)) else total_steps // 4
    tau_min = np.nanmin(tau) if np.any(np.isfinite(tau)) else total_steps // 4
    burnin = int(3 * tau_max)
    thin = max(1, int(0.5 * tau_min))
    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

    if verbose:
        print(f"Autocorrelation times: {tau}")
        print(f"Mean tau: {np.nanmean(tau):.1f} steps (total chain: {total_steps} steps)")
        print(f"Discarded {burnin} steps, thinned by {thin}")
        print(f"Final sample size: {samples.shape[0]}")

    return samples


def fit_kurtosis_los(
    vr: NDArray[np.floating],
    vr_err: NDArray[np.floating],
    method: str = 'fast',
    sampler: str = 'emcee',
    nwalkers: int = 12,
    nsteps: int = 1000,
    auto_extend: bool = True,
    max_steps: int = 10000,
    convergence_factor: float = 50.0,
    verbose: bool = True,
) -> NDArray[np.floating]:
    """Fit velocity dispersion and kurtosis using MCMC.

    Parameters
    ----------
    vr : NDArray[np.floating]
        Line-of-sight velocities in km/s.
    vr_err : NDArray[np.floating]
        Velocity uncertainties in km/s.
    method : str, optional
        PDF model: 'fast' (generalised Gaussian, approximate error folding),
        'full' (generalised Gaussian, exact error convolution via quad),
        or 'sanders_evans' (Sanders & Evans 2020 non-Gaussian PDF).
        Default is 'fast'.
    sampler : str, optional
        Sampling backend: 'emcee' (ensemble MCMC) or 'pocomc' (preconditioned
        sequential Monte Carlo). Default is 'emcee'.
    nwalkers : int, optional
        Number of walkers (emcee) or particles (pocomc), by default 12.
    nsteps : int, optional
        Number of MCMC steps per walker (emcee only), by default 1000.
    auto_extend : bool, optional
        Extend the chain if autocorrelation time indicates insufficient samples
        (emcee only).
    max_steps : int, optional
        Maximum total steps when auto-extending (emcee only), by default 10000.
    convergence_factor : float, optional
        Chain is considered converged when n_steps > convergence_factor * tau
        (emcee only), by default 50.0.
    verbose : bool, optional
        Print convergence diagnostics.

    Returns
    -------
    NDArray[np.floating]
        Flattened posterior samples of shape (n_samples, 3) with columns
        [mu, sigma, kappa].
    """
    ndim = 3
    data = (vr, vr_err)

    if method == 'sanders_evans':
        log_like_fn = log_likelihood_se
        posterior = log_posterior_se
        p0 = (np.random.rand(nwalkers, ndim) * np.array([200.0, 10.0, 0.332])
              - np.array([100.0, 5.0, -0.187]))
    elif method in ('fast', 'full'):
        log_like_fn = log_likelihood_genGauss if method == 'fast' else log_likelihood_genGauss_full
        posterior = log_posterior_genGauss if method == 'fast' else log_posterior_genGauss_full
        mu0 = np.median(vr)
        alp0 = np.std(vr)
        bet0 = 2.0
        p0 = (np.array([mu0, np.log(alp0), np.log(bet0)])
              + 1e-3 * np.random.randn(nwalkers, ndim))
    else:
        raise ValueError(f"method={method!r} must be 'fast', 'full', or 'sanders_evans'")

    if sampler == 'emcee':
        emcee_sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, args=[data])
        raw = _run_mcmc(emcee_sampler, p0, nsteps, auto_extend, max_steps, convergence_factor, verbose)
    elif sampler == 'pocomc':

        # pocomc requires different priors
        if method == 'sanders_evans':
            dists = [
                sp_uniform(-100.0, 200.0),
                sp_uniform(-5.0, 10.0),
                sp_uniform(-0.187, 0.332),
            ]
        else:
            log_alp_lo = np.log(0.1)
            dists = [
                sp_uniform(-100.0, 200.0),
                sp_uniform(log_alp_lo, np.log(100.0) - log_alp_lo),
                sp_uniform(np.log(0.5), np.log(4.0) - np.log(0.5)),
            ]
        prior = pocomc.Prior(dists)
        raw = _run_pocomc(log_like_fn, prior, data, nwalkers, verbose)
    else:
        raise ValueError(f"sampler={sampler!r} must be 'emcee' or 'pocomc'")

    if method == 'sanders_evans':
        sigma_s = np.exp(raw[:, 1])
        h4_s = raw[:, 2]

        kappa_s = np.zeros(len(h4_s))
        sigma2_s = np.zeros(len(h4_s))
        mask_neg = h4_s < 0
        mask_pos = h4_s > 0
        vn, kn = uniform_kernel_variance_kurtosis(sigma_s[mask_neg], 0.0, h4_s[mask_neg])
        vp, kp = laplace_kernel_variance_kurtosis(sigma_s[mask_pos], 0.0, h4_s[mask_pos])
        kappa_s[mask_neg] = kn + 3.0  # convert excess kurtosis to kurtosis
        sigma2_s[mask_neg] = vn
        kappa_s[mask_pos] = kp + 3.0
        sigma2_s[mask_pos] = vp
        kappa_s[~(mask_neg | mask_pos)] = 3.0
        sigma2_s[~(mask_neg | mask_pos)] = sigma_s[~(mask_neg | mask_pos)]**2
    else:
        alp_s = np.exp(raw[:, 1])
        bet_s = np.exp(raw[:, 2])
        sigma_s = alp_s * np.sqrt(Gamma(3.0/bet_s) / Gamma(1.0/bet_s))
        kappa_s = Gamma(5.0/bet_s) * Gamma(1.0/bet_s) / Gamma(3.0/bet_s)**2

    return np.column_stack([raw[:, 0], sigma_s, kappa_s])


def calc_kurtosis_los_binned(
    R_proj: NDArray[np.floating],
    vlos: NDArray[np.floating],
    vlos_err: NDArray[np.floating],
    method: str = 'fast',
    sampler: str = 'emcee',
    bins: Optional[NDArray[np.floating]] = None,
    ntracer_per_bin: int = 50,
    nbins_min: int = 4,
    nbins_max: int = 8,
    verbose: bool = True,
    sampler_args: Optional[dict] = None,
) -> dict:
    """Calculate observed kurtosis profile in radial bins.

    Parameters
    ----------
    R_proj : NDArray[np.floating]
        Projected radius of tracers in kpc.
    vlos : NDArray[np.floating]
        Line-of-sight velocities in km/s.
    vlos_err : NDArray[np.floating]
        Velocity errors in km/s.
    method : str, optional
        PDF model passed to :func:`fit_kurtosis_los`. Default is 'fast'.
    sampler : str, optional
        Sampling backend passed to :func:`fit_kurtosis_los`: 'emcee' or
        'pocomc'. Default is 'emcee'.
    bins : NDArray[np.floating] | None, optional
        Bin edges in kpc. If None, uses equal-count binning.
    ntracer_per_bin : int, optional
        Target number of tracers per bin (for equal-count binning).
    nbins_min : int, optional
        Minimum number of bins.
    nbins_max : int, optional
        Maximum number of bins.
    verbose : bool, optional
        Whether to print verbose output during fitting.
    sampler_args : dict | None, optional
        Extra keyword arguments forwarded to :func:`fit_kurtosis_los`, e.g.
        ``nwalkers``, ``nsteps``, ``max_steps``, ``auto_extend``,
        ``convergence_factor``. Default is ``None`` (use function defaults).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'R_mid': median radius of each bin
        - 'R_em': lower radius error
        - 'R_ep': upper radius error
        - 'sigma': median velocity dispersion
        - 'sigma_em': 16th percentile error
        - 'sigma_ep': 84th percentile error
        - 'kappa': median kurtosis
        - 'kappa_em': 16th percentile error
        - 'kappa_ep': 84th percentile error
    """
    if bins is None:
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
    kappa, kappa_lo, kappa_hi = [], [], []

    for i in range(nbins):
        bin_mask = (R_proj >= bins[i]) & (R_proj < bins[i + 1])
        if np.sum(bin_mask) < 3:
            continue

        vr_bin = vlos[bin_mask]
        err_bin = vlos_err[bin_mask]
        R_bin = R_proj[bin_mask]

        R_lo.append(R_bin.min())
        R_hi.append(R_bin.max())
        R_mid.append(0.5 * (R_lo[-1] + R_hi[-1]))

        samples = fit_kurtosis_los(
            vr_bin, err_bin, method=method, sampler=sampler,
            verbose=verbose, **(sampler_args or {}),
        )
        # columns: [mu, sigma, kappa]
        sigma_samples = samples[:, 1]
        kappa_samples = samples[:, 2]

        sigma.append(np.median(sigma_samples))
        sigma_lo.append(np.percentile(sigma_samples, 16))
        sigma_hi.append(np.percentile(sigma_samples, 84))

        kappa.append(np.median(kappa_samples))
        kappa_lo.append(np.percentile(kappa_samples, 16))
        kappa_hi.append(np.percentile(kappa_samples, 84))

    R_mid = np.array(R_mid)
    R_lo = np.array(R_lo)
    R_hi = np.array(R_hi)
    sigma = np.array(sigma)
    sigma_lo = np.array(sigma_lo)
    sigma_hi = np.array(sigma_hi)
    kappa = np.array(kappa)
    kappa_lo = np.array(kappa_lo)
    kappa_hi = np.array(kappa_hi)

    return {
        'R_mid': R_mid,
        'R_em': R_mid - R_lo,
        'R_ep': R_hi - R_mid,
        'sigma': sigma,
        'sigma_em': sigma - sigma_lo,
        'sigma_ep': sigma_hi - sigma,
        'kappa': kappa,
        'kappa_em': kappa - kappa_lo,
        'kappa_ep': kappa_hi - kappa,
    }
