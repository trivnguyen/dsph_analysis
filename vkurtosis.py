"""Non-Gaussian velocity kurtosis models for dwarf galaxies."""

from typing import Tuple, Optional

import emcee
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erf, erfc, erfcx
from scipy.special import gamma as Gamma


# non-gaussian velocity PDF from Sanders & Evans (2020)
def _kappa_to_a(kappa: float) -> Tuple[float, str]:
    """Solve for parameter a given target kurtosis kappa.

    Returns (a, branch) where branch is 'uniform' or 'laplace'.
    Uses sign convention: a >= 0 always, branch encodes the side.

    Valid range: 1.8 < kappa < 6.
    """
    if not (1.8 < kappa < 6.0):
        raise ValueError(f"kappa={kappa} outside valid range (1.8, 6.0)")

    if kappa <= 3.0:
        # uniform kernel: kappa = 3 - 2a^4/15 / (1 + a^2/3)^2
        def eq(a):
            return 3 - 2*a**4/15 / (1 + a**2/3)**2 - kappa
        a = brentq(eq, 0, 1000)
        return a, 'uniform'
    else:
        # laplace kernel: kappa = 3 + 12a^4 / (2a^2 + 1)^2
        def eq(a):
            return 3 + 12*a**4 / (2*a**2 + 1)**2 - kappa
        a = brentq(eq, 0, 1000)
        return a, 'laplace'


def _a_to_kappa(a: float) -> float:
    """Convert signed parameter a to kurtosis kappa.

    Sign convention:
    - a < 0 : uniform kernel, kappa < 3
    - a = 0 : Gaussian, kappa = 3
    - a > 0 : Laplace kernel, kappa > 3
    """
    abs_a = abs(a)
    return np.where(
        a < 0, 3 - 2 * abs_a**4/15 / (1 + abs_a**2/3)**2,
        3 + 12 * abs_a**4 / (2 * abs_a**2 + 1)**2
    )


def _Phi(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Standard normal CDF."""
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def _sanders_evans_pdf(
    w: NDArray[np.floating],
    s: NDArray[np.floating],
    kappa: float,
) -> NDArray[np.floating]:
    """Sanders & Evans (2020) non-Gaussian velocity PDF.

    Parameters
    ----------
    w : NDArray
        Scaled velocity (v - mu) / sigma.
    s : NDArray
        Scaled error delta_v / sigma.
    kappa : float
        Target kurtosis (1.8 < kappa < 6).

    Returns
    -------
    NDArray
        PDF values f(w).
    """
    a, branch = _kappa_to_a(kappa)

    # Gaussian limit: as a->0, both branches reduce to N(0, sqrt(1+s^2))
    if a < 1e-6:
        t = np.sqrt(1.0 + s**2)
        return np.exp(-0.5 * (w / t)**2) / (np.sqrt(2 * np.pi) * t)

    if branch == 'uniform':
        b2 = 1 + a**2 / 3
        b = np.sqrt(b2)
        t2 = 1 + b**2 * s**2
        t = np.sqrt(t2)
        # Eq. A.1
        pdf = b / (2*a) * (
            _Phi((b*w + a) / t) - _Phi((b*w - a) / t)
        )
    else:  # laplace
        b2 = 2*a**2 + 1
        b = np.sqrt(b2)
        t2 = 1 + b**2 * s**2
        t = np.sqrt(t2)
        # Eq. A.6, numerically stable form.
        # Identity: exp(A)*erfc(u) = exp(-(b*w/t)^2/2)*erfcx(u)  for u >= 0.
        # For u < 0: erfc(u) = 2 - erfc(-u), giving
        #   exp(A)*erfc(u) = 2*exp(A) - exp(-(b*w/t)^2/2)*erfcx(-u).
        # When u < 0 the corresponding A < 0, so exp(A) is always bounded.
        # A is clamped only to suppress overflow warnings in the discarded
        # np.where branch (the correct branch never sees the clamped value).
        bw = b * w
        common = np.exp(-0.5 * (bw / t)**2)
        u1 = (t2 - a * bw) / (np.sqrt(2) * t * a)
        u2 = (t2 + a * bw) / (np.sqrt(2) * t * a)
        A1 = np.minimum((t2 - 2*a*bw) / (2*a**2), 500.0)
        A2 = np.minimum((t2 + 2*a*bw) / (2*a**2), 500.0)
        T1 = np.where(u1 >= 0,
                      common * erfcx(np.maximum(u1, 0.0)),
                      2*np.exp(A1) - common * erfcx(np.maximum(-u1, 0.0)))
        T2 = np.where(u2 >= 0,
                      common * erfcx(np.maximum(u2, 0.0)),
                      2*np.exp(A2) - common * erfcx(np.maximum(-u2, 0.0)))
        pdf = b / (4*a) * (T1 + T2)

    return pdf


def log_prior_kappa(params: Tuple[float, float, float]) -> float:
    """Log-prior for (mu, log_sigma, a) parameterization.

    Priors:
    - mu: flat in (-100, 100) km/s
    - log_sigma: flat in (-5, 5)
    - a: flat in (-5, 5), sign encodes branch (negative=uniform, positive=laplace)
    """
    mu, log_sigma, a = params
    if -100.0 < mu < 100.0 and -5.0 < log_sigma < 5.0 and -5.0 < a < 5.0:
        return 0.0
    return -np.inf


def log_likelihood_kappa(
    params: Tuple[float, float, float],
    data: Tuple[NDArray[np.floating], NDArray[np.floating]],
) -> float:
    """Log-likelihood using Sanders & Evans (2020) PDF.

    Parameters
    ----------
    params : Tuple[float, float, float]
        (mu, log_sigma, a) where sign of a encodes PDF branch.
    data : Tuple
        (velocities, velocity_errors) in km/s.
    """
    mu, log_sigma, a = params
    vlos, vlos_err = data

    sigma = np.exp(log_sigma)
    kappa = _a_to_kappa(a)

    # guard against kappa outside valid range
    if not (1.8 < kappa < 6.0):
        return -np.inf

    # scaled variables
    w = (vlos - mu) / sigma
    s = vlos_err / sigma

    pdf = _sanders_evans_pdf(w, s, kappa)

    # guard against numerical zeros or non-finite values
    if np.any(~np.isfinite(pdf)) or np.any(pdf <= 0):
        return -np.inf

    # jacobian: f(v) = f(w) / sigma
    return np.sum(np.log(pdf / sigma))


def log_posterior_kappa(
    params: Tuple[float, float, float],
    data: Tuple[NDArray[np.floating], NDArray[np.floating]],
) -> float:
    """Log-posterior for kurtosis fitting."""
    lp = log_prior_kappa(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_kappa(params, data)


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
) -> float:
    """Log-likelihood for generalised Gaussian PDF with full error convolution.

    Convolves the generalised Gaussian with a per-star Gaussian error via quad.

    Parameters
    ----------
    params : Tuple[float, float, float]
        (mu, log_alp, log_bet).
    data : Tuple
        (velocities, velocity_errors) in km/s.
    """
    mu, log_alp, log_bet = params
    vlos, vlos_err = data

    alp = np.exp(log_alp)
    bet = np.exp(log_bet)
    sig = alp * np.sqrt(Gamma(3.0/bet) / Gamma(1.0/bet))

    # integration range: +/- 10 sigma around mean
    vzlow = mu - 10.0 * sig
    vzhigh = mu + 10.0 * sig

    def _integrand(vzint, v_i, err_i):
        """Generalised Gaussian * Gaussian error kernel."""
        genGauss = bet / (2.0 * alp * Gamma(1.0/bet)) * \
                   np.exp(-(np.abs(vzint - mu) / alp)**bet)
        gauss_err = np.exp(-0.5 * ((v_i - vzint) / err_i)**2) / \
                    (np.sqrt(2.0 * np.pi) * err_i)
        return genGauss * gauss_err

    log_like = 0.0
    for v_i, err_i in zip(vlos, vlos_err):
        pdf_i, _ = quad(_integrand, vzlow, vzhigh, args=(v_i, err_i),
                        epsabs=1e-6, epsrel=1e-6)
        if pdf_i <= 0 or not np.isfinite(pdf_i):
            return -np.inf
        log_like += np.log(pdf_i)

    return log_like


def log_posterior_genGauss_full(
    params: Tuple[float, float, float],
    data: Tuple[NDArray[np.floating], NDArray[np.floating]],
) -> float:
    """Log-posterior for full generalised Gaussian fitting."""
    lp = log_prior_genGauss(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_genGauss_full(params, data)


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

                steps_needed = int(convergence_factor * np.nanmax(tau)) - total_steps
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
    nwalkers : int, optional
        Number of MCMC walkers, by default 12.
    nsteps : int, optional
        Number of MCMC steps per walker, by default 1000.
    auto_extend : bool, optional
        Extend the chain if autocorrelation time indicates insufficient samples.
    max_steps : int, optional
        Maximum total steps when auto-extending, by default 10000.
    convergence_factor : float, optional
        Chain is considered converged when n_steps > convergence_factor * tau,
        by default 50.0.
    verbose : bool, optional
        Print convergence diagnostics.

    Returns
    -------
    NDArray[np.floating]
        Flattened MCMC samples of shape (n_samples, 3) with columns
        [mu, sigma, kappa].
    """
    ndim = 3

    if method == 'sanders_evans':
        posterior = log_posterior_kappa
        p0 = (np.random.rand(nwalkers, ndim) * np.array([200.0, 10.0, 10.0])
              - np.array([100.0, 5.0, 5.0]))
    elif method in ('fast', 'full'):
        posterior = log_posterior_genGauss if method == 'fast' else log_posterior_genGauss_full
        mu0 = np.median(vr)
        alp0 = np.std(vr)
        bet0 = 2.0
        p0 = (np.array([mu0, np.log(alp0), np.log(bet0)])
              + 1e-3 * np.random.randn(nwalkers, ndim))
    else:
        raise ValueError(f"method={method!r} must be 'fast', 'full', or 'sanders_evans'")

    sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, args=[(vr, vr_err)])
    raw = _run_mcmc(sampler, p0, nsteps, auto_extend, max_steps, convergence_factor, verbose)

    if method == 'sanders_evans':
        sigma_s = np.exp(raw[:, 1])
        kappa_s = _a_to_kappa(raw[:, 2])
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
    bins: Optional[NDArray[np.floating]] = None,
    ntracer_per_bin: int = 50,
    nbins_min: int = 4,
    nbins_max: int = 8,
    nsteps: int = 2000,
    max_steps: int = 10000,
    auto_extend: bool = True,
    verbose: bool = True,
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
    bins : NDArray[np.floating] | None, optional
        Bin edges in kpc. If None, uses equal-count binning.
    ntracer_per_bin : int, optional
        Target number of tracers per bin (for equal-count binning).
    nbins_min : int, optional
        Minimum number of bins.
    nbins_max : int, optional
        Maximum number of bins.
    nsteps : int, optional
        Number of MCMC steps for fitting.
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
            vr_bin, err_bin, method=method,
            nsteps=nsteps, auto_extend=auto_extend,
            max_steps=max_steps, verbose=verbose,
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
