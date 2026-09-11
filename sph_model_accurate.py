"""
Accurate, vectorised moments for GeneralizedOMJeans.

``GeneralizedOMJeans`` evaluates every one of its integrals with
``scipy.quad`` at ``epsabs = epsrel = 1``, which accepts the first
Gauss-Kronrod estimate. Two of them are not accurate enough for mock
generation or for a likelihood:

* the projection of sigma_r^2 under-estimates sigma_los^2 by 2-5%,
  radius dependent (jeans_calibration audit, 2026-09-08, checked against
  a tight re-integration and against agama's DF moments). The
  substitution r = R cosh(u) removes the endpoint singularity of
  r / sqrt(r^2 - R^2) and leaves a smooth integrand that a fixed
  Gauss-Legendre rule handles to < 1e-4, vectorised over all projected
  radii at once;
* the radial integral for sigma_r^2 runs to infinity and misses the
  integrand's peak whenever r_star is small in kpc: over the sbi_twins
  Prior A, 37 of 100 draws were wrong by more than 10% and many returned
  ~0 (checked against a tight quad, 2026-09-08). Here it is a cumulative
  trapezoid in log r on a dense grid, with a power-law tail beyond the
  grid.

The fourth-order moments (Richardson & Fairbairn 2013; Appendix C of
Nguyen et al. 2026) reuse the same two schemes, with the closure
beta' = beta that holds exactly for a Cuddeford-Osipkov-Merritt DF.

``GeneralizedOMJeans`` itself is deliberately left untouched: the same
tolerance sits in its proper-motion and fourth-moment projections, and
changing it there moves the results of every project that imports the
library. ``AccurateOMJeans`` is a drop-in subclass, so adopting it stays
a per-project choice.

Example:
    >>> from dsph_analysis.sph_model_accurate import build_jeans
    >>> model = build_jeans(theta)          # theta[8] is r_star [kpc]
    >>> sigma_los = np.sqrt(model.sigma2_los(R))
    >>> kappa_los = model.kurtosis_los(R)
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import constants

from .sph_model import _TO_KM2_S2, GeneralizedOMJeans

# Radial grid for sigma_r^2, in units of r_star. Stars are kept inside
# 10 r_star and the Plummer density falls as r^-5, so 1e3 r_star is far
# enough for the projection integral.
GRID_MIN_RSTAR = 1e-3
GRID_MAX_RSTAR = 1e3
N_GRID = 300
# Reason: the radial integrands are evaluated on their own dense log
# grid, one decade past the sigma_r^2 grid, so the trapezoid error stays
# below 1e-5 and the tail correction is small.
N_INTEGRAND = 4000
TAIL_DECADES = 1.0


def _inward_integral(s: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    int_{s_j}^inf f ds for every point of a log-spaced grid.

    Trapezoid in ln s from the outer edge inward, plus a power-law tail
    fitted to the last two points beyond the grid.

    Args:
        s: Increasing, log-spaced radii.
        f: Integrand evaluated at s.

    Returns:
        Array of the same length as s.
    """
    y = f * s
    segments = 0.5 * (y[1:] + y[:-1]) * np.diff(np.log(s))
    inward = np.concatenate([np.cumsum(segments[::-1])[::-1], [0.0]])
    slope = np.log(f[-1] / f[-2]) / np.log(s[-1] / s[-2])
    tail = -f[-1] * s[-1] / (slope + 1) if slope < -1 else 0.0
    return inward + tail


class AccurateOMJeans(GeneralizedOMJeans):
    """GeneralizedOMJeans with accurate, vectorised 2nd and 4th moments."""

    N_NODES = 160
    CHUNK = 20000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_sigma2_r_grid = None
        self._log_sigma4_r_grid = None

    def _integrand_radii(self) -> np.ndarray:
        """Dense log grid for the radial integrals."""
        return np.logspace(
            np.log10(self.r_grid[0]),
            np.log10(self.r_grid[-1]) + TAIL_DECADES, N_INTEGRAND)

    def _sigma2_r_loglog(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Cache log sigma_r^2 on the dense radial grid.

        sigma_r^2(r) = 1 / (nu g) int_r^inf nu g G M / s^2 ds with g the
        integrating factor of beta(r).
        """
        if self._log_sigma2_r_grid is None:
            s = self._integrand_radii()
            weight = self.nu(s) * self.gbeta(s)
            f = constants.G * self.M(s) / s ** 2 * weight
            sigma2 = _inward_integral(s, f) / weight * _TO_KM2_S2
            self._log_sigma2_r_grid = (
                np.log(s), np.log(np.maximum(sigma2, 1e-300)))
        return self._log_sigma2_r_grid

    def _sigma4_r_loglog(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Cache log <v_r^4> on the dense radial grid.

        <v_r^4>(r) = 3 / (nu g') int_r^inf nu g' G M sigma_r^2 / s^2 ds
        (Eq. C6 of Nguyen et al. 2026) with g' the integrating factor of
        beta'; the library sets beta' = beta.
        """
        if self._log_sigma4_r_grid is None:
            log_s, log_sigma2 = self._sigma2_r_loglog()
            s = np.exp(log_s)
            weight = self.nu(s) * self.gbeta_prime(s)
            f = constants.G * self.M(s) / s ** 2 * weight * np.exp(log_sigma2)
            v4 = 3.0 * _inward_integral(s, f) / weight * _TO_KM2_S2
            self._log_sigma4_r_grid = (log_s, np.log(np.maximum(v4, 1e-300)))
        return self._log_sigma4_r_grid

    def sigma2_r_vec(self, r: np.ndarray) -> np.ndarray:
        """sigma_r^2 [km^2/s^2] at arbitrary radii, log-log interpolated."""
        log_r, log_s2 = self._sigma2_r_loglog()
        return np.exp(np.interp(np.log(r), log_r, log_s2))

    def sigma4_r_vec(self, r: np.ndarray) -> np.ndarray:
        """<v_r^4> [km^4/s^4] at arbitrary radii, log-log interpolated."""
        log_r, log_v4 = self._sigma4_r_loglog()
        return np.exp(np.interp(np.log(r), log_r, log_v4))

    def _integrand_sigma2(self, r: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Projection integrand of sigma_los^2 without the Abel kernel."""
        return ((1.0 - self.beta(r) * (R / r) ** 2) * self.nu(r)
                * self.sigma2_r_vec(r))

    def _integrand_sigma4(self, r: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Projection integrand of <v_los^4> without the Abel kernel.

        Uses the library's F_los(r, R) (Eq. C8), which drops the
        (beta' - beta) term; that term vanishes for beta' = beta.
        """
        return self._F_los(r, R) * self.nu(r) * self.sigma4_r_vec(r)

    def _project(self, R: np.ndarray, integrand) -> np.ndarray:
        """
        2 / I(R) int_R^inf integrand(r, R) r / sqrt(r^2 - R^2) dr.

        Args:
            R: Projected radii [kpc].
            integrand: Callable of (r, R) broadcasting over both.

        Returns:
            Projected moment at each R.
        """
        R = np.atleast_1d(np.asarray(R, dtype=float))
        # Reason: the (n_radii, N_NODES) work arrays reach ~1 GB for the
        # 1e5-star samples used in the notebooks, so evaluate in chunks.
        out = np.empty_like(R)
        for start in range(0, len(R), self.CHUNK):
            sl = slice(start, start + self.CHUNK)
            out[sl] = self._project_chunk(R[sl], integrand)
        return out

    def _project_chunk(self, R: np.ndarray, integrand) -> np.ndarray:
        """Projection integral for one chunk of radii, r = R cosh(u)."""
        x, w = leggauss(self.N_NODES)
        umax = np.arccosh(self.r_grid[-1] / R)
        u = 0.5 * umax[:, None] * (x[None, :] + 1.0)
        wu = 0.5 * umax[:, None] * w[None, :]
        cosh_u = np.cosh(u)
        r = R[:, None] * cosh_u
        f = integrand(r, R[:, None]) * R[:, None] * cosh_u
        return 2.0 / self.I(R) * np.sum(wu * f, axis=1)

    def sigma2_los(self, R: np.ndarray) -> np.ndarray:
        """Line-of-sight velocity dispersion squared [km^2/s^2] at R [kpc]."""
        return self._project(R, self._integrand_sigma2)

    def sigma4_los(self, R: np.ndarray) -> np.ndarray:
        """Fourth line-of-sight velocity moment [km^4/s^4] at R [kpc]."""
        return self._project(R, self._integrand_sigma4)

    def kurtosis_los(self, R: np.ndarray) -> np.ndarray:
        """Line-of-sight kurtosis <v_los^4> / sigma_los^4 at R [kpc]."""
        return self.sigma4_los(R) / self.sigma2_los(R) ** 2


def build_jeans(theta, n_grid: int = N_GRID) -> AccurateOMJeans:
    """
    Construct the model with the radial grid scaled to r_star.

    The default grid of GeneralizedOMJeans is fixed in kpc, so it misses
    the integrands whenever r_star is far from 0.1 kpc. Here the bounds
    are set from theta[8], the Plummer scale radius.

    Args:
        theta: The 12-element GeneralizedOMJeans parameter vector
            [log_rho_s, log_r_s, alp, bet, gam, log_r_a, two_to_beta0,
            two_to_betainf, rh, vsys_los, vsys_pmR, vsys_pmT], radii in
            kpc.
        n_grid: Number of log-spaced radii for the sigma_r^2 grid.

    Returns:
        AccurateOMJeans instance with the grid scaled to r_star.
    """
    r_star = float(np.asarray(theta)[8])
    return AccurateOMJeans(
        theta, min_rgrid=GRID_MIN_RSTAR * r_star,
        max_rgrid=GRID_MAX_RSTAR * r_star, n_grid=n_grid)
