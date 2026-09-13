
import numpy as np
import scipy.special as sc
from numpy.polynomial.legendre import leggauss
from scipy import constants
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import brentq

import astropy.units as auni
import astropy.cosmology as acosm

_TO_KM2_S2 = 1.989e12 / 3.0856  # G unit conversion: (10^7 M_sun, kpc) -> km^2/s^2

# Radial grid for sigma_r^2, in units of r_star, used by build_jeans. Stars
# are kept inside 10 r_star and the Plummer density falls as r^-5, so
# 1e3 r_star is far enough for the projection integral.
GRID_MIN_RSTAR = 1e-3
GRID_MAX_RSTAR = 1e3
N_GRID = 300
# Reason: the radial integrands are evaluated on their own dense log grid,
# one decade past the sigma_r^2 grid, so the trapezoid error stays below
# 1e-5 and the tail correction is small.
N_INTEGRAND = 4000
TAIL_DECADES = 1.0


def _inward_integral(s, f):
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


class GeneralizedOMJeans:
    """
    A class to model the velocity dispersion profile using the Jeans equation
    with a generalized Osipkov-Merritt anisotropy profile.

    The generalized OM profile allows arbitrary central (beta_0) and outer
    (beta_inf) anisotropy, with a transition controlled by the anisotropy
    radius r_a.

    Model parameters:
    - log_rho_s : Logarithm of the characteristic density of the dark matter halo
    - log_r_s   : Logarithm of the scale radius of the dark matter halo
    - alp       : Transition sharpness of the generalized NFW profile
    - bet       : Outer slope of the generalized NFW profile
    - gam       : Inner slope of the generalized NFW profile
    - log_r_a   : Logarithm of the anisotropy radius for the OM
    - two_to_beta0 : 2 raised to the power of the central anisotropy
    - two_to_betainf : 2 raised to the power of the outer anisotropy
    - rh        : Plummer scale radius for the stellar distribution
    - vsys_los   : Systematic velocity offset for line-of-sight velocities
    - vsys_pmR   : Systematic velocity offset for radial proper motions
    - vsys_pmT   : Systematic velocity offset for tangential proper motions

    """
    # Reason: these were epsabs=1, epsrel=1 -- the same loose setting as
    # the old fast path, so the "high-precision" variants were not
    # actually precise and could not serve as an independent check.
    _QUAD_EXACT_KW = dict(epsabs=0, epsrel=1e-10, limit=400)

    def __init__(self, theta, min_rgrid=1e-3, max_rgrid=50, n_grid=500):
        """
        Args:
            theta (list): Model parameters [log_rho_s, log_r_s, alp, bet, gam,
                log_r_a, two_to_beta0, two_to_betainf, rh,
                vsys_los, vsys_pmR, vsys_pmT].
        """
        self.param = theta
        self.log_rho_s = self.param[0]
        self.log_r_s = self.param[1]
        self.alp = self.param[2]
        self.bet = self.param[3]
        self.gam = self.param[4]
        self.log_r_a = self.param[5]
        self.two_to_beta0 = self.param[6]
        self.two_to_betainf = self.param[7]
        self.rh = self.param[8]
        self.vsys_los = self.param[9]
        self.vsys_pmR = self.param[10]
        self.vsys_pmT = self.param[11]

        # derived parameters
        self.beta0 = np.log2(self.two_to_beta0)
        self.betainf = np.log2(self.two_to_betainf)
        self.rho_s = 10.0 ** self.log_rho_s
        self.r_s = 10.0 ** self.log_r_s
        self.r_a = 10.0 ** self.log_r_a

        # interpolation grid for sigma_r^2
        self.min_rgrid = min_rgrid
        self.max_rgrid = max_rgrid
        self.n_grid = n_grid
        self.r_grid = np.logspace(np.log10(self.min_rgrid), np.log10(self.max_rgrid), self.n_grid)
        self._log_sigma2_r_grid = None
        self._log_sigma4_r_grid = None

    def rho(self, r):
        """Dark matter density profile (generalized NFW)."""
        c1 = self.rho_s * (r / self.r_s)**(-self.gam)
        c2 = (1 + (r / self.r_s)**(self.alp))**(-(self.bet - self.gam)/self.alp)
        return c1 * c2

    def rho_log_slope(self, r):
        """ Logarithmic slope of the dark matter density profile at radius r."""
        x_alpha = (r / self.r_s)**self.alp
        return -self.gam - (self.bet - self.gam) * x_alpha / (1 + x_alpha)

    def M(self, r):
        """Enclosed dark matter mass profile for the generalized NFW profile."""
        r_n = r / self.r_s
        a1 = (3.0 - self.gam) / self.alp
        a2 = (self.bet - self.gam) / self.alp
        a3 = 1.0 + (3.0 - self.gam) / self.alp
        a4 = -(r_n**self.alp)
        c1 = (4 * np.pi * self.rho_s * self.r_s**3) / (3.0 - self.gam)
        c2 = r_n ** (3.0 - self.gam)
        return c1 * c2 * sc.hyp2f1(a1, a2, a3, a4)

    def nu(self, r):
        """3D stellar density profile (Plummer)."""
        return 3.0 / (4.0 * np.pi * self.rh**3) * (1 + (r / self.rh)**2)**(-2.5)

    def I(self, R):
        """Projected stellar surface density (Plummer)."""
        return 1.0 / (np.pi * self.rh**2) * (1 + (R / self.rh)**2)**(-2)

    def beta(self, r):
        """Velocity anisotropy parameter for the generalized OM profile.

        Args:
            r (float): Radius [kpc].

        Returns:
            float: Velocity anisotropy beta(r) = beta_0 + (beta_inf - beta_0) * r^2 / (r^2 + r_a^2).
        """
        return self.beta0 + (self.betainf - self.beta0) * r**2 / (r**2 + self.r_a**2)

    def gbeta(self, r):
        """Integrating factor g(r) for the generalized OM profile.

        Args:
            r (float): Radius [kpc].

        Returns:
            float: g(r) = r^(2*beta_0) * (1 + r^2/r_a^2)^(beta_inf - beta_0).
        """
        return r**(2 * self.beta0) * (1 + r**2 / self.r_a**2)**(self.betainf - self.beta0)

    def beta_prime(self, r):
        """4th-order anisotropy analog beta'(r), using the same generalized OM form.
        This is used for the higher-order Jeans equations. Assume to be the same
        functional form with beta(r).
        """
        return self.beta(r)

    def gbeta_prime(self, r):
        """Integrating factor g'(r) for the 4th-order Jeans equations, using the same generalized OM form.
        Assume to be the same functional form with beta(r).
        """
        return self.gbeta(r)

    def dbeta_dr(self, r):
        """ Derivative of the anisotropy parameter beta with respect to radius r."""
        return 2 * (self.betainf - self.beta0) * self.r_a**2 * r / (r**2 + self.r_a**2)**2

    def dbeta_prime_dr(self, r):
        """ Derivative of the 4th-order anisotropy parameter beta' with respect to radius r."""
        return self.dbeta_dr(r)

    ### Jeans modeling methods to compute velocity dispersion profiles ###
    #
    # The integrals are done with fixed rules rather than scipy.quad. The
    # radial ones run to infinity and quad misses the integrand's peak
    # whenever r_star is small in kpc; the projection kernel
    # r / sqrt(r^2 - R^2) is singular at the lower limit, and at quad's
    # default tolerance here the line-of-sight projection came out 3-4% low
    # (2026-09-10 benchmark). Instead sigma_r^2 and <v_r^4> are cumulative
    # trapezoids in log r on a dense grid with a power-law tail, and the
    # projection uses the substitution r = R cosh(u), which removes the
    # singularity and leaves a smooth integrand that a fixed Gauss-Legendre
    # rule handles to ~1e-5, vectorised over all radii at once.
    #
    # Validated against a closed-form Jeans solution, an 18-digit
    # reimplementation and agama's distribution-function moments; see
    # benchmark_sph_model.py. The *_exact methods below are an
    # independent, slow cross-check.

    N_NODES = 40
    CHUNK = 20000

    def _integrand_radii(self):
        """Dense log grid for the radial integrals."""
        return np.logspace(
            np.log10(self.r_grid[0]),
            np.log10(self.r_grid[-1]) + TAIL_DECADES, N_INTEGRAND)

    def _sigma2_r_loglog(self):
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

    def _sigma4_r_loglog(self):
        """
        Cache log <v_r^4> on the dense radial grid.

        <v_r^4>(r) = 3 / (nu g') int_r^inf nu g' G M sigma_r^2 / s^2 ds
        (Eq. C6 of Nguyen et al. 2026) with g' the integrating factor of
        beta'; beta' = beta here.
        """
        if self._log_sigma4_r_grid is None:
            log_s, log_sigma2 = self._sigma2_r_loglog()
            s = np.exp(log_s)
            weight = self.nu(s) * self.gbeta_prime(s)
            f = constants.G * self.M(s) / s ** 2 * weight * np.exp(log_sigma2)
            v4 = 3.0 * _inward_integral(s, f) / weight * _TO_KM2_S2
            self._log_sigma4_r_grid = (log_s, np.log(np.maximum(v4, 1e-300)))
        return self._log_sigma4_r_grid

    def sigma2_r_vec(self, r):
        """sigma_r^2 [km^2/s^2] at arbitrary radii, log-log interpolated."""
        log_r, log_s2 = self._sigma2_r_loglog()
        return np.exp(np.interp(np.log(r), log_r, log_s2))

    def sigma4_r_vec(self, r):
        """<v_r^4> [km^4/s^4] at arbitrary radii, log-log interpolated."""
        log_r, log_v4 = self._sigma4_r_loglog()
        return np.exp(np.interp(np.log(r), log_r, log_v4))

    def _integrand_sigma2_los(self, r, R):
        """Projection integrand of sigma_los^2 without the Abel kernel."""
        return ((1.0 - self.beta(r) * (R / r) ** 2) * self.nu(r)
                * self.sigma2_r_vec(r))

    def _integrand_sigma2_pmR(self, r, R):
        """Projection integrand of sigma_pmR^2 without the Abel kernel."""
        return ((1.0 - self.beta(r) + self.beta(r) * (R / r) ** 2)
                * self.nu(r) * self.sigma2_r_vec(r))

    def _integrand_sigma2_pmT(self, r, R):
        """Projection integrand of sigma_pmT^2 without the Abel kernel."""
        return (1.0 - self.beta(r)) * self.nu(r) * self.sigma2_r_vec(r)

    def _integrand_sigma4_los(self, r, R):
        """
        Projection integrand of <v_los^4> without the Abel kernel.

        Uses F_los(r, R) (Eq. C8), which drops the (beta' - beta) term;
        that term vanishes for beta' = beta.
        """
        return self._F_los(r, R) * self.nu(r) * self.sigma4_r_vec(r)

    def _project(self, R, integrand):
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

    def _project_chunk(self, R, integrand):
        """Projection integral for one chunk of radii, r = R cosh(u)."""
        x, w = leggauss(self.N_NODES)
        umax = np.arccosh(self.r_grid[-1] / R)
        u = 0.5 * umax[:, None] * (x[None, :] + 1.0)
        wu = 0.5 * umax[:, None] * w[None, :]
        cosh_u = np.cosh(u)
        r = R[:, None] * cosh_u
        f = integrand(r, R[:, None]) * R[:, None] * cosh_u
        return 2.0 / self.I(R) * np.sum(wu * f, axis=1)

    def sigma2_los(self, r):
        """Line-of-sight velocity dispersion squared [km^2/s^2] at R [kpc]."""
        return self._project(r, self._integrand_sigma2_los)

    def sigma2_pmR(self, r):
        """Radial proper motion dispersion squared [km^2/s^2] at R [kpc]."""
        return self._project(r, self._integrand_sigma2_pmR)

    def sigma2_pmT(self, r):
        """Tangential proper motion dispersion squared at R [kpc]."""
        return self._project(r, self._integrand_sigma2_pmT)

    ### Higher-order moments code """
    def _F_los(self, r, R):
        """ Higher-order Jeans term F_los(r, R) for the line-of-sight velocity dispersion.
        Eq. (20) in Bañares-Hernández, Read, and Júlio 2025 but without the <v_r^4> term, which is computed separately.
        Separating <v_r^4> requires assuming beta = beta_prime.
        """
        beta_prime_r = self.beta_prime(r)
        dbeta_prime_dr_r = self.dbeta_prime_dr(r)

        a1 = 1 - 2 * beta_prime_r * R**2 / r**2
        a2 = 0.5 * beta_prime_r * (1 + beta_prime_r) * R**4 / r**4
        a3 = -0.25 * dbeta_prime_dr_r * R**4 / r**3
        return a1 + a2 + a3

    def _F_pmT(self, r, R):
        """ Higher-order Jeans term F_pmT(r, R) for the line-of-sight velocity dispersion.
        Eq. (21) in Bañares-Hernández, Read, and Júlio 2025 but without the <v_r^4> term, which is computed separately.
        Separating <v_r^4> requires assuming beta = beta_prime.
        """
        beta_prime_r = self.beta_prime(r)
        dbeta_prime_dr_r = self.dbeta_prime_dr(r)

        a1 = (1 - beta_prime_r) * (2 - beta_prime_r)
        a2 = -0.5 * r * dbeta_prime_dr_r
        return 0.5 * (a1 + a2)

    def _F_pmR(self, r, R):
        """ Higher-order Jeans term F_pmR(r, R) for the line-of-sight velocity dispersion.
        Eq. (22) in Bañares-Hernández, Read, and Júlio 2025 but without the <v_r^4> term, which is computed separately.
        Separating <v_r^4> requires assuming beta = beta_prime.
        """
        beta_prime_r = self.beta_prime(r)
        dbeta_prime_dr_r = self.dbeta_prime_dr(r)

        a1 = (1 - 2 * R**2 / r**2 + R**4 / r**4) * self._F_pmT(r, R)
        a2 = 2 * (1 - beta_prime_r) * R**2 / r**2
        a3 = (1 - 2 * beta_prime_r) * R**4 / r**4
        return a1 + a2 + a3

    def sigma4_los(self, r):
        """Fourth line-of-sight velocity moment [km^4/s^4] at R [kpc]."""
        return self._project(r, self._integrand_sigma4_los)

    def kurtosis_los(self, r):
        """Projected LOS kurtosis kappa(R) = <v^4_los>(R) / sigma^2_los(R)^2."""
        return self.sigma4_los(r) / self.sigma2_los(r) ** 2

    ### High-precision "exact" variants (slow, no interpolation, tight quad tolerances) ###
    def _sigma2_r_exact(self, r):
        """Radial velocity dispersion squared at r (no interpolation)."""
        def integrand(s):
            return constants.G * self.M(s) / s**2 * self.nu(s) * self.gbeta(s)

        c1 = 1.0 / (self.nu(r) * self.gbeta(r))
        integral, _ = quad(integrand, r, np.inf, **self._QUAD_EXACT_KW)
        return c1 * integral * _TO_KM2_S2

    def _sigma4_r_exact(self, r):
        """4th-order radial moment at r (no interpolation)."""
        def integrand(s):
            return self._sigma2_r_exact(s) * constants.G * self.M(s) / s**2 * self.nu(s) * self.gbeta_prime(s)

        c1 = 3.0 / (self.nu(r) * self.gbeta_prime(r))
        integral, _ = quad(integrand, r, np.inf, **self._QUAD_EXACT_KW)
        return c1 * integral * _TO_KM2_S2

    def sigma2_los_exact(self, r):
        """Line-of-sight velocity dispersion profile (high-precision, slow)."""
        result = np.empty(len(r))
        for i, R in enumerate(r):
            def integrand(s, R=R):
                anisotropy_term = 1 - self.beta(s) * (R / s)**2
                kernel = s / np.sqrt(s**2 - R**2)
                return anisotropy_term * self.nu(s) * self._sigma2_r_exact(s) * kernel

            integral, _ = quad(integrand, R, np.inf, **self._QUAD_EXACT_KW)
            result[i] = 2.0 / self.I(R) * integral
        return result

    def sigma2_pmR_exact(self, r):
        """Radial proper motion velocity dispersion profile (high-precision, slow)."""
        result = np.empty(len(r))
        for i, R in enumerate(r):
            def integrand(s, R=R):
                anisotropy_term = 1 - self.beta(s) + self.beta(s) * (R / s)**2
                kernel = s / np.sqrt(s**2 - R**2)
                return anisotropy_term * self.nu(s) * self._sigma2_r_exact(s) * kernel

            integral, _ = quad(integrand, R, np.inf, **self._QUAD_EXACT_KW)
            result[i] = 2.0 / self.I(R) * integral
        return result

    def sigma2_pmT_exact(self, r):
        """Tangential proper motion velocity dispersion profile (high-precision, slow)."""
        result = np.empty(len(r))
        for i, R in enumerate(r):
            def integrand(s, R=R):
                anisotropy_term = 1 - self.beta(s)
                kernel = s / np.sqrt(s**2 - R**2)
                return anisotropy_term * self.nu(s) * self._sigma2_r_exact(s) * kernel

            integral, _ = quad(integrand, R, np.inf, **self._QUAD_EXACT_KW)
            result[i] = 2.0 / self.I(R) * integral
        return result

    def sigma4_los_exact(self, r):
        """Line-of-sight 4th velocity moment (high-precision, slow)."""
        result = np.empty(len(r))
        for i, R in enumerate(r):
            def integrand(s, R=R):
                kernel = s / np.sqrt(s**2 - R**2)
                return self._F_los(s, R) * self.nu(s) * self._sigma4_r_exact(s) * kernel

            integral, _ = quad(integrand, R, np.inf, **self._QUAD_EXACT_KW)
            result[i] = 2.0 / self.I(R) * integral
        return result

    def kurtosis_los_exact(self, r):
        """Projected LOS kurtosis profile kappa(R) = <v^4_los>(R) / sigma^2_los(R)^2 (high-precision, slow)."""
        v4 = self.sigma4_los_exact(r)
        s2 = self.sigma2_los_exact(r)
        return v4 / s2**2

    ### Additional methods that are not directly related to the Jeans modeling but are useful for analysis ###
    def rho_bar(self, r):
        """ Mean enclosed density within radius r."""
        return self.M(r) / (4/3 * np.pi * r**3)

    def r200(self, rho_crit=None, r_min=None, r_max=50):
        """Compute r200 where the mean enclosed density equals 200 * rho_crit.

        For profiles with gamma < 0, rho_bar(r) is non-monotone at small r.
        The search is therefore restricted to r > r_s to avoid the inner region.
        """
        if rho_crit is None:
            rho_crit = acosm.Planck18.critical_density0.to(
                auni.Msun / auni.kpc**3
            ).value

        target = 200.0 * rho_crit
        # r_min = self.r_s if r_min is None else r_min
        # r_min = 1e-3 if r_min is None else r_min

        if self.rho_bar(r_min) < target:
            raise ValueError(
                f"rho_bar(r_min={r_min:.3e}) < target at r_min. "
                f"r200 may be smaller than r_s or the profile is too diffuse."
            )
        if self.rho_bar(r_max) > target:
            raise ValueError(
                f"rho_bar(r_max={r_max:.3e}) > target. "
                f"Increase r_max to bracket r200."
            )

        return brentq(lambda r: self.rho_bar(r) - target, r_min, r_max)

    def M200(self, rho_crit=None, r_min=None, r_max=50):
        """Compute M200 = M(r200)."""
        r200 = self.r200(rho_crit=rho_crit, r_min=r_min, r_max=r_max)
        return self.M(r200).item()



def build_jeans(theta, n_grid=N_GRID):
    """
    Construct the model with the radial grid scaled to r_star.

    The default grid is fixed in kpc, so it misses the integrands whenever
    r_star is far from 0.1 kpc. Here the bounds are set from theta[8], the
    Plummer scale radius.

    Args:
        theta: The 12-element parameter vector [log_rho_s, log_r_s, alp,
            bet, gam, log_r_a, two_to_beta0, two_to_betainf, rh, vsys_los,
            vsys_pmR, vsys_pmT], radii in kpc.
        n_grid: Number of log-spaced radii for the sigma_r^2 grid.

    Returns:
        GeneralizedOMJeans instance with the grid scaled to r_star.
    """
    r_star = float(np.asarray(theta)[8])
    return GeneralizedOMJeans(
        theta, min_rgrid=GRID_MIN_RSTAR * r_star,
        max_rgrid=GRID_MAX_RSTAR * r_star, n_grid=n_grid)


class TwoPopGeneralizedOMJeans:
    """
    Two-population Jeans model sharing a single generalized NFW dark matter halo.

    Each stellar population has its own Plummer profile and generalized
    Osipkov-Merritt anisotropy. Populations are combined using a mixture weight.

    Model parameters:
    - log_rho_s : Logarithm of the characteristic density of the dark matter halo
    - log_r_s   : Logarithm of the scale radius of the dark matter halo
    - alp       : Transition sharpness of the generalized NFW profile
    - bet       : Outer slope of the generalized NFW profile
    - gam       : Inner slope of the generalized NFW profile
    - log_r_a_1 : Logarithm of the anisotropy radius for population 1
    - two_to_beta0_1 : 2 raised to the power of the central anisotropy for population 1
    - two_to_betainf_1 : 2 raised to the power of the outer anisotropy for population 1
    - rh_1       : Plummer scale radius for population 1
    - log_r_a_2 : Logarithm of the anisotropy radius for population 2
    - two_to_beta0_2 : 2 raised to the power of the central anisotropy for population 2
    - two_to_betainf_2 : 2 raised to the power of the outer anisotropy for population 2
    - rh_2       : Plummer scale radius for population 2
    - w1         : Mixture weight for population 1 (0 < w1 < 1)
    - vsys_los   : Systematic velocity offset for line-of-sight velocities
    - vsys_pmR   : Systematic velocity offset for radial proper motions
    - vsys_pmT   : Systematic velocity offset for tangential proper motions

    """
    def __init__(self, theta, min_radius=1e-3, max_radius=5, n_radius=200):
        """
        Args:
            theta (list): Model parameters
                [log_rho_s, log_r_s, alp, bet, gam,
                 log_r_a_1, two_to_beta0_1, two_to_betainf_1, rh_1,
                 log_r_a_2, two_to_beta0_2, two_to_betainf_2, rh_2,
                 w1,
                 vsys_los, vsys_pmR, vsys_pmT].
            min_radius (float, optional): Minimum radius [kpc]. Defaults to 1e-3.
            max_radius (float, optional): Maximum radius [kpc]. Defaults to 5.
            n_radius (int, optional): Number of radius points. Defaults to 200.
        """
        self.param = theta

        # Dark matter halo (shared)
        self.log_rho_s = self.param[0]
        self.rho_s = 10.0 ** self.param[0]
        self.log_r_s = self.param[1]
        self.r_s = 10.0 ** self.param[1]
        self.alp = self.param[2]
        self.bet = self.param[3]
        self.gam = self.param[4]

        # Population 1
        self.log_r_a_1 = self.param[5]
        self.r_a_1 = 10.0 ** self.param[5]
        self.two_to_beta0_1 = self.param[6]
        self.beta0_1 = np.log2(self.two_to_beta0_1)
        self.two_to_betainf_1 = self.param[7]
        self.betainf_1 = np.log2(self.two_to_betainf_1)
        self.rh_1 = self.param[8]

        # Population 2
        self.log_r_a_2 = self.param[9]
        self.r_a_2 = 10.0 ** self.param[9]
        self.two_to_beta0_2 = self.param[10]
        self.beta0_2 = np.log2(self.two_to_beta0_2)
        self.two_to_betainf_2 = self.param[11]
        self.betainf_2 = np.log2(self.two_to_betainf_2)
        self.rh_2 = self.param[12]

        # Mixture weight
        self.w1 = self.param[13]
        self.w2 = 1.0 - self.w1

        # Systematics
        self.vsys_los = self.param[14]
        self.vsys_pmR = self.param[15]
        self.vsys_pmT = self.param[16]

        self.r_vec = np.logspace(np.log10(min_radius), np.log10(max_radius), n_radius)

    def rho(self, r):
        """Dark matter density profile (generalized NFW)."""
        c1 = self.rho_s * (r / self.r_s)**(-self.gam)
        c2 = (1 + (r / self.r_s)**(self.alp))**(-(self.bet - self.gam)/self.alp)
        return c1 * c2

    def M(self, r):
        """Enclosed dark matter mass profile for the generalized NFW profile."""
        r_n = r / self.r_s
        a1 = (3.0 - self.gam) / self.alp
        a2 = (self.bet - self.gam) / self.alp
        a3 = 1.0 + (3.0 - self.gam) / self.alp
        a4 = -(r_n**self.alp)
        c1 = (4 * np.pi * self.rho_s * self.r_s**3) / (3.0 - self.gam)
        c2 = r_n ** (3.0 - self.gam)
        return c1 * c2 * sc.hyp2f1(a1, a2, a3, a4)

    def nu(self, r, pop):
        """3D Plummer stellar density for population pop (1 or 2).

        Args:
            r (float): Radius [kpc].
            pop (int): Population index (1 or 2).

        Returns:
            float: Stellar number density nu(r).
        """
        rh = self.rh_1 if pop == 1 else self.rh_2
        return 3.0 / (4.0 * np.pi * rh**3) * (1 + (r / rh)**2)**(-2.5)

    def I(self, R, pop):
        """Projected Plummer surface density for population pop (1 or 2).

        Args:
            R (float): Projected radius [kpc].
            pop (int): Population index (1 or 2).

        Returns:
            float: Surface density Sigma(R).
        """
        rh = self.rh_1 if pop == 1 else self.rh_2
        return 1.0 / (np.pi * rh**2) * (1 + (R / rh)**2)**(-2)

    def beta(self, r, pop):
        """Velocity anisotropy parameter for population pop (1 or 2).

        Args:
            r (float): Radius [kpc].
            pop (int): Population index (1 or 2).

        Returns:
            float: Velocity anisotropy beta(r) = beta_0 + (beta_inf - beta_0) * r^2 / (r^2 + r_a^2).
        """
        r_a, b0, binf = (
            (self.r_a_1, self.beta0_1, self.betainf_1) if pop == 1
            else (self.r_a_2, self.beta0_2, self.betainf_2)
        )
        return b0 + (binf - b0) * r**2 / (r**2 + r_a**2)

    def gbeta(self, r, pop):
        """Integrating factor g(r) for population pop (1 or 2).

        Args:
            r (float): Radius [kpc].
            pop (int): Population index (1 or 2).

        Returns:
            float: g(r) = r^(2*beta_0) * (1 + r^2/r_a^2)^(beta_inf - beta_0).
        """
        r_a, b0, binf = (
            (self.r_a_1, self.beta0_1, self.betainf_1) if pop == 1
            else (self.r_a_2, self.beta0_2, self.betainf_2)
        )
        return r**(2 * b0) * (1 + r**2 / r_a**2)**(binf - b0)

    def _sigma2_r(self, r, pop):
        """Radial velocity dispersion squared at radius r for population pop."""
        def integrand(s):
            return constants.G * self.M(s) / s**2 * self.nu(s, pop) * self.gbeta(s, pop)

        c1 = 1.0 / (self.nu(r, pop) * self.gbeta(r, pop))
        integral, _ = quad(integrand, r, np.inf, epsabs=1, epsrel=1)
        return c1 * integral * _TO_KM2_S2

    def _sigma2_r_fn(self, r_vec, pop):
        """Interpolator for sigma_r^2 of population pop over r_vec."""
        return interp1d(
            r_vec,
            [self._sigma2_r(r, pop) for r in r_vec],
            bounds_error=False,
            fill_value=0.0,
            kind="linear",
        )

    def _sigma2_los_R_single(self, R, sigma2_r_fn, pop):
        """Line-of-sight velocity dispersion squared at projected radius R for a single population."""
        def integrand(r):
            anisotropy_term = 1 - self.beta(r, pop) * (R / r)**2
            kernel = r / np.sqrt(r**2 - R**2)
            return anisotropy_term * self.nu(r, pop) * sigma2_r_fn(r) * kernel

        integral, _ = quad(integrand, R, np.inf, epsabs=1, epsrel=1)
        return 2.0 / self.I(R, pop) * integral

    def _sigma2_pmR_R_single(self, R, sigma2_r_fn, pop):
        """Radial proper motion velocity dispersion squared at projected radius R for a single population."""
        def integrand(r):
            anisotropy_term = 1 - self.beta(r, pop) + self.beta(r, pop) * (R / r)**2
            kernel = r / np.sqrt(r**2 - R**2)
            return anisotropy_term * self.nu(r, pop) * sigma2_r_fn(r) * kernel

        integral, _ = quad(integrand, R, np.inf, epsabs=1, epsrel=1)
        return 2.0 / self.I(R, pop) * integral

    def _sigma2_pmT_R_single(self, R, sigma2_r_fn, pop):
        """Tangential proper motion velocity dispersion squared at projected radius R for a single population."""
        def integrand(r):
            anisotropy_term = 1 - self.beta(r, pop)
            kernel = r / np.sqrt(r**2 - R**2)
            return anisotropy_term * self.nu(r, pop) * sigma2_r_fn(r) * kernel

        integral, _ = quad(integrand, R, np.inf, epsabs=1, epsrel=1)
        return 2.0 / self.I(R, pop) * integral

    def sigma2_los(self, min_radius=1e-3):
        """Compute the mixture-weighted line-of-sight velocity dispersion profile."""
        r_grid = np.logspace(np.log10(min_radius), np.log10(50), 500)
        fn1 = self._sigma2_r_fn(r_grid, 1)
        fn2 = self._sigma2_r_fn(r_grid, 2)

        sig1_sq = np.array([self._sigma2_los_R_single(R, fn1, 1) for R in self.r_vec])
        sig2_sq = np.array([self._sigma2_los_R_single(R, fn2, 2) for R in self.r_vec])

        return self.w1 * sig1_sq + self.w2 * sig2_sq

    def sigma2_pmR(self, min_radius=1e-3):
        """Compute the mixture-weighted radial proper motion velocity dispersion profile."""
        r_grid = np.logspace(np.log10(min_radius), np.log10(50), 500)
        fn1 = self._sigma2_r_fn(r_grid, 1)
        fn2 = self._sigma2_r_fn(r_grid, 2)

        sig1_sq = np.array([self._sigma2_pmR_R_single(R, fn1, 1) for R in self.r_vec])
        sig2_sq = np.array([self._sigma2_pmR_R_single(R, fn2, 2) for R in self.r_vec])

        return self.w1 * sig1_sq + self.w2 * sig2_sq

    def sigma2_pmT(self, min_radius=1e-3):
        """Compute the mixture-weighted tangential proper motion velocity dispersion profile."""
        r_grid = np.logspace(np.log10(min_radius), np.log10(50), 500)
        fn1 = self._sigma2_r_fn(r_grid, 1)
        fn2 = self._sigma2_r_fn(r_grid, 2)

        sig1_sq = np.array([self._sigma2_pmT_R_single(R, fn1, 1) for R in self.r_vec])
        sig2_sq = np.array([self._sigma2_pmT_R_single(R, fn2, 2) for R in self.r_vec])

        return self.w1 * sig1_sq + self.w2 * sig2_sq

    def beta_eff(self, r):
        """Effective velocity anisotropy from mixture-weighted combination.

        This is a diagnostic quantity, NOT a physical anisotropy for either population.

        Computed as:
        beta_eff(r) = [w1 * nu_1(r) * sigma_r1^2(r) * beta_1(r) + w2 * nu_2(r) * sigma_r2^2(r) * beta_2(r)]
                      / [w1 * nu_1(r) * sigma_r1^2(r) + w2 * nu_2(r) * sigma_r2^2(r)]

        Args:
            r (float or array): Radius [kpc].

        Returns:
            float or array: Effective anisotropy beta_eff(r).
        """
        # Compute sigma_r^2 for both populations
        r_grid = np.logspace(np.log10(1e-3), np.log10(50), 500)
        fn1 = self._sigma2_r_fn(r_grid, 1)
        fn2 = self._sigma2_r_fn(r_grid, 2)

        # Handle scalar or array input
        r = np.atleast_1d(r)
        beta_eff_vals = np.zeros_like(r)

        for i, r_val in enumerate(r):
            nu1 = self.nu(r_val, 1)
            nu2 = self.nu(r_val, 2)
            sig1_sq = fn1(r_val)
            sig2_sq = fn2(r_val)
            b1 = self.beta(r_val, 1)
            b2 = self.beta(r_val, 2)

            numerator = self.w1 * nu1 * sig1_sq * b1 + self.w2 * nu2 * sig2_sq * b2
            denominator = self.w1 * nu1 * sig1_sq + self.w2 * nu2 * sig2_sq

            beta_eff_vals[i] = numerator / denominator if denominator > 0 else 0.0

        return beta_eff_vals[0] if len(r) == 1 else beta_eff_vals