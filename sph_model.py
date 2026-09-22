
import numpy as np
import scipy.special as sc
from scipy import constants
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import brentq

from .quadrature import leg_nodes

import astropy.units as auni
import astropy.cosmology as acosm

_TO_KM2_S2 = 1.989e12 / 3.0856  # G unit conversion: (10^7 M_sun, kpc) -> km^2/s^2

# Radial grid for sigma_r^2, in units of r_star, used by build_jeans. Stars
# are kept inside 10 r_star and the Plummer density falls as r^-5, so
# 1e3 r_star is far enough for the projection integral. A Zhao tracer
# falls as r^-beta_star with beta_star >= 3.2; ZhaoTracerOMJeans.I adds
# the power-law tail beyond the grid analytically.
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

    The fourth-order anisotropy beta'(r) = 1 - 3 <v_r^2 v_theta^2> / <v_r^4>
    (Eq. 12 of Bañares-Hernández, Read & Júlio 2025) has the same form with
    its own parameters, `theta_prime`; by default beta' = beta. The fourth
    moments (`sigma4_los`, `sigma4_pmR`, `sigma4_pmT`, `sigma4_theta`)
    carry the (beta' - beta) coupling term, so they are general in both.
    """
    # Reason: these were epsabs=1, epsrel=1 -- the same loose setting as
    # the old fast path, so the "high-precision" variants were not
    # actually precise and could not serve as an independent check.
    _QUAD_EXACT_KW = dict(epsabs=0, epsrel=1e-10, limit=400)

    def __init__(self, theta, min_rgrid=1e-3, max_rgrid=50, n_grid=500,
                 theta_prime=None):
        """
        Args:
            theta (list): Model parameters [log_rho_s, log_r_s, alp, bet, gam,
                log_r_a, two_to_beta0, two_to_betainf, rh,
                vsys_los, vsys_pmR, vsys_pmT].
            theta_prime (list, optional): [log_r_a', two_to_beta0',
                two_to_betainf'] of the fourth-order anisotropy beta'(r);
                None for beta' = beta.
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

        # fourth-order anisotropy beta'(r): the same generalized OM form with
        # its own (log_r_a', two_to_beta0', two_to_betainf'). None ties it to
        # beta(r), which is exact for f(E) L^-2beta, Osipkov-Merritt and
        # Cuddeford DFs and reproduces every result from before 2026-09-21.
        if theta_prime is None:
            theta_prime = (self.log_r_a, self.two_to_beta0, self.two_to_betainf)
        (self.log_r_a_prime, self.two_to_beta0_prime,
         self.two_to_betainf_prime) = theta_prime
        self.beta0_prime = np.log2(self.two_to_beta0_prime)
        self.betainf_prime = np.log2(self.two_to_betainf_prime)
        self.r_a_prime = 10.0 ** self.log_r_a_prime

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

    @staticmethod
    def _om_beta(r, beta0, betainf, r_a):
        """beta(r) = beta_0 + (beta_inf - beta_0) r^2 / (r^2 + r_a^2)."""
        return beta0 + (betainf - beta0) * r**2 / (r**2 + r_a**2)

    @staticmethod
    def _om_gbeta(r, beta0, betainf, r_a):
        """g(r) = r^(2 beta_0) (1 + r^2 / r_a^2)^(beta_inf - beta_0), so dln g / dln r = 2 beta(r)."""
        return r**(2 * beta0) * (1 + r**2 / r_a**2)**(betainf - beta0)

    @staticmethod
    def _om_dbeta_dr(r, beta0, betainf, r_a):
        """d beta / d r of the generalized OM profile."""
        return 2 * (betainf - beta0) * r_a**2 * r / (r**2 + r_a**2)**2

    def beta(self, r):
        """Velocity anisotropy parameter for the generalized OM profile.

        Args:
            r (float): Radius [kpc].

        Returns:
            float: Velocity anisotropy beta(r) = beta_0 + (beta_inf - beta_0) * r^2 / (r^2 + r_a^2).
        """
        return self._om_beta(r, self.beta0, self.betainf, self.r_a)

    def gbeta(self, r):
        """Integrating factor g(r) for the generalized OM profile.

        Args:
            r (float): Radius [kpc].

        Returns:
            float: g(r) = r^(2*beta_0) * (1 + r^2/r_a^2)^(beta_inf - beta_0).
        """
        return self._om_gbeta(r, self.beta0, self.betainf, self.r_a)

    def dbeta_dr(self, r):
        """ Derivative of the anisotropy parameter beta with respect to radius r."""
        return self._om_dbeta_dr(r, self.beta0, self.betainf, self.r_a)

    def beta_prime(self, r):
        """
        Fourth-order anisotropy beta'(r) = 1 - 3 <v_r^2 v_theta^2> / <v_r^4>.

        Eq. (12) of Bañares-Hernández, Read & Júlio 2025. Same generalized OM
        form as beta(r) with the theta_prime parameters; equal to beta(r) when
        theta_prime was not given.
        """
        return self._om_beta(r, self.beta0_prime, self.betainf_prime, self.r_a_prime)

    def gbeta_prime(self, r):
        """Integrating factor g'(r) of the radial fourth-order Jeans equation, dln g' / dln r = 2 beta'(r)."""
        return self._om_gbeta(r, self.beta0_prime, self.betainf_prime, self.r_a_prime)

    def dbeta_prime_dr(self, r):
        """ Derivative of the 4th-order anisotropy parameter beta' with respect to radius r."""
        return self._om_dbeta_dr(r, self.beta0_prime, self.betainf_prime, self.r_a_prime)

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

        F_los(r, R) <v_r^4> nu plus the (beta' - beta) coupling term with
        the line-of-sight weight (R/r)^4 (Eq. 20 of Bañares-Hernández et
        al. 2025); the second part is exactly zero for beta' = beta.
        """
        return (self._F_los(r, R) * self.nu(r) * self.sigma4_r_vec(r)
                + self._coupling(r, self.sigma2_r_vec(r)) * (R / r) ** 4
                * self.nu(r))

    def _integrand_sigma4_pmR(self, r, R):
        """Projection integrand of <v_pmR^4>; coupling weight (1 - R^2/r^2)^2."""
        return (self._F_pmR(r, R) * self.nu(r) * self.sigma4_r_vec(r)
                + self._coupling(r, self.sigma2_r_vec(r))
                * (1 - (R / r) ** 2) ** 2 * self.nu(r))

    def _integrand_sigma4_pmT(self, r, R):
        """Projection integrand of <v_pmT^4>, the projected <v_theta^4>; coupling weight 1."""
        return (self._F_pmT(r, R) * self.nu(r) * self.sigma4_r_vec(r)
                + self._coupling(r, self.sigma2_r_vec(r)) * self.nu(r))

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
        return self._abel(R, integrand) / self.I(R)

    def _abel(self, R, integrand):
        """
        2 int_R^rmax integrand(r, R) r / sqrt(r^2 - R^2) dr, rmax = r_grid[-1].

        Args:
            R: Projected radii [kpc], 1-D array.
            integrand: Callable of (r, R) broadcasting over both.

        Returns:
            The truncated Abel integral at each R.
        """
        # Reason: the (n_radii, N_NODES) work arrays reach ~1 GB for the
        # 1e5-star samples used in the notebooks, so evaluate in chunks.
        out = np.empty_like(R)
        for start in range(0, len(R), self.CHUNK):
            sl = slice(start, start + self.CHUNK)
            out[sl] = self._abel_chunk(R[sl], integrand)
        return out

    def _abel_chunk(self, R, integrand):
        """Abel integral for one chunk of radii, r = R cosh(u)."""
        x, w = leg_nodes(self.N_NODES)
        umax = np.arccosh(self.r_grid[-1] / R)
        u = 0.5 * umax[:, None] * (x[None, :] + 1.0)
        wu = 0.5 * umax[:, None] * w[None, :]
        cosh_u = np.cosh(u)
        r = R[:, None] * cosh_u
        f = integrand(r, R[:, None]) * R[:, None] * cosh_u
        return 2.0 * np.sum(wu * f, axis=1)

    def sigma2_los(self, r):
        """Line-of-sight velocity dispersion squared [km^2/s^2] at R [kpc]."""
        return self._project(r, self._integrand_sigma2_los)

    def sigma2_pmR(self, r):
        """Radial proper motion dispersion squared [km^2/s^2] at R [kpc]."""
        return self._project(r, self._integrand_sigma2_pmR)

    def sigma2_pmT(self, r):
        """Tangential proper motion dispersion squared at R [kpc]."""
        return self._project(r, self._integrand_sigma2_pmT)

    ### Higher-order moments ###
    #
    # Radial: <v_r^4> from the fourth-order Jeans equation with the
    # integrating factor of beta' (_sigma4_r_loglog). Tangential: the
    # tangential fourth-order Jeans equation gives, with no further closure,
    #     <v_theta^4> = F_pmT <v_r^4> + (3/4) (beta' - beta) sigma_r^2 G M / r,
    # and every projection is a geometric combination of <v_r^4>,
    # <v_r^2 v_theta^2> = (1 - beta') <v_r^4> / 3 and <v_theta^4>. The F
    # kernels below multiply <v_r^4>; the (beta' - beta) coupling enters the
    # _integrand_sigma4_* methods with weight (R/r)^4 (los), 1 (pmT) and
    # (1 - R^2/r^2)^2 (pmR), and vanishes for beta' = beta. Eqs. (13) and
    # (20)-(22) of Bañares-Hernández, Read & Júlio 2025; derivation in
    # sbi_twins/notes/fourth_order_jeans_beta_prime.md.
    def _F_los(self, r, R):
        """Kernel on <v_r^4> in <v_los^4>(R): Eq. (20) of Bañares-Hernández, Read & Júlio 2025 without its (beta' - beta) term."""
        beta_prime_r = self.beta_prime(r)
        dbeta_prime_dr_r = self.dbeta_prime_dr(r)

        a1 = 1 - 2 * beta_prime_r * R**2 / r**2
        a2 = 0.5 * beta_prime_r * (1 + beta_prime_r) * R**4 / r**4
        a3 = -0.25 * dbeta_prime_dr_r * R**4 / r**3
        return a1 + a2 + a3

    def _F_pmT(self, r, R):
        """Kernel on <v_r^4> in <v_pmT^4>(R), i.e. <v_theta^4> / <v_r^4> for beta' = beta: Eq. (21)."""
        beta_prime_r = self.beta_prime(r)
        dbeta_prime_dr_r = self.dbeta_prime_dr(r)

        a1 = (1 - beta_prime_r) * (2 - beta_prime_r)
        a2 = -0.5 * r * dbeta_prime_dr_r
        return 0.5 * (a1 + a2)

    def _F_pmR(self, r, R):
        """Kernel on <v_r^4> in <v_pmR^4>(R): Eq. (22). Equals 1 at R = r, where v_pmR = v_r."""
        beta_prime_r = self.beta_prime(r)

        a1 = (1 - 2 * R**2 / r**2 + R**4 / r**4) * self._F_pmT(r, R)
        a2 = 2 * (1 - beta_prime_r) * R**2 / r**2
        # Reason: this term had a + sign until 2026-09-21, giving 3 - 4 beta'
        # instead of 1 at the tangent point; the minus follows from
        # v_pmR = v_r sin(a) - v_theta cos(a), and GravSphere2 has it too.
        a3 = -(1 - 2 * beta_prime_r) * R**4 / r**4
        return a1 + a2 + a3

    def _coupling(self, r, sigma2_r):
        """
        (3/4) (beta' - beta) sigma_r^2 G M / r [km^4/s^4].

        The term of the tangential fourth-order Jeans equation that carries
        the second-order anisotropy into <v_theta^4>; zero for beta' = beta.
        sigma2_r is passed in so the interpolated and the exact paths share it.
        """
        return (0.75 * (self.beta_prime(r) - self.beta(r)) * sigma2_r
                * constants.G * self.M(r) / r * _TO_KM2_S2)

    def sigma4_theta(self, r):
        """
        Intrinsic tangential fourth moment <v_theta^4> = <v_phi^4> [km^4/s^4] at r [kpc].

        F_pmT <v_r^4> + (3/4) (beta' - beta) sigma_r^2 G M / r. A negative
        value means the (beta, beta') pair admits no distribution function;
        GravSphere2 rejects such models and so do the DF-free simulators.
        """
        r = np.asarray(r, dtype=float)
        return (self._F_pmT(r, r) * self.sigma4_r_vec(r)
                + self._coupling(r, self.sigma2_r_vec(r)))

    def sigma4_los(self, r):
        """Fourth line-of-sight velocity moment [km^4/s^4] at R [kpc]."""
        return self._project(r, self._integrand_sigma4_los)

    def sigma4_pmR(self, r):
        """Fourth radial proper-motion velocity moment [km^4/s^4] at R [kpc]."""
        return self._project(r, self._integrand_sigma4_pmR)

    def sigma4_pmT(self, r):
        """Fourth tangential proper-motion velocity moment [km^4/s^4] at R [kpc]."""
        return self._project(r, self._integrand_sigma4_pmT)

    def kurtosis_los(self, r):
        """Projected LOS kurtosis kappa(R) = <v^4_los>(R) / sigma^2_los(R)^2."""
        return self.sigma4_los(r) / self.sigma2_los(r) ** 2

    def kurtosis_pmR(self, r):
        """Projected radial proper-motion kurtosis <v^4_pmR>(R) / sigma^2_pmR(R)^2."""
        return self.sigma4_pmR(r) / self.sigma2_pmR(r) ** 2

    def kurtosis_pmT(self, r):
        """Projected tangential proper-motion kurtosis <v^4_pmT>(R) / sigma^2_pmT(R)^2."""
        return self.sigma4_pmT(r) / self.sigma2_pmT(r) ** 2

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
                coupling = self._coupling(s, self._sigma2_r_exact(s)) * (R / s)**4
                return (self._F_los(s, R) * self._sigma4_r_exact(s) + coupling) * self.nu(s) * kernel

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

    def r200(self, rho_crit=None, r_min=0.001, r_max=50):
        """Compute r200 where the mean enclosed density equals 200 * rho_crit.

        For profiles with gamma < 0, rho_bar(r) is non-monotone at small r.
        The search is therefore restricted to r > r_s to avoid the inner region.
        """
        if rho_crit is None:
            rho_crit = acosm.Planck18.critical_density0.to(
                1e7 * auni.Msun / auni.kpc**3
            ).value
        target = 200.0 * rho_crit

        if self.rho_bar(r_min) < target:
            raise ValueError(
                f"rho_bar(r_min={r_min:.3e}) < target at r_min. "
                f"Value {self.rho_bar(r_min)} vs {target}. "
                f"r200 may be smaller than r_s or the profile is too diffuse."
            )
        if self.rho_bar(r_max) > target:
            raise ValueError(
                f"rho_bar(r_max={r_max:.3e}) > target. "
                f"Value {self.rho_bar(r_max)} vs {target}. "
                f"Increase r_max to bracket r200."
            )

        return brentq(lambda r: self.rho_bar(r) - target, r_min, r_max)

    def M200(self, rho_crit=None, r_min=0.001, r_max=50):
        """Compute M200 = M(r200)."""
        r200 = self.r200(rho_crit=rho_crit, r_min=r_min, r_max=r_max)
        return self.M(r200).item()


class ZhaoTracerOMJeans(GeneralizedOMJeans):
    """
    GeneralizedOMJeans with an alpha-beta-gamma (Zhao 1996) stellar tracer.

    nu(r) = nu_0 x^-gamma_star (1 + x^alpha_star)^(-(beta_star - gamma_star)
    / alpha_star) with x = r / rh, normalised to unit mass; rh (theta[8]) is
    the Zhao scale radius, not a half-light radius. Plummer is
    (alpha_star, beta_star, gamma_star) = (2, 5, 0). The surface density
    has no closed form and is projected numerically with the same
    r = R cosh(u) rule as the velocity moments. Everything else -- the
    halo, the anisotropy, the Jeans integrals -- is inherited unchanged.
    """

    def __init__(self, theta, alpha_star=2.0, beta_star=5.0,
                 gamma_star=0.0, **kwargs):
        """
        Args:
            theta: As for GeneralizedOMJeans; theta[8] is the Zhao scale
                radius [kpc].
            alpha_star: Transition sharpness of the tracer profile.
            beta_star: Outer slope of the tracer profile, > 3.
            gamma_star: Inner slope of the tracer profile, < 3.
            **kwargs: Grid arguments passed to GeneralizedOMJeans.

        Raises:
            ValueError: If the tracer mass is not finite.
        """
        super().__init__(theta, **kwargs)
        if not gamma_star < 3.0 < beta_star:
            raise ValueError(
                f"Zhao tracer needs gamma_star < 3 < beta_star, got "
                f"gamma_star={gamma_star}, beta_star={beta_star}")
        self.alpha_star = alpha_star
        self.beta_star = beta_star
        self.gamma_star = gamma_star
        # Reason: unit total mass, so nu and I stay a consistent pair
        # (the moments themselves do not depend on the normalisation).
        self._nu0 = alpha_star / (4.0 * np.pi * self.rh ** 3 * sc.beta(
            (3.0 - gamma_star) / alpha_star, (beta_star - 3.0) / alpha_star))

    def nu(self, r):
        """3D stellar density profile (Zhao), unit total mass."""
        x = r / self.rh
        return (self._nu0 * x ** (-self.gamma_star)
                * (1.0 + x ** self.alpha_star)
                ** (-(self.beta_star - self.gamma_star) / self.alpha_star))

    def I(self, R):
        """Projected stellar surface density (Zhao), by Abel projection."""
        R_arr = np.atleast_1d(np.asarray(R, dtype=float))
        sigma = self._abel(R_arr, lambda r, _: self.nu(r))
        # Reason: the rule stops at r_grid[-1]; beyond it nu is the pure
        # power law r^-beta_star, for which the remaining Abel integral
        # is 2 nu r / (b - 1) 2F1(1/2, (b-1)/2; (b+1)/2; (R/r)^2).
        r_max = self.r_grid[-1]
        b = self.beta_star
        sigma += (2.0 * self.nu(r_max) * r_max / (b - 1.0)
                  * sc.hyp2f1(0.5, 0.5 * (b - 1.0), 0.5 * (b + 1.0),
                              (R_arr / r_max) ** 2))
        return sigma[0] if np.ndim(R) == 0 else sigma


def _bvh_beta(r, beta0, betainf, r0, n):
    """beta(r) = beta0 + (betainf - beta0) (r / r0)^n / (1 + (r / r0)^n)."""
    x = (r / r0) ** n
    return beta0 + (betainf - beta0) * x / (1.0 + x)


def _bvh_gbeta(r, beta0, betainf, r0, n):
    """
    Integrating factor with d ln g / d ln r = 2 beta(r).

    g(r) = (r / r0)^(2 beta0) (1 + (r / r0)^n)^(2 (betainf - beta0) / n). The
    r0^(-2 beta0) normalisation is a constant that cancels in the Jeans
    integrals; it keeps the powers inside float range down to beta0 = -18
    (the symmetrised beta / (2 - beta) = -0.9 of the GravSphere priors).
    """
    x = (r / r0) ** n
    return (r / r0) ** (2.0 * beta0) * (1.0 + x) ** (2.0 * (betainf - beta0) / n)


def _bvh_dbeta_dr(r, beta0, betainf, r0, n):
    """d beta / d r of the Baes & van Hese profile."""
    x = (r / r0) ** n
    return (betainf - beta0) * n * x / (r * (1.0 + x) ** 2)


class BaesVanHeseJeans(GeneralizedOMJeans):
    """
    GeneralizedOMJeans with Baes & van Hese (2007) anisotropy profiles.

    beta(r) = beta0 + (betainf - beta0) (r / r0)^n / (1 + (r / r0)^n), the
    four-parameter form fitted by GravSphere (Read & Steger 2017) and
    GravSphere2. n = 2 is the generalised Osipkov-Merritt profile of the
    parent class, and n = 2, betainf = 1 the Cuddeford-Osipkov-Merritt DF of
    the agama mocks. r0 sits in the log_r_a slot of theta (theta[5]) and n
    is a constructor argument, as the tracer shape is for ZhaoTracerOMJeans.
    The fourth-order anisotropy beta'(r) has the same form with the parent's
    theta_prime = (log_r0', two_to_beta0', two_to_betainf') and
    n_aniso_prime, as GravSphere2 fits it (its Eq. 43); by default
    beta' = beta. Everything else -- tracer, halo, Jeans integrals, the
    (beta' - beta) coupling -- is inherited unchanged.
    """

    def __init__(self, theta, n_aniso=2.0, n_aniso_prime=None, **kwargs):
        """
        Args:
            theta: As for GeneralizedOMJeans; theta[5] is log10 r0 [kpc].
            n_aniso: Sharpness n of the anisotropy transition.
            n_aniso_prime: Sharpness n' of beta'(r); None copies n_aniso.
            **kwargs: theta_prime and the grid arguments of
                GeneralizedOMJeans.
        """
        super().__init__(theta, **kwargs)
        self.n_aniso = n_aniso
        self.n_aniso_prime = n_aniso if n_aniso_prime is None else n_aniso_prime

    def beta(self, r):
        """Velocity anisotropy beta(r) of the Baes & van Hese profile."""
        return _bvh_beta(r, self.beta0, self.betainf, self.r_a, self.n_aniso)

    def gbeta(self, r):
        """Integrating factor g(r) with d ln g / d ln r = 2 beta(r)."""
        return _bvh_gbeta(r, self.beta0, self.betainf, self.r_a, self.n_aniso)

    def dbeta_dr(self, r):
        """d beta / d r of the Baes & van Hese profile."""
        return _bvh_dbeta_dr(r, self.beta0, self.betainf, self.r_a, self.n_aniso)

    def beta_prime(self, r):
        """Fourth-order anisotropy beta'(r), Baes & van Hese form."""
        return _bvh_beta(r, self.beta0_prime, self.betainf_prime, self.r_a_prime,
                         self.n_aniso_prime)

    def gbeta_prime(self, r):
        """Integrating factor g'(r) with d ln g' / d ln r = 2 beta'(r)."""
        return _bvh_gbeta(r, self.beta0_prime, self.betainf_prime, self.r_a_prime,
                          self.n_aniso_prime)

    def dbeta_prime_dr(self, r):
        """d beta' / d r of the Baes & van Hese profile."""
        return _bvh_dbeta_dr(r, self.beta0_prime, self.betainf_prime,
                             self.r_a_prime, self.n_aniso_prime)


def build_jeans(theta, n_grid=N_GRID, tracer=None, theta_prime=None):
    """
    Construct the model with the radial grid scaled to r_star.

    The default grid is fixed in kpc, so it misses the integrands whenever
    r_star is far from 0.1 kpc. Here the bounds are set from theta[8], the
    tracer scale radius.

    Args:
        theta: The 12-element parameter vector [log_rho_s, log_r_s, alp,
            bet, gam, log_r_a, two_to_beta0, two_to_betainf, rh, vsys_los,
            vsys_pmR, vsys_pmT], radii in kpc.
        n_grid: Number of log-spaced radii for the sigma_r^2 grid.
        tracer: (alpha_star, beta_star, gamma_star) of a Zhao light
            profile, or None for the Plummer tracer.
        theta_prime: [log_r_a', two_to_beta0', two_to_betainf'] of the
            fourth-order anisotropy beta'(r), or None for beta' = beta.

    Returns:
        GeneralizedOMJeans (or ZhaoTracerOMJeans) instance with the grid
        scaled to r_star.
    """
    r_star = float(np.asarray(theta)[8])
    grid = dict(min_rgrid=GRID_MIN_RSTAR * r_star,
                max_rgrid=GRID_MAX_RSTAR * r_star, n_grid=n_grid,
                theta_prime=theta_prime)
    if tracer is None:
        return GeneralizedOMJeans(theta, **grid)
    return ZhaoTracerOMJeans(theta, *tracer, **grid)

