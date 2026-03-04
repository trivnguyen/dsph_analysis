
import numpy as np
import scipy.special as sc
from scipy import constants
from scipy.interpolate import interp1d
from scipy.integrate import quad

_TO_KM2_S2 = 1.989e12 / 3.0856  # G unit conversion: (10^7 M_sun, kpc) -> km^2/s^2


class GeneralizedOMJeans:
    """
    A class to model the velocity dispersion profile using the Jeans equation
    with a generalized Osipkov-Merritt anisotropy profile.

    The generalized OM profile allows arbitrary central (beta_0) and outer
    (beta_inf) anisotropy, with a transition controlled by the anisotropy
    radius r_a.

    Model parameters:
    - log_rho_s: Logarithm of the characteristic density of the dark matter halo
    - log_r_s: Logarithm of the scale radius of the dark matter halo

    """
    def __init__(self, theta, min_radius=1e-3, max_radius=5, n_radius=200):
        """
        Args:
            theta (list): Model parameters [log_rho_s, log_r_s, alp, bet, gam,
                log_r_a, two_to_beta0, two_to_betainf, rh,
                vsys_los, vsys_pmR, vsys_pmT].
            min_radius (float, optional): Minimum radius [kpc]. Defaults to 1e-3.
            max_radius (float, optional): Maximum radius [kpc]. Defaults to 5.
            n_radius (int, optional): Number of radius points. Defaults to 200.
        """
        self.param = theta
        self.log_rho_s = self.param[0]
        self.rho_s = 10.0 ** self.param[0]
        self.log_r_s = self.param[1]
        self.r_s = 10.0 ** self.param[1]
        self.alp = self.param[2]
        self.bet = self.param[3]
        self.gam = self.param[4]
        self.log_r_a = self.param[5]
        self.r_a = 10.0 ** self.param[5]
        self.two_to_beta0 = self.param[6]
        self.beta0 = np.log2(self.two_to_beta0)
        self.two_to_betainf = self.param[7]
        self.betainf = np.log2(self.two_to_betainf)
        self.rh = self.param[8]
        self.vsys_los = self.param[9]
        self.vsys_pmR = self.param[10]
        self.vsys_pmT = self.param[11]

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

    def _sigma2_r(self, r):
        """Radial velocity dispersion squared at radius r."""
        def integrand(s):
            return constants.G * self.M(s) / s**2 * self.nu(s) * self.gbeta(s)

        c1 = 1.0 / (self.nu(r) * self.gbeta(r))
        integral, _ = quad(integrand, r, np.inf, epsabs=1, epsrel=1)
        return c1 * integral * _TO_KM2_S2

    def _sigma2_los_R(self, R, sigma2_r_fn):
        """Line-of-sight velocity dispersion squared at projected radius R."""
        def integrand(r):
            anisotropy_term = 1 - self.beta(r) * (R / r)**2
            kernel = r / np.sqrt(r**2 - R**2)
            return anisotropy_term * self.nu(r) * sigma2_r_fn(r) * kernel

        integral, _ = quad(integrand, R, np.inf, epsabs=1, epsrel=1)
        return 2.0 / self.I(R) * integral

    def _sigma2_pmR_R(self, R, sigma2_r_fn):
        """Radial proper motion velocity dispersion squared at projected radius R."""
        def integrand(r):
            anisotropy_term = 1 - self.beta(r) + self.beta(r) * (R / r)**2
            kernel = r / np.sqrt(r**2 - R**2)
            return anisotropy_term * self.nu(r) * sigma2_r_fn(r) * kernel

        integral, _ = quad(integrand, R, np.inf, epsabs=1, epsrel=1)
        return 2.0 / self.I(R) * integral

    def _sigma2_pmT_R(self, R, sigma2_r_fn):
        """Tangential proper motion velocity dispersion squared at projected radius R."""
        def integrand(r):
            anisotropy_term = 1 - self.beta(r)
            kernel = r / np.sqrt(r**2 - R**2)
            return anisotropy_term * self.nu(r) * sigma2_r_fn(r) * kernel

        integral, _ = quad(integrand, R, np.inf, epsabs=1, epsrel=1)
        return 2.0 / self.I(R) * integral

    def sigma2(self, r_vec):
        """Compute sigma_r^2 over an array of radii and return an interpolator."""
        return interp1d(
            r_vec,
            list(map(self._sigma2_r, r_vec)),
            bounds_error=False,
            fill_value=0.0,
            kind="linear",
        )

    def sigma2_los(self, min_radius=1e-3):
        """Compute the line-of-sight velocity dispersion profile."""
        r_grid = np.logspace(np.log10(min_radius), np.log10(50), 500)
        sigma2_r_fn = self.sigma2(r_grid)
        return np.array([self._sigma2_los_R(R, sigma2_r_fn) for R in self.r_vec])

    def sigma2_pmR(self, min_radius=1e-3):
        """Compute the radial proper motion velocity dispersion profile."""
        r_grid = np.logspace(np.log10(min_radius), np.log10(50), 500)
        sigma2_r_fn = self.sigma2(r_grid)
        return np.array([self._sigma2_pmR_R(R, sigma2_r_fn) for R in self.r_vec])

    def sigma2_pmT(self, min_radius=1e-3):
        """Compute the tangential proper motion velocity dispersion profile."""
        r_grid = np.logspace(np.log10(min_radius), np.log10(50), 500)
        sigma2_r_fn = self.sigma2(r_grid)
        return np.array([self._sigma2_pmT_R(R, sigma2_r_fn) for R in self.r_vec])


class TwoPopGeneralizedOMJeans:
    """
    Two-population Jeans model sharing a single generalized NFW dark matter halo.

    Each stellar population has its own Plummer profile and generalized
    Osipkov-Merritt anisotropy. Populations are combined using a mixture weight.

    Model parameters:
    - log_rho_s : Logarithm of the characteristic density of the dark matter halo
    - log_r_s   : Logarithm of the scale radius of the dark matter halo
    - w1        : Mixture weight for population 1 (w2 = 1 - w1)
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