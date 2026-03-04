
from typing import Optional, List, Dict
import numpy as np
import bilby

from .solver import calc_sigma2_los


def log_gaussian(x, mu, var):
    """ Calculate the log of Gaussian distribution """
    ll = -0.5 * (x - mu)**2 / var
    ll = ll - 0.5 * np.log(2 * np.pi * var)
    return ll


class JeansModel(bilby.Likelihood):

    def __init__(self, data: Dict, priors: Dict) -> None:
        """
        Parameters
        ----------
        data: Dict
            Dictionary containing the kinematics data
        priors: Dict
            The priors in bilby format
        """

        super().__init__(
            parameters={
                'gamma': None,
                'r_dm': None,
                'rho_dm': None,
                'beta_ani_0': None,
                'r_ani': None,
                'r_star': None,
                'v_sys': None,
            }
        )

        self.data = data
        self.priors = priors


    def log_likelihood(self):
        """ The log likelihood given a set of DM parameters.
        For each star the log likelihood is defined as:
        .. math::
        logL = -0.5 * (v - v_mean)^2 / (sigma2_p + v_err^2) - 0.5 * log(2 pi  * (sigma2_p + verr^2))

        where:
        - v is the velocity of the star
        - v_mean is the mean velocity of all stars
        - v_err is the measurement error
        - sigma2_p is the velocity dispersion
        """
        # get all parameters and data
        gamma = self.parameters['gamma']
        r_dm = self.parameters['r_dm']
        rho_dm = self.parameters['rho_dm']
        beta_0 = self.parameters['beta_ani_0']
        r_ani = self.parameters['r_ani']
        r_star = self.parameters['r_star']
        v_sys = self.parameters['v_sys']

        rad2d = self.data['rad2d']
        vel_los = self.data['vel_los']
        vel_los_err = self.data['vel_los_err']


        # determine the integration range and points based on the parameters
        rint_min = max(0.01 * r_star, 1e-4)      # 1% of Plummer scale
        rint_max = max(20 * r_star, 10 * max(rad2d), 5 * r_dm)
        rint_max = min(rint_max, 100)  # kpc
        nint = int(2000 * np.log10(rint_max/rint_min)) # 2000 points/decade
        nint = np.clip(nint, 1000, 15000)
        rint = np.logspace(np.log10(rint_min), np.log10(rint_max), nint)

        # calculate the velocity dispersion
        sigma2p = calc_sigma2_los(
            np.log(rad2d),
            gamma,
            np.log(r_dm),
            np.log(rho_dm),
            beta_0,
            np.log(r_ani),
            np.log(r_star),
            rint=rint
        )

        # calculate the log likelihood from the velocity dispersion
        # and the velocity measurement error
        ll = np.sum(log_gaussian(vel_los, v_sys, sigma2p + vel_los_err**2))

        return ll

    def run_sampler(self, *args, **kargs):
        self.result = bilby.run_sampler(
            likelihood=self, priors=self.priors, *args, **kargs)
