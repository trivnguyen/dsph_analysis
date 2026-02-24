import numpy as np
import bilby

from ..profiles import ln_Plummer2d
from ..data_utils import calc_Sigma_star_binned

class LightProfile(bilby.Likelihood):
    """ Fit the light profile model from kinematics data """
    def __init__(
        self,
        data: dict,
        priors: dict,
    ) -> None:
        """
        Parameters
        ----------
        data: dict
            Dictionary containing the kinematics data
        priors : dict
            The priors in bilby format
        """
        parameters = ['L', 'a']
        super().__init__(parameters={k: None for k in parameters})

        self.data = data
        self.priors = priors
        self._setup_likelihood()

    def _setup_likelihood(self):
        """ Setup before running the likelihood function by calculating
        the light profile and the variance of the light profile
        """
        Sigma, Sigma_lo, Sigma_hi, logRbins_lo, logRbins_hi = calc_Sigma_star_binned(
            self.data['rad2d'], alpha=0.32, return_bounds=True)
        Rbins_lo = 10**logRbins_lo
        Rbins_hi = 10**logRbins_hi
        sig_lo = Sigma - Sigma_lo
        sig_hi = Sigma_hi - Sigma
        V1 = sig_lo * sig_hi
        V2 = sig_hi - sig_lo

        self.Sigma = Sigma
        self.V1 = V1
        self.V2 = V2
        self.ln_R = np.log(0.5 * (Rbins_lo + Rbins_hi))


    def log_likelihood(self):
        """ Log likelihood function defined as:
        ```
            logL = -0.5 * (Sigma - Sigma_hat)^2 / (V1 - V2 * (Sigma - Sigma_hat))
        ```
        where:
        - Sigma is the light profile as inferred from data
        - Sigma_hat is the estimated light profile
        - V1 and V2
        """
        ln_L = np.log(self.parameters['L'])
        ln_a = np.log(self.parameters['a'])
        Sigma_hat = np.exp(ln_Plummer2d(self.ln_R, ln_L, ln_a))
        delta_Sigma = self.Sigma - Sigma_hat
        ll = -0.5 * np.sum(delta_Sigma**2 / (self.V1 - self.V2 * delta_Sigma))
        return ll

    def run_sampler(self, *args, **kargs):
        self.result = bilby.run_sampler(
            likelihood=self, priors=self.priors, *args, **kargs)
