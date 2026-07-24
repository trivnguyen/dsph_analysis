"""
Backwards compatibility layer for dsph_analysis utilities.

This module re-exports functions from their new locations for backwards compatibility.
New code should import directly from the specific modules:

- profiles: Analytical density and anisotropy profiles
- jeans.solver: Jeans equation calculations
- data_utils: Data processing and binning utilities
- coord_utils: Coordinate transformations
- vdisp: Velocity dispersion analysis and MCMC fitting
"""

import warnings

# Re-export from profiles
from .profiles import (
    ln_Plummer2d,
    ln_Plummer3d,
    ln_rho_gnfw,
    ln_mass_gnfw,
    ln_rhobar_gnfw,
    beta_osipkov_merritt,
    ln_g_osipkov_merritt,
)

# # Re-export from jeans.solver
# from .jeans.solver import (
#     calc_ln_sigma2_nu,
#     calc_ln_sigma2p_Sigma,
#     calc_sigma2_los,
#     calc_vsp1,
#     calc_vsp2,
# )

# Re-export from data_utils
from .data_utils import (
    poisson_confidence_interval,
    calc_projected_nstar_binned,
    calc_Sigma_star_binned,
    calc_rho_binned,
    calc_mass_enclosed_binned,
    calc_sigma_spherical,
    calc_systemic_velocity,
    calc_perspective_rotation_corr,
    calc_projected_xy
)

# Re-export from coord_utils
from .coord_utils import (
    rotation_matrix_from_vectors,
    cartesian_to_spherical,
)

__all__ = [
    # Profiles
    'ln_Plummer2d',
    'ln_Plummer3d',
    'ln_rho_gnfw',
    'ln_mass_gnfw',
    'ln_rhobar_gnfw',
    'beta_osipkov_merritt',
    'ln_g_osipkov_merritt',
    # Jeans solver
    'calc_ln_sigma2_nu',
    'calc_ln_sigma2p_Sigma',
    'calc_sigma2_los',
    'calc_vsp1',
    'calc_vsp2',
    # Data utils
    'poisson_confidence_interval',
    'calc_projected_nstar',
    'calc_Sigma_los_data',
    'calc_rho_data',
    'calc_mass_enclosed',
    'calc_velsig_los_data',
    'calc_sigma_spherical',
    'calc_systemic_velocity',
    'calc_perspective_rotation_corr',
    'calc_projected_xy',
    # Coord utils
    'rotation_matrix_from_vectors',
    'cartesian_to_spherical',
]
