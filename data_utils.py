"""Data processing utilities for dwarf spheroidal analysis."""

from typing import Tuple, Optional

import numpy as np
from numpy.typing import NDArray
import scipy.stats
import astropy.coordinates as acoo
import astropy.units as auni

from .coord_utils import rotation_matrix_from_vectors, cartesian_to_spherical

# Conversion factor from proper motion x distance to tangential velocity:
# v[km/s] = PM_TO_KMS * pm[mas/yr] * distance[kpc]
# (equivalently, 1 AU/yr = 4.74047 km/s)
PM_TO_KMS = 4.74047


def poisson_confidence_interval(
    n: NDArray[np.integer],
    alpha: float = 0.32
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute Poisson confidence interval for count data.

    Uses chi-squared distribution to compute confidence intervals.
    See PDG statistics review section 39.4.2.3:
    http://pdg.lbl.gov/2018/reviews/rpp2018-rev-statistics.pdf

    Parameters
    ----------
    n : NDArray[np.integer]
        Number of counts.
    alpha : float, optional
        Significance level. alpha=0.32 gives 68% confidence. Default is 0.32.

    Returns
    -------
    n_lo : NDArray[np.floating]
        Lower bound on counts.
    n_hi : NDArray[np.floating]
        Upper bound on counts.
    """
    n_lo = scipy.stats.chi2.ppf(alpha / 2, 2 * n) / 2
    n_hi = scipy.stats.chi2.ppf(1 - alpha / 2, 2 * (n + 1)) / 2
    return n_lo, n_hi


def calc_projected_nstar_binned(
    R: NDArray[np.floating],
    alpha: float = 0.32,
    return_bounds: bool = False,
    nbins: Optional[int] = None
) -> Tuple:
    """
    Calculate projected number of stars as a function of projected radius.

    Parameters
    ----------
    R : NDArray[np.floating]
        Projected radius of stars.
    alpha : float, optional
        Significance level for confidence bounds. Default is 0.32 (68% CI).
    return_bounds : bool, optional
        If True, return confidence bounds. Default is False.
    nbins : int | None, optional
        Number of bins. Default is sqrt(N).

    Returns
    -------
    n_data : NDArray[np.integer]
        Number of stars in each bin.
    logR_bins_lo : NDArray[np.floating]
        Log10 of lower edge of bins.
    logR_bins_hi : NDArray[np.floating]
        Log10 of upper edge of bins.
    n_data_lo : NDArray[np.floating]
        Lower bound (only if return_bounds=True).
    n_data_hi : NDArray[np.floating]
        Upper bound (only if return_bounds=True).
    """
    logR = np.log10(R)
    n_total = len(logR)

    # Bin the projected radius using log scale
    n_bins = int(np.ceil(np.sqrt(n_total))) if nbins is None else nbins
    logR_min = np.floor(np.min(logR) * 10) / 10
    logR_max = np.ceil(np.max(logR) * 10) / 10
    print(np.min(R))

    n_data, logR_bins = np.histogram(logR, n_bins, range=(logR_min, logR_max))

    # Remove bins with zero counts
    select = n_data > 0
    n_data = n_data[select]
    logR_bins_lo = logR_bins[:-1][select]
    logR_bins_hi = logR_bins[1:][select]

    if return_bounds:
        n_data_lo, n_data_hi = poisson_confidence_interval(n_data, alpha=alpha)
        return n_data, n_data_lo, n_data_hi, logR_bins_lo, logR_bins_hi
    return n_data, logR_bins_lo, logR_bins_hi


def calc_Sigma_star_binned(
    R: NDArray[np.floating],
    alpha: float = 0.32,
    return_bounds: bool = False,
    nbins: Optional[int] = None
) -> Tuple:
    """
    Calculate projected 2D surface density profile from star positions.

    Parameters
    ----------
    R : NDArray[np.floating]
        Projected radius of stars.
    alpha : float, optional
        Significance level for confidence bounds. Default is 0.32.
    return_bounds : bool, optional
        If True, return confidence bounds. Default is False.
    nbins : int | None, optional
        Number of bins. Default is sqrt(N).

    Returns
    -------
    Sigma_data : NDArray[np.floating]
        Surface density in each bin.
    logR_bins_lo : NDArray[np.floating]
        Log10 of lower edge of bins.
    logR_bins_hi : NDArray[np.floating]
        Log10 of upper edge of bins.
    Sigma_data_lo : NDArray[np.floating]
        Lower bound (only if return_bounds=True).
    Sigma_data_hi : NDArray[np.floating]
        Upper bound (only if return_bounds=True).
    """
    data = calc_projected_nstar_binned(R, alpha=alpha, return_bounds=return_bounds, nbins=nbins)
    if return_bounds:
        n_data, n_data_lo, n_data_hi, logR_bins_lo, logR_bins_hi = data
    else:
        n_data, logR_bins_lo, logR_bins_hi = data

    # Calculate surface density from projected number of stars
    R_bins_lo = 10**logR_bins_lo
    R_bins_hi = 10**logR_bins_hi
    delta_R2 = R_bins_hi**2 - R_bins_lo**2
    Sigma_data = n_data / (np.pi * delta_R2)

    if return_bounds:
        Sigma_data_lo = n_data_lo / (np.pi * delta_R2)
        Sigma_data_hi = n_data_hi / (np.pi * delta_R2)
        return Sigma_data, Sigma_data_lo, Sigma_data_hi, logR_bins_lo, logR_bins_hi
    return Sigma_data, logR_bins_lo, logR_bins_hi


def calc_rho_binned(
    r: NDArray[np.floating],
    mass: NDArray[np.floating],
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    num_bins: Optional[int] = None,
    return_count: bool = False
) -> Tuple:
    """
    Calculate 3D density profile from particle data.

    Parameters
    ----------
    r : NDArray[np.floating]
        Radii of particles from galaxy center.
    mass : NDArray[np.floating]
        Mass of each particle.
    r_min : float | None, optional
        Minimum bin radius. Default is min(r).
    r_max : float | None, optional
        Maximum bin radius. Default is max(r).
    num_bins : int | None, optional
        Number of bins. Default is sqrt(N).
    return_count : bool, optional
        If True, also return particle counts per bin. Default is False.

    Returns
    -------
    rho : NDArray[np.floating]
        Density profile.
    sigma_rho : NDArray[np.floating]
        Uncertainty on density.
    log_rbins : NDArray[np.floating]
        Log10 of radial bin edges.
    count_bins : NDArray[np.integer]
        Particle counts (only if return_count=True).
    """
    logr = np.log10(r)
    num_bins = int(np.sqrt(len(r))) if num_bins is None else num_bins
    log_rmin = np.floor(np.min(logr) * 10) / 10 if r_min is None else np.log10(r_min)
    log_rmax = np.floor(np.max(logr) * 10) / 10 if r_max is None else np.log10(r_max)

    # Histogram mass profile
    count_bins, log_rbins = np.histogram(logr, num_bins, range=(log_rmin, log_rmax))
    mass_bins, log_rbins = np.histogram(logr, num_bins, range=(log_rmin, log_rmax), weights=mass)
    r_bins = np.power(10, log_rbins)

    # Calculate density profile
    dV = 4 * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3) / 3
    rho = mass_bins / dV
    sigma_rho = rho / np.sqrt(count_bins)

    if return_count:
        return rho, sigma_rho, log_rbins, count_bins
    return rho, sigma_rho, log_rbins


def calc_mass_enclosed_binned(
    r: NDArray[np.floating],
    mass: NDArray[np.floating],
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    num_bins: Optional[int] = None
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Calculate enclosed mass profile from particle data.

    Parameters
    ----------
    r : NDArray[np.floating]
        Radii of particles from galaxy center.
    mass : NDArray[np.floating]
        Mass of each particle.
    r_min : float | None, optional
        Minimum bin radius. Default is min(r).
    r_max : float | None, optional
        Maximum bin radius. Default is max(r).
    num_bins : int | None, optional
        Number of bins. Default is sqrt(N).

    Returns
    -------
    mass_enc : NDArray[np.floating]
        Enclosed mass at each radius.
    log_rbins : NDArray[np.floating]
        Log10 of radial bin edges.
    """
    logr = np.log10(r)
    num_bins = int(np.sqrt(len(r))) if num_bins is None else num_bins
    log_rmin = np.floor(np.min(logr) * 10) / 10 if r_min is None else np.log10(r_min)
    log_rmax = np.floor(np.max(logr) * 10) / 10 if r_max is None else np.log10(r_max)

    # Histogram mass profile
    mass_bins, log_rbins = np.histogram(logr, num_bins, range=(log_rmin, log_rmax), weights=mass)
    mass_enc = np.cumsum(mass_bins)

    return mass_enc, log_rbins


def calc_sigma_spherical(
    pos: NDArray[np.floating],
    vel: NDArray[np.floating],
    mass: NDArray[np.floating],
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    num_bins: Optional[int] = None
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Calculate velocity dispersion in spherical coordinates from particle data.

    Accounts for bulk rotation by aligning with angular momentum axis.

    Parameters
    ----------
    pos : NDArray[np.floating]
        Position of particles relative to galaxy center, shape (N, 3).
    vel : NDArray[np.floating]
        Velocity of particles in galaxy frame, shape (N, 3).
    mass : NDArray[np.floating]
        Mass of each particle.
    r_min : float | None, optional
        Minimum bin radius. Default is min(r).
    r_max : float | None, optional
        Maximum bin radius. Default is max(r).
    num_bins : int | None, optional
        Number of bins. Default is sqrt(N).

    Returns
    -------
    sigma : NDArray[np.floating]
        Velocity dispersion (sigma_r, sigma_theta, sigma_phi) for each bin.
    log_rbins : NDArray[np.floating]
        Log10 of radial bin edges.
    """
    radius = np.linalg.norm(pos, axis=1)
    logr = np.log10(radius)

    num_bins = int(np.sqrt(len(logr))) if num_bins is None else num_bins
    log_rmin = np.min(logr) if r_min is None else np.log10(r_min)
    log_rmax = np.max(logr) if r_max is None else np.log10(r_max)
    log_rbins = np.linspace(log_rmin, log_rmax, num_bins + 1)
    rbins = np.power(10, log_rbins)

    sigma_r, sigma_th, sigma_ph = [], [], []
    for i in range(num_bins):
        # Select spherical shell (with some overlap for smoothing)
        i_lo = max(0, i - 3)
        i_hi = min(num_bins, i + 3)
        r_lo = rbins[i_lo]
        r_hi = rbins[i_hi]

        select = (r_lo <= radius) & (radius < r_hi)

        if select.sum() == 0:
            sigma_r.append(np.nan)
            sigma_th.append(np.nan)
            sigma_ph.append(np.nan)
            continue

        pos_bins = pos[select]
        vel_bins = vel[select]
        mass_bins = mass[select]

        # Calculate angular momentum and rotational velocity of shell
        J = np.sum(mass_bins[:, np.newaxis] * np.cross(pos_bins, vel_bins), axis=0)
        I = 2.0 / 5.0 * mass_bins.sum() * (r_hi**5 - r_lo**5) / (r_hi**3 - r_lo**3)
        omega = np.linalg.norm(J / I)

        # Rotate coordinate so that J is aligned with the z-axis
        rot_matrix = rotation_matrix_from_vectors(J, np.array([0, 0, 1]))
        pos_rot = pos_bins @ rot_matrix.T
        vel_rot = vel_bins @ rot_matrix.T

        # Convert cartesian to spherical coordinate system
        r, th, ph, vr, vth, vph = cartesian_to_spherical(
            pos_rot * u.kpc, vel_rot * u.km / u.s
        )
        r = r.to_value(u.kpc)
        th = th.to_value(u.rad)
        vr = vr.to_value(u.km / u.s)
        vth = vth.to_value(u.km / u.s)
        vph = vph.to_value(u.km / u.s)
        vph = vph - omega * r * np.sin(th)  # Subtract bulk rotation

        # Store dispersions
        sigma_r.append(np.std(vr))
        sigma_th.append(np.std(vth))
        sigma_ph.append(np.std(vph))

    sigma = np.stack([sigma_r, sigma_th, sigma_ph], axis=1)
    return sigma, log_rbins


def calc_systemic_velocity(
    vr: NDArray[np.floating],
    vr_err: NDArray[np.floating]
) -> float:
    """
    Calculate the systemic velocity of the dwarf using weighted average.

    Parameters
    ----------
    vr : NDArray[np.floating]
        Radial velocities of member stars in km/s.
    vr_err : NDArray[np.floating]
        Uncertainties on radial velocities in km/s.

    Returns
    -------
    float
        Weighted mean systemic velocity in km/s.
    """
    weights = 1.0 / vr_err**2
    return np.average(vr, weights=weights)


def calc_perspective_rotation_corr(
    ra: NDArray[np.floating],
    dec: NDArray[np.floating],
    ra_center: float,
    dec_center: float,
    dist_center: float,
    pmra_cosdec_center: float,
    pmdec_center: float,
    vrad_center: float,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Calculate the correction to the expected proper motion and radial velocity
    due to perspective rotation effect.

    Credit: Ting Li (University of Toronto)

    Arguments
    ---------
    ra, dec: array-like
        Right ascension and declination of stars in degrees.
    ra_center, dec_center: float
        Right ascension and declination of the galaxy center in degrees.
    dist_center: float
        Distance to the galaxy center in kpc.
    pmra_cosdec_center, pmdec_center: float
        Proper motion of the galaxy center in mas/yr.
    vrad_center: float
        Radial velocity of the galaxy center in km/s.

    Returns
    -------
    dpmra_cosdec, dpmdec, dvrad: array-like
        Corrections to the proper motion in mas/yr and radial velocity in km/s
        for each star due to perspective rotation effect.
    """
    kms, masyr = auni.km / auni.s, auni.mas / auni.yr
    C0 = acoo.SkyCoord(ra=ra_center * auni.deg,
                       dec=dec_center * auni.deg,
                       distance=dist_center * auni.kpc,
                       radial_velocity=vrad_center * kms,
                       pm_ra_cosdec=pmra_cosdec_center * masyr,
                       pm_dec=pmdec_center * masyr)
    Cg0 = C0.transform_to(acoo.Galactocentric)
    # center of the system
    N = len(ra)
    Cs1 = acoo.SkyCoord(ra=ra * auni.deg,
                        dec=dec * auni.deg,
                        distance=dist_center * auni.kpc)
    Cg1 = Cs1.transform_to(acoo.Galactocentric)
    # just positions of stars assuming they are at the same distance
    Cg2 = acoo.Galactocentric(x=Cg1.x,
                              y=Cg1.y,
                              z=Cg1.z,
                              v_x=np.ones(N) * Cg0.v_x,
                              v_y=np.ones(N) * Cg0.v_y,
                              v_z=np.ones(N) * Cg0.v_z)
    # set their velocity to velocity of the center.
    Cs2 = Cg2.transform_to(acoo.ICRS())
    # go back to helio frame
    return (
        Cs2.pm_ra_cosdec.to_value(masyr),
        Cs2.pm_dec.to_value(masyr),
        Cs2.radial_velocity.to_value(kms)
    )


def calc_projected_xy(
    ra: NDArray[np.floating],
    dec: NDArray[np.floating],
    ra_center: float,
    dec_center: float,
    dist_center: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Calculate the projected tangent-plane coordinates from the galaxy
    center.

    Uses astropy's sky-offset frame for an exact tangent-plane
    projection (rather than a small-angle approximation), with
    astropy units handling the angle-to-physical-distance conversion.
    The projected radius can be recovered as
    ``sqrt(x_proj**2 + y_proj**2)``.

    Parameters
    ----------
    ra : NDArray[np.floating]
        Right ascension of stars in degrees.
    dec : NDArray[np.floating]
        Declination of stars in degrees.
    ra_center : float
        Right ascension of galaxy center in degrees.
    dec_center : float
        Declination of galaxy center in degrees.
    dist_center : float
        Distance to the galaxy in kpc.

    Returns
    -------
    x_proj : NDArray[np.floating]
        Projected coordinate along the RA direction in kpc.
    y_proj : NDArray[np.floating]
        Projected coordinate along the Dec direction in kpc.
    """
    center = acoo.SkyCoord(ra=ra_center * auni.deg, dec=dec_center * auni.deg)
    offset = acoo.SkyCoord(ra=ra * auni.deg, dec=dec * auni.deg).transform_to(
        center.skyoffset_frame()
    )

    dist_center = dist_center * auni.kpc
    angle_to_length = auni.dimensionless_angles()
    x_proj = (dist_center * offset.lon.wrap_at(180 * auni.deg)).to_value(
        auni.kpc, equivalencies=angle_to_length)
    y_proj = (dist_center * offset.lat).to_value(
        auni.kpc, equivalencies=angle_to_length)
    return x_proj, y_proj


def preprocess_kinematic_data(
    ra, dec, vlos_raw, vlos_err, mem_prob,
    meta,
    vlos_abs_max=None,
    apply_perspective_corr=True,
    extra_mask=None,
    R_proj_catalog=None,
    pmra_cosdec=None,
    pmdec=None,
    pmra_cosdec_err=None,
    pmdec_err=None,
):
    """
    Apply common preprocessing to raw kinematic data arrays.

    Applies NaN masking, optional velocity cut, perspective (or systemic)
    velocity correction, and computes projected tangent-plane coordinates
    and radius. If proper motions are given, also perspective- (or
    systemic-) corrects them and converts them to tangential velocities
    vX, vY in km/s, consistent with the vlos correction.

    Parameters
    ----------
    ra, dec : ndarray
        Sky coordinates in degrees.
    vlos_raw, vlos_err : ndarray
        Raw line-of-sight velocities and errors in km/s.
    mem_prob : ndarray
        Membership probabilities.
    meta : DwarfMeta
        Dwarf galaxy metadata.
    vlos_abs_max : float, optional
        Maximum |vlos - v_sys| in km/s. Stars beyond this are removed.
    apply_perspective_corr : bool
        If True, apply full perspective rotation correction to vlos (and
        to pmra_cosdec/pmdec, if given); otherwise only subtract the systemic
        velocity (and systemic proper motion).
    extra_mask : ndarray of bool, optional
        Additional boolean mask ANDed with the NaN mask before cuts.
    R_proj_catalog : ndarray, optional
        Pre-computed projected radius from the catalog, in kpc. If given,
        it is masked and used in place of the radius derived from
        ``x_proj``/``y_proj``.
    pmra_cosdec, pmdec : ndarray, optional
        Raw proper motions in mas/yr (pmra_cosdec assumed to already include the
        cos(dec) factor, i.e. pmra_cosdec = mu_alpha* = mu_alpha * cos(dec)).
        If either is None, no proper-motion-derived quantities are
        computed and the corresponding return values are all None.
        Individual stars may have NaN vlos and/or NaN pmra_cosdec/pmdec (mixed-
        completeness catalogs where not every star has both vlos and PM
        measured) -- a star is kept as long as it has at least one of
        the two; the missing quantity comes back as NaN rather than the
        star being dropped.
    pmra_cosdec_err, pmdec_err : ndarray, optional
        Uncertainties on pmra_cosdec/pmdec in mas/yr. Only used to propagate
        into vX_err/vY_err; either can be omitted independently.

    Returns
    -------
    ra, dec, vlos_raw, vlos_err, mem_prob, vlos, x_proj, y_proj, R_proj,
    mask : ndarray
        Masked and processed arrays. All velocities in km/s, x_proj,
        y_proj, and R_proj in kpc.
    pmra_cosdec, pmdec, pmra_cosdec_err, pmdec_err : ndarray or None
        Masked, uncorrected proper motions and their errors in mas/yr
        (None if not provided).
    vX, vY : ndarray or None
        Perspective- (or systemic-) corrected tangential velocities from
        pmra_cosdec/pmdec, in km/s (None if pmra_cosdec/pmdec not provided).
    vX_err, vY_err : ndarray or None
        Propagated uncertainties on vX/vY in km/s (None if the
        corresponding pmra_cosdec_err/pmdec_err was not provided).
    """
    has_pm = pmra_cosdec is not None and pmdec is not None

    # A star is kept if it has a valid vlos OR a valid PM measurement --
    # mixed-completeness catalogs (e.g. some stars vlos-only, others
    # PM-only) are expected, so this must be an OR, not an AND. Whichever
    # quantity a given star lacks comes back as NaN, not dropped.
    has_vlos = ~np.isnan(vlos_raw) & ~np.isnan(vlos_err)
    if has_pm:
        has_valid_pm = ~np.isnan(pmra_cosdec) & ~np.isnan(pmdec)
        if pmra_cosdec_err is not None:
            has_valid_pm &= ~np.isnan(pmra_cosdec_err)
        if pmdec_err is not None:
            has_valid_pm &= ~np.isnan(pmdec_err)
        mask = has_vlos | has_valid_pm
    else:
        mask = has_vlos
    if extra_mask is not None:
        mask &= extra_mask

    # apply vlos_abs_max cut -- only to stars that actually have a vlos,
    # so PM-only stars aren't dropped by a cut that doesn't apply to them
    # TODO: in the future, we may want to apply this cut after perspective correction instead of on the raw vlos
    # for now, keep it here so that the cut is consistent with previous analysis
    if vlos_abs_max is not None:
        vlos_raw_nosys = vlos_raw - meta.vlos_systemic.to_value(auni.km / auni.s)
        within_vlos_cut = np.abs(vlos_raw_nosys) < vlos_abs_max
        mask &= within_vlos_cut | ~has_vlos

    ra = ra[mask]
    dec = dec[mask]
    vlos_raw = vlos_raw[mask]
    vlos_err = vlos_err[mask]
    mem_prob = mem_prob[mask]
    if R_proj_catalog is not None:
        R_proj_catalog = R_proj_catalog[mask]
    if has_pm:
        pmra_cosdec = pmra_cosdec[mask]
        pmdec = pmdec[mask]
        if pmra_cosdec_err is not None:
            pmra_cosdec_err = pmra_cosdec_err[mask]
        if pmdec_err is not None:
            pmdec_err = pmdec_err[mask]

    if apply_perspective_corr:
        dpmra_cosdec_corr, dpmdec_corr, dvr_corr = calc_perspective_rotation_corr(
            ra, dec,
            meta.ra.to_value(auni.deg),
            meta.dec.to_value(auni.deg),
            meta.distance.to_value(auni.kpc),
            meta.pmra_cosdec.to_value(auni.mas / auni.yr),
            meta.pmdec.to_value(auni.mas / auni.yr),
            meta.vlos_systemic.to_value(auni.km / auni.s),
        )
        vlos = vlos_raw - dvr_corr
    else:
        dpmra_cosdec_corr = np.full_like(ra, meta.pmra_cosdec.to_value(auni.mas / auni.yr))
        dpmdec_corr = np.full_like(ra, meta.pmdec.to_value(auni.mas / auni.yr))
        vlos = vlos_raw - meta.vlos_systemic.to_value(auni.km / auni.s)

    x_proj, y_proj = calc_projected_xy(
        ra, dec,
        meta.ra.to_value(auni.deg),
        meta.dec.to_value(auni.deg),
        meta.distance.to_value(auni.kpc),
    )
    if R_proj_catalog is not None:
        R_proj = R_proj_catalog
    else:
        R_proj = np.sqrt(x_proj**2 + y_proj**2)

    vX = vY = vX_err = vY_err = None
    if has_pm:
        dist_kpc = meta.distance.to_value(auni.kpc)
        vX = PM_TO_KMS * (pmra_cosdec - dpmra_cosdec_corr) * dist_kpc
        vY = PM_TO_KMS * (pmdec - dpmdec_corr) * dist_kpc
        if pmra_cosdec_err is not None:
            vX_err = PM_TO_KMS * pmra_cosdec_err * dist_kpc
        if pmdec_err is not None:
            vY_err = PM_TO_KMS * pmdec_err * dist_kpc

    return (
        ra, dec, vlos_raw, vlos_err, mem_prob, vlos,
        x_proj, y_proj, R_proj, mask,
        pmra_cosdec, pmdec, pmra_cosdec_err, pmdec_err, vX, vY, vX_err, vY_err,
    )
