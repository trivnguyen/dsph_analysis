"""
Kinematic data I/O utilities for dwarf spheroidal Jeans analysis.

This module provides data loading and processing functions for kinematic
catalogs from different observational sources (DESI, Walker+23, etc.).
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import astropy.table as at
import astropy.units as auni
from astropy.units.quantity import Quantity

from . import data_utils

DEFAULT_META_URL = (
    "https://raw.githubusercontent.com/apace7/local_volume_database/refs/heads/"
    "main/data/dwarf_mw.csv"
)
ALL_LOADERS = ('desi', 'walker23', 'bootes1_ting', 'deimos', 'mock_cartesian', 'mock_icrs')

@dataclass
class KinematicData:
    """Container for kinematic data of member stars."""
    ra: Quantity  # deg
    dec: Quantity  # deg
    vlos: Quantity  # km/s
    vlos_err: Quantity  # km/s
    X_proj: Quantity  # kpc
    Y_proj: Quantity  # kpc
    R_proj: Quantity  # kpc
    mem_prob: Quantity  # dimensionless
    vlos_raw: Optional[Quantity] = None  # km/s
    source: Optional[str] = None


@dataclass
class DwarfMeta:
    """Container for dwarf galaxy metadata."""
    key: str
    ra: Quantity  # deg
    dec: Quantity  # deg
    distance: Quantity  # kpc
    pmra: Quantity  # mas/yr
    pmdec: Quantity  # mas/yr
    vlos_systemic: Quantity  # km/s
    rhalf_arcmin: Quantity  # arcmin
    rhalf_arcmin_em: Quantity  # arcmin
    rhalf_arcmin_ep: Quantity  # arcmin
    rhalf_kpc: Quantity  # kpc
    rhalf_kpc_em: Quantity  # kpc
    rhalf_kpc_ep: Quantity  # kpc
    log_mass_wolf: Optional[Quantity] = None # log10(M_sun)
    log_mass_wolf_em: Optional[Quantity] = None
    log_mass_wolf_ep: Optional[Quantity] = None

def load_meta_table(
    meta_path: str = DEFAULT_META_URL
):
    """
    Load the entire metadata table from CSV file or URL.

    Parameters
    ----------
    meta_path : str
        Path to the metadata CSV file or URL.
        Supports local files and URLs (e.g., GitHub raw links).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the metadata for all dwarf galaxies.
    """
    return pd.read_csv(meta_path)

def load_meta(
    target_key,
    meta_path: str = DEFAULT_META_URL
) -> DwarfMeta:
    """
    Load dwarf galaxy metadata from CSV file or URL.

    Parameters
    ----------
    meta_path : str
        Path to the metadata CSV file or URL.
        Supports local files and URLs (e.g., GitHub raw links).
    target_key : str
        Key identifying the target dwarf galaxy.

    Returns
    -------
    DwarfMeta
        Metadata container for the dwarf galaxy.
    """
    meta_df = pd.read_csv(meta_path)
    # check if target_key exists in meta_df
    if target_key not in meta_df['key'].values:
        raise ValueError(f"target_key '{target_key}' not found in metadata table.")
    row = meta_df[meta_df['key'] == target_key].iloc[0]

    return DwarfMeta(
        key=target_key,
        ra=row.ra *  auni.deg,
        dec=row.dec * auni.deg,
        distance=row.distance * auni.kpc,
        pmra=row.pmra * auni.mas / auni.yr,
        pmdec=row.pmdec * auni.mas / auni.yr,
        vlos_systemic=row.vlos_systemic * auni.km / auni.s,
        rhalf_arcmin=row.rhalf * auni.arcmin,
        rhalf_arcmin_em=row.rhalf_em * auni.arcmin,
        rhalf_arcmin_ep=row.rhalf_ep * auni.arcmin,
        rhalf_kpc=row.rhalf_physical / 1000 * auni.kpc,
        rhalf_kpc_em=row.rhalf_physical_em / 1000 * auni.kpc,
        rhalf_kpc_ep=row.rhalf_physical_ep / 1000 * auni.kpc,
        log_mass_wolf=row.get('mass_dynamical_wolf', np.nan),  # always log10
        log_mass_wolf_em=row.get('mass_dynamical_wolf_em', np.nan),
        log_mass_wolf_ep=row.get('mass_dynamical_wolf_ep', np.nan),
    )


# =============================================================================
# Source-specific loaders
# =============================================================================

def _load_desi(
    catalog_path: str,
    meta: DwarfMeta,
    mem_prob_min: float = 0.8,
    vlos_abs_max: Optional[float] = None,
    vlos_err_floor: float = 0.9,
    apply_perspective_corr: bool = True,
) -> KinematicData:
    """Load kinematic data from DESI catalog."""
    data = pd.read_csv(catalog_path)
    data_cut = data[data['prob'] > mem_prob_min]

    ra = data_cut['RA'].values
    dec = data_cut['DEC'].values
    vlos_raw = data_cut['VRAD'].values
    vlos_err = data_cut['VRAD_ERR'].values
    vlos_err = np.sqrt(vlos_err**2 + vlos_err_floor**2)  # add error floor in quadrature
    mem_prob = data_cut['prob'].values

    (ra, dec, vlos_raw, vlos_err, mem_prob, vlos,
     X_proj, Y_proj, R_proj, mask) = (
        data_utils.preprocess_kinematic_data(
            ra, dec, vlos_raw, vlos_err, mem_prob, meta,
            vlos_abs_max=vlos_abs_max,
            apply_perspective_corr=apply_perspective_corr,
        )
    )

    return KinematicData(
        ra=ra * auni.deg,
        dec=dec * auni.deg,
        vlos=vlos * auni.km / auni.s,
        vlos_err=vlos_err * auni.km / auni.s,
        vlos_raw=vlos_raw * auni.km / auni.s,
        X_proj=X_proj * auni.kpc,
        Y_proj=Y_proj * auni.kpc,
        R_proj=R_proj * auni.kpc,
        mem_prob=mem_prob,
        source='desi',
    )


def _load_walker23(
    catalog_path: str,
    meta: DwarfMeta,
    mem_prob_min: float = 0.8,
    target_system: str = '',
    vlos_abs_max: Optional[float] = None,
    apply_perspective_corr: bool = True,
) -> KinematicData:
    """Load kinematic data from Walker+23 catalog."""
    data = pd.read_csv(catalog_path)
    select = (data['target_system'] == target_system) & (data['prob'] > mem_prob_min)
    if np.sum(select) == 0:
        raise ValueError(f"No stars selected for target_system={target_system} with mem_prob_min={mem_prob_min}")
    data_cut = data[select]

    ra = data_cut['ra'].values
    dec = data_cut['dec'].values
    vlos_raw = data_cut['vlos_mean'].values
    vlos_err = data_cut['vlos_mean_error'].values
    mem_prob = data_cut['prob'].values

    (ra, dec, vlos_raw, vlos_err, mem_prob, vlos,
     X_proj, Y_proj, R_proj, mask) = (
        data_utils.preprocess_kinematic_data(
            ra, dec, vlos_raw, vlos_err, mem_prob, meta,
            vlos_abs_max=vlos_abs_max,
            apply_perspective_corr=apply_perspective_corr,
        )
    )

    return KinematicData(
        ra=ra * auni.deg,
        dec=dec * auni.deg,
        vlos=vlos * auni.km / auni.s,
        vlos_err=vlos_err * auni.km / auni.s,
        vlos_raw=vlos_raw * auni.km / auni.s,
        X_proj=X_proj * auni.kpc,
        Y_proj=Y_proj * auni.kpc,
        R_proj=R_proj * auni.kpc,
        mem_prob=mem_prob,
        source='walker23',
    )

def _load_bootes1_ting(
    catalog_path: str,
    meta: DwarfMeta,
    mem_prob_min: float = 0.8,
    instrument='mmt',
    vlos_abs_max: Optional[float] = None,
    apply_perspective_corr: bool = True,
    use_sandford_perspective_corr: bool = False,
    remove_binaries: bool = True,
) -> KinematicData:
    """Load kinematic data from Boo I Ting catalog."""
    if instrument not in ['mmt', 'vlt', 's5', 'aat', 'avg']:
        raise ValueError(f"Unknown instrument: {instrument}")

    data = at.Table.read(catalog_path, format='ascii.ecsv').to_pandas()

    if instrument == 'avg':
        data_cut = data[data['member']]
        mem_prob = np.ones(len(data_cut))
    else:
        mem_prob_key = 'mem_p_' + instrument
        data_cut = data[data[mem_prob_key] > mem_prob_min]
        mem_prob = data_cut[mem_prob_key].values

    ra = data_cut['RA'].values
    dec = data_cut['Dec'].values
    vlos_raw = data_cut['vel_' + instrument].values
    vlos_err = data_cut['vel_err_' + instrument].values

    if remove_binaries:
        if instrument != 'avg':
            extra_mask = data_cut['vel_q_' + instrument].values.astype(bool)
        else:
            extra_mask = ~data_cut['binary'].values.astype(bool)
    else:
        extra_mask = None

    if not use_sandford_perspective_corr:
        (ra, dec, vlos_raw, vlos_err, mem_prob, vlos,
         X_proj, Y_proj, R_proj, mask) = (
            data_utils.preprocess_kinematic_data(
                ra, dec, vlos_raw, vlos_err, mem_prob, meta,
                vlos_abs_max=vlos_abs_max,
                apply_perspective_corr=apply_perspective_corr,
                extra_mask=extra_mask,
            )
        )
    else:
        (ra, dec, vlos_raw, vlos_err, mem_prob, vlos,
         X_proj, Y_proj, R_proj, mask) = (
            data_utils.preprocess_kinematic_data(
                ra, dec, vlos_raw, vlos_err, mem_prob, meta,
                vlos_abs_max=vlos_abs_max,
                apply_perspective_corr=False,
                extra_mask=extra_mask,
            )
        )
        vcorr = data_cut['vel_persp_rot'].values
        if extra_mask is not None:
            vcorr = vcorr[mask]
        vlos -= vcorr

    return KinematicData(
        ra=ra * auni.deg,
        dec=dec * auni.deg,
        vlos=vlos * auni.km / auni.s,
        vlos_err=vlos_err * auni.km / auni.s,
        vlos_raw=vlos_raw * auni.km / auni.s,
        X_proj=X_proj * auni.kpc,
        Y_proj=Y_proj * auni.kpc,
        R_proj=R_proj * auni.kpc,
        mem_prob=mem_prob,
        source='bootes1_ting_' + instrument
    )

def _load_deimos(
    catalog_path: str,
    meta: DwarfMeta,
    mem_prob_min: float = 0.8,
    target_system: str = '',
    vlos_abs_max: Optional[float] = None,
    apply_perspective_corr: bool = True,
) -> KinematicData:
    """Load kinematic data from DEIMOS catlaog."""
    data = pd.read_csv(catalog_path)
    select = (data['key'] == target_system) & (data['mem_prob'] > mem_prob_min)
    data_cut = data[select]

    ra = data_cut['RA'].values
    dec = data_cut['DEC'].values
    vlos_raw = data_cut['vr'].values
    vlos_err = data_cut['vr_err'].values
    mem_prob = data_cut['mem_prob'].values
    R_proj = data_cut['R_kin'].values

    (ra, dec, vlos_raw, vlos_err, mem_prob, vlos,
     X_proj, Y_proj, R_proj, mask) = (
        data_utils.preprocess_kinematic_data(
            ra, dec, vlos_raw, vlos_err, mem_prob, meta,
            vlos_abs_max=vlos_abs_max,
            apply_perspective_corr=apply_perspective_corr,
            R_proj_catalog=R_proj,
        )
    )

    return KinematicData(
        ra=ra * auni.deg,
        dec=dec * auni.deg,
        vlos=vlos * auni.km / auni.s,
        vlos_err=vlos_err * auni.km / auni.s,
        vlos_raw=vlos_raw * auni.km / auni.s,
        X_proj=X_proj * auni.kpc,
        Y_proj=Y_proj * auni.kpc,
        R_proj=R_proj * auni.kpc,
        mem_prob=mem_prob,
        source='deimos',
    )

def _load_mock_cartesian(
    catalog_path: str,
    meta: DwarfMeta,
    projection_axis: int = 0,
    num_max_stars: Optional[int] = None,
    seed: int = 42,
) -> KinematicData:
    """Load kinematic data from mock catalog."""
    data = pd.read_csv(catalog_path)
    if projection_axis == 0:
        X_proj = data['y_kpc'].values
        Y_proj = data['z_kpc'].values
        vlos = data['vx_kms'].values
        vlos_err = data['err_vx_kms'].values
        vlos_raw = data['vx_true_kms'].values
    elif projection_axis == 1:
        X_proj = data['x_kpc'].values
        Y_proj = data['z_kpc'].values
        vlos = data['vy_kms'].values
        vlos_err = data['err_vy_kms'].values
        vlos_raw = data['vy_true_kms'].values
    elif projection_axis == 2:
        X_proj = data['x_kpc'].values
        Y_proj = data['y_kpc'].values
        vlos = data['vz_kms'].values
        vlos_err = data['err_vz_kms'].values
        vlos_raw = data['vz_true_kms'].values
    else:
        raise ValueError(f"Invalid projection_axis: {projection_axis}")

    R_proj = np.sqrt(X_proj**2 + Y_proj**2)
    ra = X_proj
    dec = Y_proj

    rng = np.random.default_rng(seed)
    if num_max_stars is not None and len(data) > num_max_stars:
        selected_indices = rng.choice(len(data), size=num_max_stars, replace=False)
        X_proj = X_proj[selected_indices]
        Y_proj = Y_proj[selected_indices]
        R_proj = R_proj[selected_indices]
        vlos = vlos[selected_indices]
        vlos_err = vlos_err[selected_indices]
        ra = ra[selected_indices]
        dec = dec[selected_indices]

    return KinematicData(
        ra=ra * auni.deg,
        dec=dec * auni.deg,
        vlos=vlos * auni.km / auni.s,
        vlos_err=vlos_err * auni.km / auni.s,
        vlos_raw=vlos * auni.km / auni.s,
        X_proj=X_proj * auni.kpc,
        Y_proj=Y_proj * auni.kpc,
        R_proj=R_proj * auni.kpc,
        source='mock',
        mem_prob=np.ones(len(data))
    )

def _load_mock_icrs(
    catalog_path: str,
    meta: DwarfMeta,
    num_max_stars: Optional[int] = None,
    seed: int = 42,
    vlos_abs_max: Optional[float] = None,
    vlos_err_floor: float = 0.0,
    apply_perspective_corr: bool = True,
) -> KinematicData:
    """Load kinematic data from mock catalog in ICRS coordinates."""
    data = pd.read_csv(catalog_path)
    ra = data['ra'].values
    dec = data['dec'].values
    vlos_raw = data['vlos'].values
    vlos_err = data['vlos_err'].values
    mem_prob = np.ones(len(data))

    vlos_err = np.sqrt(vlos_err**2 + vlos_err_floor**2)

    rng = np.random.default_rng(seed)
    if num_max_stars is not None and len(data) > num_max_stars:
        selected_indices = rng.choice(len(data), size=num_max_stars, replace=False)
        ra = ra[selected_indices]
        dec = dec[selected_indices]
        vlos_raw = vlos_raw[selected_indices]
        vlos_err = vlos_err[selected_indices]
        mem_prob = mem_prob[selected_indices]

    (ra, dec, vlos_raw, vlos_err, mem_prob, vlos,
     X_proj, Y_proj, R_proj, _) = (
        data_utils.preprocess_kinematic_data(
            ra, dec, vlos_raw, vlos_err, mem_prob, meta,
            vlos_abs_max=vlos_abs_max,
            apply_perspective_corr=apply_perspective_corr,
        )
    )

    return KinematicData(
        ra=ra * auni.deg,
        dec=dec * auni.deg,
        vlos=vlos * auni.km / auni.s,
        vlos_err=vlos_err * auni.km / auni.s,
        vlos_raw=vlos_raw * auni.km / auni.s,
        X_proj=X_proj * auni.kpc,
        Y_proj=Y_proj * auni.kpc,
        R_proj=R_proj * auni.kpc,
        source='mock_icrs',
        mem_prob=mem_prob,
    )

def load_kinematic_data(
    catalog_path: str,
    meta: DwarfMeta,
    source: str,
    mem_prob_min: float = 0.8,
    **kwargs,
) -> KinematicData:
    """
    Load kinematic data from a catalog file.

    Parameters
    ----------
    catalog_path : str
        Path to the kinematic catalog CSV file.
    meta : DwarfMeta
        Metadata for the dwarf galaxy.
    source : str
        Source of the data. Available sources: 'desi', 'walker23'.
        New sources can be registered with `register_loader`.
    mem_prob_min : float, optional
        Minimum membership probability threshold. Default is 0.8.
    **kwargs
        Additional keyword arguments passed to the source-specific loader.

    Returns
    -------
    KinematicData
        Container with kinematic data for member stars.
    """
    if source not in ALL_LOADERS:
        raise ValueError(f"Unknown source: {source}. Available sources: {ALL_LOADERS}")
    LOADERS = {
        'desi': _load_desi,
        'walker23': _load_walker23,
        'bootes1_ting': _load_bootes1_ting,
        'deimos': _load_deimos,
        'mock_cartesian': _load_mock_cartesian,
        'mock_icrs': _load_mock_icrs,
    }
    loader = LOADERS[source]
    return loader(catalog_path, meta, **kwargs)
