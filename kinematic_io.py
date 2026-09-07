"""
Kinematic data I/O utilities for dwarf spheroidal Jeans analysis.

This module provides data loading and processing functions for kinematic
catalogs from different observational sources (DESI, Walker+23, etc.).
"""

import functools
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import astropy.table as at
import astropy.units as auni
from astropy.units.quantity import Quantity

from . import data_utils

# Upstream source, kept for reference and for refresh_local_meta_table().
DWARF_MW_URL = (
    "https://raw.githubusercontent.com/apace7/local_volume_database/refs/heads/"
    "main/data/dwarf_mw.csv"
)
# Local cache of DWARF_MW_URL, used by default so loading metadata does not
# depend on network access. Refresh with refresh_local_meta_table().
DEFAULT_META_PATH = Path(__file__).parent / "data" / "dwarf_mw.csv"
ALL_LOADERS = ('desi', 'walker23', 's5comp', 'deimos', 'pace',
               'mock_cartesian', 'mock_icrs')
# Populated at bottom of module, once every _load_* function is defined.
LOADERS: dict[str, Callable] = {}

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
    # Optional proper motion data. pmra_cosdec/pmdec are the perspective- (or
    # systemic-) corrected proper motions; vX/vY are the corresponding
    # tangential velocities. All None if the source has no PM data.
    pmra_cosdec: Optional[Quantity] = None  # mas/yr
    pmdec: Optional[Quantity] = None  # mas/yr
    pmra_cosdec_err: Optional[Quantity] = None  # mas/yr
    pmdec_err: Optional[Quantity] = None  # mas/yr
    vX: Optional[Quantity] = None  # km/s
    vY: Optional[Quantity] = None  # km/s
    vX_err: Optional[Quantity] = None  # km/s
    vY_err: Optional[Quantity] = None  # km/s
    source: Optional[str] = None

    def __len__(self):
        return len(self.ra)

    @property
    def has_pm(self) -> bool:
        """Whether this dataset includes proper motion measurements."""
        return self.pmra_cosdec is not None and self.pmdec is not None


@dataclass
class DwarfMeta:
    """Container for dwarf galaxy metadata."""
    key: str
    ra: Quantity  # deg
    dec: Quantity  # deg
    distance: Quantity  # kpc
    pmra_cosdec: Quantity  # mas/yr
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
    meta_path: str = DEFAULT_META_PATH
):
    """
    Load the entire metadata table from CSV file or URL.

    Defaults to the local cached copy (DEFAULT_META_PATH) so this works
    offline; pass DWARF_MW_URL (or any other path/URL) to read from
    elsewhere, or call refresh_local_meta_table() to update the cache.

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
    meta_path: str = DEFAULT_META_PATH
) -> DwarfMeta:
    """
    Load dwarf galaxy metadata from CSV file or URL.

    Defaults to the local cached copy (DEFAULT_META_PATH) so this works
    offline; pass DWARF_MW_URL (or any other path/URL) to read from
    elsewhere, or call refresh_local_meta_table() to update the cache.

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
        pmra_cosdec=row.pmra * auni.mas / auni.yr,
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
# Galaxy name abbreviations
# =============================================================================

# Compact galaxy tokens used in catalog file names: a three-letter stem
# plus the LVDB numeral, e.g. 'draco_1' -> 'dra1'. The numeral is never
# dropped, because 'draco_1' and 'draco_2' are different galaxies and a
# bare 'dra' would be ambiguous. Verified collision-free across every
# numbered key in dwarf_mw.csv (65 of them) by test_abbrev_unique().
# Local cache of the `abbreviation` field LVDB carries in its per-system
# YAML inputs (data_input/<key>.yaml, under `name_discovery`). It is NOT
# in the combined dwarf_mw.csv table, which is why it needs its own
# cache; refresh it with refresh_local_abbrev_table().
DEFAULT_ABBREV_PATH = Path(__file__).parent / "data" / "lvdb_abbrev.csv"

# Template for the per-system YAML inputs, used only by the refresh.
LVDB_YAML_URL = (
    "https://raw.githubusercontent.com/apace7/local_volume_database/main/"
    "data_input/{key}.yaml"
)


@functools.lru_cache(maxsize=None)
def _abbrev_tables(abbrev_path: str = DEFAULT_ABBREV_PATH) -> tuple:
    """
    Load the LVDB abbreviation table in both directions.

    Abbreviations are lower-cased here: LVDB writes them in mixed case
    ('BooI', 'HyaII', 'CVnI') but its own keys are all lower-case, and
    file names should be too. Lower-casing is checked not to introduce
    collisions.

    Args:
        abbrev_path: CSV with `key` and `abbreviation` columns.

    Returns:
        (key -> abbreviation, abbreviation -> key), both lower-cased.

    Raises:
        ValueError: If two keys share an abbreviation once lower-cased,
            which would make file names ambiguous.
    """
    table = pd.read_csv(abbrev_path)
    # Reason: a few LVDB abbreviations contain spaces ('Do V'); file
    # name tokens must be a single alphanumeric word.
    forward = {
        str(row.key): re.sub(r'\s+', '', str(row.abbreviation)).lower()
        for row in table.itertuples()
    }
    reverse: dict[str, list[str]] = {}
    for key, abbrev in forward.items():
        reverse.setdefault(abbrev, []).append(key)
    collisions = {a: k for a, k in reverse.items() if len(k) > 1}
    if collisions:
        raise ValueError(f"Ambiguous galaxy abbreviations: {collisions}")
    return forward, {a: k[0] for a, k in reverse.items()}


def key_to_abbrev(key: str, abbrev_path: str = DEFAULT_ABBREV_PATH) -> str:
    """
    Convert an LVDB target key to its LVDB abbreviation, lower-cased.

    This is LVDB's own `abbreviation` field, not a truncation of the
    name. That distinction matters: truncating to three letters collides
    Hydra with Hydrus and Leo with Leo Minor, whereas LVDB (following the
    IAU constellation abbreviations) distinguishes them as 'hyaii'/'hyii'
    and 'leoi'/'lmii'. Abbreviations are therefore not fixed-length -
    classical dwarfs are bare ('dra', 'scl', 'umi', 'for') and newer ones
    carry a Roman numeral ('booi', 'hyaii').

    Args:
        key: LVDB key, e.g. 'draco_1'.
        abbrev_path: CSV cache to resolve against.

    Returns:
        The lower-cased abbreviation, e.g. 'dra'.

    Raises:
        ValueError: If `key` is absent from the abbreviation table. LVDB
            populates `abbreviation` for MW dwarf galaxies only, so a
            key outside that set legitimately has none.

    Example:
        >>> key_to_abbrev('ursa_minor_1')
        'umi'
    """
    forward, _ = _abbrev_tables(abbrev_path)
    if key not in forward:
        raise ValueError(
            f"No LVDB abbreviation for key {key!r}. LVDB defines "
            f"`abbreviation` for MW dwarf galaxies only; refresh the "
            f"cache with refresh_local_abbrev_table() if it is new."
        )
    return forward[key]


def abbrev_to_key(abbrev: str, abbrev_path: str = DEFAULT_ABBREV_PATH) -> str:
    """
    Convert an LVDB abbreviation back to its target key.

    Args:
        abbrev: Abbreviation, case-insensitive, e.g. 'dra' or 'Dra'.
        abbrev_path: CSV cache to resolve against.

    Returns:
        The LVDB key, e.g. 'draco_1'.

    Raises:
        ValueError: If `abbrev` matches no key in the table.

    Example:
        >>> abbrev_to_key('umi')
        'ursa_minor_1'
    """
    _, reverse = _abbrev_tables(abbrev_path)
    token = abbrev.lower()
    if token not in reverse:
        raise ValueError(
            f"Unknown galaxy abbreviation {abbrev!r}. "
            f"Known: {sorted(reverse)}"
        )
    return reverse[token]


def refresh_local_abbrev_table(
    keys: Optional[list] = None,
    meta_path: str = DEFAULT_META_PATH,
    local_path: str = DEFAULT_ABBREV_PATH,
) -> None:
    """
    Re-scrape LVDB's `abbreviation` field and overwrite the local cache.

    One HTTP request per system, so this is deliberately not called at
    import time - the shipped CSV is the normal path.

    Args:
        keys: Keys to fetch. Defaults to every key in the metadata table.
        meta_path: Metadata CSV used to derive `keys` when not given.
        local_path: CSV to write.
    """
    import urllib.request  # Reason: only needed on an explicit refresh.

    if keys is None:
        keys = sorted(load_meta_table(meta_path)['key'].astype(str))
    rows = []
    for key in keys:
        with urllib.request.urlopen(
                LVDB_YAML_URL.format(key=key), timeout=60) as response:
            text = response.read().decode()
        match = re.search(r'^\s+abbreviation:\s*(.+?)\s*$', text, re.M)
        if match:
            rows.append((key, match.group(1).strip().strip('"\'')))
    pd.DataFrame(rows, columns=['key', 'abbreviation']).to_csv(
        local_path, index=False)


def refresh_local_meta_table(
    url: str = DWARF_MW_URL,
    local_path: str = DEFAULT_META_PATH,
) -> None:
    """
    Re-download the metadata table and overwrite the local cache.

    Parameters
    ----------
    url : str
        URL to fetch the metadata CSV from.
    local_path : str
        Local file path to write the metadata CSV to.
    """
    pd.read_csv(url).to_csv(local_path, index=False)


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
     X_proj, Y_proj, R_proj, mask,
     pmra_cosdec, pmdec, pmra_cosdec_err, pmdec_err, vX, vY, vX_err, vY_err) = (
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
     X_proj, Y_proj, R_proj, mask,
     pmra_cosdec, pmdec, pmra_cosdec_err, pmdec_err, vX, vY, vX_err, vY_err) = (
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

def _load_s5comp(
    catalog_path: str,
    meta: DwarfMeta,
    mem_prob_min: float = 0.8,
    instrument='mmt',
    vlos_abs_max: Optional[float] = None,
    apply_perspective_corr: bool = True,
    use_sandford_perspective_corr: bool = False,
    remove_binaries: bool = True,
) -> KinematicData:
    """Load kinematic data from the Bootes I S5 compilation."""
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
         X_proj, Y_proj, R_proj, mask,
         pmra_cosdec, pmdec, pmra_cosdec_err, pmdec_err, vX, vY, vX_err, vY_err) = (
            data_utils.preprocess_kinematic_data(
                ra, dec, vlos_raw, vlos_err, mem_prob, meta,
                vlos_abs_max=vlos_abs_max,
                apply_perspective_corr=apply_perspective_corr,
                extra_mask=extra_mask,
            )
        )
    else:
        (ra, dec, vlos_raw, vlos_err, mem_prob, vlos,
         X_proj, Y_proj, R_proj, mask,
         pmra_cosdec, pmdec, pmra_cosdec_err, pmdec_err, vX, vY, vX_err, vY_err) = (
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
        source='s5comp_' + instrument
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
     X_proj, Y_proj, R_proj, mask,
     pmra_cosdec, pmdec, pmra_cosdec_err, pmdec_err, vX, vY, vX_err, vY_err) = (
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
        source='deimos',
    )

# Membership columns carried by every `combined_structure_*.fits` file.
# Probability columns are cut at `mem_prob_min`; flag columns are 0/1 and
# are used as a straight boolean selection (mem_prob is then set to 1).
PACE_MEMBER_PROB_COLUMNS = (
    'member_v10d1', 'member_v10d2', 'member_v11d1', 'member_v11d2',
    'member_all_v10', 'member_all_v11',
)
PACE_MEMBER_FLAG_COLUMNS = (
    'member_candidate', 'member_final', 'member_zscore',
)
PACE_MEMBER_COLUMNS = PACE_MEMBER_PROB_COLUMNS + PACE_MEMBER_FLAG_COLUMNS

# Value marking "membership was not evaluated for this star". It is not
# NaN, so preprocess_kinematic_data's NaN mask would not catch it, and it
# would survive any `mem_prob_min` below -99.
PACE_MEMBER_SENTINEL = -99.0


def _load_pace(
    catalog_path: str,
    meta: DwarfMeta,
    mem_prob_min: float = 0.8,
    member_column: str = 'member_v11d2',
    vlos_abs_max: Optional[float] = None,
    apply_perspective_corr: bool = True,
) -> KinematicData:
    """
    Load kinematic data from a Pace combined-structure FITS catalog.

    These files (`combined_structure_<galaxy>_v<N>d<M>.fits`) are one
    galaxy per file and combine several spectroscopic programmes, so
    there is no single instrument to select on - the per-instrument
    counts (`num_mmt`, `num_m2fs`, `num_vlt`, ...) vary between files and
    are not used here. What *does* vary meaningfully is the membership
    model: each file carries nine membership columns from different
    versions of the mixture-model fit, and they disagree at the ~5%
    level. `member_column` picks one; it is deliberately required to be
    explicit in the catalog config rather than silently defaulted deep in
    the pipeline.

    The version tag in the file name (e.g. `v9d2`) is the structural-fit
    version and does *not* correspond to the membership column versions
    (`v10d1`, `v11d2`, ...), which is why the column cannot be inferred
    from the file name.

    Args:
        catalog_path: Path to the `combined_structure_*.fits` file.
        meta: Metadata for the dwarf galaxy.
        mem_prob_min: Minimum membership probability. Ignored when
            `member_column` is one of PACE_MEMBER_FLAG_COLUMNS, which are
            0/1 flags rather than probabilities.
        member_column: Which membership column to select on. Must be one
            of PACE_MEMBER_COLUMNS.
        vlos_abs_max: Maximum |vlos - v_sys| in km/s, or None for no cut.
        apply_perspective_corr: If True, apply the full perspective
            rotation correction rather than only subtracting v_sys.

    Returns:
        Container with kinematic data for the selected member stars.

    Raises:
        ValueError: If `member_column` is not a known membership column,
            or if the selection leaves no stars.
    """
    if member_column not in PACE_MEMBER_COLUMNS:
        raise ValueError(
            f"Unknown member_column: {member_column}. "
            f"Available columns: {PACE_MEMBER_COLUMNS}"
        )

    data = at.Table.read(catalog_path, format='fits').to_pandas()

    member = data[member_column].values.astype(float)
    # Reason: -99 means "not evaluated", not "probability -99"; dropping
    # it here keeps a permissive mem_prob_min from letting those stars in.
    evaluated = member != PACE_MEMBER_SENTINEL
    if member_column in PACE_MEMBER_FLAG_COLUMNS:
        select = evaluated & (member == 1)
    else:
        select = evaluated & (member > mem_prob_min)

    if np.sum(select) == 0:
        raise ValueError(
            f"No stars selected from {catalog_path} with "
            f"member_column={member_column} and mem_prob_min={mem_prob_min}"
        )
    data_cut = data[select]

    ra = data_cut['ra'].values
    dec = data_cut['dec'].values
    vlos_raw = data_cut['vlos'].values
    vlos_err = data_cut['vlos_error'].values
    if member_column in PACE_MEMBER_FLAG_COLUMNS:
        mem_prob = np.ones(len(data_cut))
    else:
        mem_prob = data_cut[member_column].values.astype(float)

    (ra, dec, vlos_raw, vlos_err, mem_prob, vlos,
     X_proj, Y_proj, R_proj, mask,
     pmra_cosdec, pmdec, pmra_cosdec_err, pmdec_err, vX, vY, vX_err, vY_err) = (
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
        source='pace_' + member_column,
    )


def _load_mock_cartesian(
    catalog_path: str,
    meta: DwarfMeta,
    projection_axis: int = 0,
    num_max_stars: Optional[int] = None,
    seed: Optional[int] = None,  # None -> np.random.default_rng draws from OS entropy
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
    seed: Optional[int] = None,  # None -> np.random.default_rng draws from OS entropy
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

    # proper motions are optional -- present for mock catalogs built
    # with a PM-only round (e.g. matched to a Gaia footprint)
    has_pm = 'pmra_cosdec' in data.columns and 'pmdec' in data.columns
    pmra_cosdec = data['pmra_cosdec'].values if has_pm else None
    pmdec = data['pmdec'].values if has_pm else None
    pmra_cosdec_err = (
        data['pmra_cosdec_err'].values
        if has_pm and 'pmra_cosdec_err' in data.columns else None
    )
    pmdec_err = (
        data['pmdec_err'].values
        if has_pm and 'pmdec_err' in data.columns else None
    )

    rng = np.random.default_rng(seed)
    if num_max_stars is not None and len(data) > num_max_stars:
        selected_indices = rng.choice(len(data), size=num_max_stars, replace=False)
        ra = ra[selected_indices]
        dec = dec[selected_indices]
        vlos_raw = vlos_raw[selected_indices]
        vlos_err = vlos_err[selected_indices]
        mem_prob = mem_prob[selected_indices]
        if has_pm:
            pmra_cosdec = pmra_cosdec[selected_indices]
            pmdec = pmdec[selected_indices]
            if pmra_cosdec_err is not None:
                pmra_cosdec_err = pmra_cosdec_err[selected_indices]
            if pmdec_err is not None:
                pmdec_err = pmdec_err[selected_indices]

    (ra, dec, vlos_raw, vlos_err, mem_prob, vlos, X_proj, Y_proj, R_proj, _,
     pmra_cosdec, pmdec, pmra_cosdec_err, pmdec_err, vX, vY, vX_err, vY_err) = (
        data_utils.preprocess_kinematic_data(
            ra, dec, vlos_raw, vlos_err, mem_prob, meta,
            vlos_abs_max=vlos_abs_max,
            apply_perspective_corr=apply_perspective_corr,
            pmra_cosdec=pmra_cosdec, pmdec=pmdec,
            pmra_cosdec_err=pmra_cosdec_err, pmdec_err=pmdec_err,
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
        pmra_cosdec=pmra_cosdec * auni.mas / auni.yr if pmra_cosdec is not None else None,
        pmdec=pmdec * auni.mas / auni.yr if pmdec is not None else None,
        pmra_cosdec_err=(
            pmra_cosdec_err * auni.mas / auni.yr if pmra_cosdec_err is not None else None
        ),
        pmdec_err=(
            pmdec_err * auni.mas / auni.yr if pmdec_err is not None else None
        ),
        vX=vX * auni.km / auni.s if vX is not None else None,
        vY=vY * auni.km / auni.s if vY is not None else None,
        vX_err=vX_err * auni.km / auni.s if vX_err is not None else None,
        vY_err=vY_err * auni.km / auni.s if vY_err is not None else None,
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
    loader = LOADERS[source]
    # Reason: mem_prob_min is a named parameter here, so it never reaches
    # **kwargs; forward it explicitly, but only to loaders that take it
    # (the mock loaders inject no membership and do not).
    if 'mem_prob_min' in inspect.signature(loader).parameters:
        kwargs.setdefault('mem_prob_min', mem_prob_min)
    return loader(catalog_path, meta, **kwargs)


LOADERS.update({
    'desi': _load_desi,
    'walker23': _load_walker23,
    's5comp': _load_s5comp,
    'deimos': _load_deimos,
    'pace': _load_pace,
    'mock_cartesian': _load_mock_cartesian,
    'mock_icrs': _load_mock_icrs,
})


def get_loader_defaults(source: str) -> dict:
    """Every keyword argument `source`'s loader accepts beyond
    `catalog_path`/`meta`, mapped to its default value (`None` if the
    parameter has none, i.e. it's required).

    Lets callers (e.g. catalog_registry.py's dataset discovery) surface a
    source's full set of options - including ones added to a loader later,
    like DESI's `vlos_err_floor` or s5comp's `remove_binaries` -
    without maintaining a separate, easily-stale list of them.
    """
    if source not in ALL_LOADERS:
        raise ValueError(f"Unknown source: {source}. Available sources: {ALL_LOADERS}")
    params = inspect.signature(LOADERS[source]).parameters
    return {
        name: (None if param.default is inspect.Parameter.empty else param.default)
        for name, param in params.items()
        if name not in ('catalog_path', 'meta')
    }
