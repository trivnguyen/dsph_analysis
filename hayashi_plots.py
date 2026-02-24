"""Plotting functions for Hayashi Jeans analysis results."""

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from corner import corner
from tqdm import tqdm
from pathlib import Path

from hayashi_jeans.sph_model import RadialDisp


# Parameter labels for corner plots
PARAM_LABELS = [
    r"$\log_{10} \rho_s$",
    r"$\log_{10} r_s$",
    r"$\alpha$",
    r"$\beta$",
    r"$\gamma$",
    r"$\log_{10} r_{\beta}$",
    r"$\eta$",
    r"$2^{\beta_{0}}$",
    r"$2^{\beta_{\infty}}$",
    r"$r_\mathrm{half}$",
    r"$V_\mathrm{sys}$",
]


def compute_profile_samples(
    posterior: NDArray[np.floating],
    r_min: float = 1e-2,
    r_max: float = 1e2,
    n_radius: int = 200,
    n_max_samples: int = 1000,
) -> dict:
    """
    Compute density, mass, anisotropy, and velocity dispersion profiles from posterior.

    Parameters
    ----------
    posterior : NDArray[np.floating]
        Posterior samples of shape (n_samples, n_params).
    r_min : float, optional
        Minimum radius in kpc. Default is 0.01.
    r_max : float, optional
        Maximum radius in kpc. Default is 100.
    n_radius : int, optional
        Number of radial points. Default is 200.
    n_max_samples : int, optional
        Maximum number of samples to use. Default is 1000.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'r': radial array
        - 'rho': density samples (n_samples, n_radius)
        - 'mass': enclosed mass samples (n_samples, n_radius)
        - 'beta': anisotropy samples (n_samples, n_radius)
        - 'sigma_los': LOS velocity dispersion samples (n_samples, n_radius)
    """
    n_samples = min(n_max_samples, posterior.shape[0])
    r_vec = np.logspace(np.log10(r_min), np.log10(r_max), n_radius)

    rho_samples = np.zeros((n_samples, n_radius))
    mass_samples = np.zeros((n_samples, n_radius))
    beta_samples = np.zeros((n_samples, n_radius))
    sigma_los_samples = np.zeros((n_samples, n_radius))

    for i in tqdm(range(n_samples), desc="Computing profiles"):
        disp = RadialDisp(
            posterior[i], min_radius=r_min, max_radius=r_max, n_radius=n_radius
        )
        rho_samples[i] = disp.rho(r_vec)
        mass_samples[i] = disp.M(r_vec)
        beta_samples[i] = disp.beta(r_vec)
        sigma_los_samples[i] = np.sqrt(disp.Sigma2_los())

    return {
        'r': r_vec,
        'rho': rho_samples,
        'mass': mass_samples,
        'beta': beta_samples,
        'sigma_los': sigma_los_samples,
    }


def plot_percentiles(
    x: NDArray[np.floating],
    samples: NDArray[np.floating],
    ax: plt.Axes,
    color: str = 'C0',
    label: str | None = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot median and credible intervals from samples.

    Parameters
    ----------
    x : NDArray[np.floating]
        X-axis values.
    samples : NDArray[np.floating]
        Samples of shape (n_samples, n_x).
    ax : plt.Axes
        Matplotlib axes to plot on.
    color : str, optional
        Color for the plot. Default is 'C0'.
    label : str | None, optional
        Label for the median line.
    **kwargs
        Additional keyword arguments passed to plot/fill_between.

    Returns
    -------
    plt.Axes
        The axes with the plot.
    """
    q = np.percentile(samples, q=[50, 16, 84, 2, 98], axis=0)
    ax.fill_between(x, q[3], q[4], alpha=0.2, color=color, **kwargs)
    ax.fill_between(x, q[1], q[2], alpha=0.5, color=color, **kwargs)
    ax.plot(x, q[0], lw=2, color=color, label=label, **kwargs)
    return ax


def plot_corner(
    posterior: NDArray[np.floating],
    labels: list[str] | None = None,
    color: str = 'black',
    fig: plt.Figure | None = None,
    save_path: str | Path | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Create corner plot of posterior samples.

    Parameters
    ----------
    posterior : NDArray[np.floating]
        Posterior samples of shape (n_samples, n_params).
    labels : list[str] | None, optional
        Parameter labels. Uses default Hayashi labels if None.
    color : str, optional
        Color for the plot. Default is 'black'.
    fig : plt.Figure | None, optional
        Existing figure to add to (for overplotting).
    save_path : str | Path | None, optional
        Path to save the figure.
    **kwargs
        Additional keyword arguments passed to corner.corner.

    Returns
    -------
    plt.Figure
        The corner plot figure.
    """
    if labels is None:
        labels = PARAM_LABELS

    default_kwargs = dict(
        bins=20,
        smooth=True,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 15},
        hist_kwargs={"density": True},
    )
    default_kwargs.update(kwargs)

    fig = corner(posterior, labels=labels, color=color, fig=fig, **default_kwargs)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_profiles(
    profile_data: dict,
    obs_vdisp: dict | None = None,
    R_kin: NDArray[np.floating] | None = None,
    rhalf_kpc: float | None = None,
    mass_wolf: float | None = None,
    mass_wolf_err: tuple[float, float] | None = None,
    title: str | None = None,
    color: str = 'C0',
    data_color: str = 'k',
    figsize: tuple[float, float] = (25, 6),
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, NDArray]:
    """
    Plot 4-panel figure with density, mass, anisotropy, and velocity dispersion profiles.

    Parameters
    ----------
    profile_data : dict
        Output from compute_profile_samples().
    obs_vdisp : dict | None, optional
        Observed velocity dispersion profile from calc_obs_vdisp_profile().
    R_kin : NDArray[np.floating] | None, optional
        Projected radii of kinematic tracers for histogram.
    rhalf_kpc : float | None, optional
        Half-light radius in kpc for vertical line annotation.
    mass_wolf : float | None, optional
        Wolf mass estimate in solar masses.
    mass_wolf_err : tuple[float, float] | None, optional
        Wolf mass error (minus, plus) in solar masses.
    title : str | None, optional
        Figure title.
    color : str, optional
        Color for model profiles. Default is 'C0'.
    data_color : str, optional
        Color for data points. Default is 'k'.
    figsize : tuple[float, float], optional
        Figure size.
    save_path : str | Path | None, optional
        Path to save the figure.

    Returns
    -------
    tuple[plt.Figure, NDArray]
        Figure and axes array.
    """
    r_vec = profile_data['r']
    log_r = np.log10(r_vec)

    fig, axes = plt.subplots(
        2, 4, figsize=figsize, dpi=300, height_ratios=[3, 1], sharex=True
    )

    # Plot profiles
    plot_percentiles(log_r, np.log10(profile_data['rho']), axes[0, 0], color=color)
    plot_percentiles(log_r, np.log10(profile_data['mass']) + 7, axes[0, 1], color=color)
    plot_percentiles(log_r, profile_data['beta'], axes[0, 2], color=color)
    plot_percentiles(log_r, profile_data['sigma_los'], axes[0, 3], color=color)

    # Plot tracer distribution histogram
    if R_kin is not None:
        for ax in axes[1, :]:
            ax.hist(np.log10(R_kin), bins=10, color='k', alpha=0.7)

    # Plot observed velocity dispersion
    if obs_vdisp is not None:
        yerr = [
            obs_vdisp['velsig'] - obs_vdisp['velsig_lo'],
            obs_vdisp['velsig_hi'] - obs_vdisp['velsig'],
        ]
        xerr = [
            np.log10(obs_vdisp['R_mid'] / obs_vdisp['R_lo']),
            np.log10(obs_vdisp['R_hi'] / obs_vdisp['R_mid']),
        ]
        axes[0, 3].errorbar(
            np.log10(obs_vdisp['R_mid']),
            obs_vdisp['velsig'],
            yerr=yerr,
            xerr=xerr,
            fmt='o',
            color=data_color,
            label='Data',
            alpha=0.7,
        )

    # Labels
    axes[0, 0].set_ylabel(r'$\log_{10} \rho(r)$ [M$_\odot$ kpc$^{-3}$]')
    axes[0, 1].set_ylabel(r'$\log_{10} M(r)$ [M$_\odot$]')
    axes[0, 2].set_ylabel(r'$\beta (r)$')
    axes[0, 3].set_ylabel(r'$\sigma_\mathrm{los} (r)$ [km s$^{-1}$]')

    for ax in axes[1, :]:
        ax.set_xlabel(r'$\log_{10} r$ [kpc]')
        ax.set_ylabel(r'$N_\mathrm{tracers}$')

    # Set axis limits
    axes[0, 0].set_xlim(-2, 1)
    axes[0, 0].set_ylim(5, 10)
    axes[0, 1].set_ylim(5, 10)
    axes[0, 3].set_ylim(0, 20)

    # Mark half-light radius
    if rhalf_kpc is not None:
        for ax in axes.ravel():
            ax.axvline(np.log10(rhalf_kpc), color='k', linestyle='--')
        for ax in axes[0].ravel():
            ax.annotate(
                r'$r_{\star, 1/2}$',
                xy=(0.5, 0.9),
                xycoords='axes fraction',
                xytext=(5, 0),
                textcoords='offset points',
                color='k',
                fontsize=20,
                va='center',
            )

    # Plot Wolf mass
    if mass_wolf is not None:
        axes[0, 1].axhline(np.log10(mass_wolf), color='k', linestyle='-')
        if mass_wolf_err is not None:
            axes[0, 1].fill_between(
                [log_r[0], log_r[-1]],
                np.log10(mass_wolf - mass_wolf_err[0]),
                np.log10(mass_wolf + mass_wolf_err[1]),
                color='k',
                alpha=0.5,
            )
        axes[0, 1].annotate(
            r'$M_{\mathrm{Wolf}}$',
            xy=(0.9, 0.3),
            xycoords='axes fraction',
            xytext=(-5, 0),
            textcoords='offset points',
            color='k',
            fontsize=20,
            va='center',
            ha='right',
        )

    # Legend
    handles = [
        mlines.Line2D([], [], color='k', lw=3, label='Median'),
        mpatches.Patch(color='gray', alpha=0.5, label=r'68\% CI'),
    ]
    axes[0, 0].legend(handles=handles, loc='lower left', fontsize=20, markerfirst=True)

    if title is not None:
        fig.suptitle(title, fontsize=24)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes


def plot_kinematic_data(
    R_kin: NDArray[np.floating],
    vlos: NDArray[np.floating],
    vlos_corr: NDArray[np.floating] | None = None,
    obs_vdisp: dict | None = None,
    obs_vdisp_uncorr: dict | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, NDArray]:
    """
    Plot velocity vs radius with velocity dispersion profile.

    Parameters
    ----------
    R_kin : NDArray[np.floating]
        Projected radius in kpc.
    vlos : NDArray[np.floating]
        Line-of-sight velocities (uncorrected) in km/s.
    vlos_corr : NDArray[np.floating] | None, optional
        Perspective-corrected velocities in km/s.
    obs_vdisp : dict | None, optional
        Velocity dispersion profile for corrected data.
    obs_vdisp_uncorr : dict | None, optional
        Velocity dispersion profile for uncorrected data.
    title : str | None, optional
        Figure title.
    figsize : tuple[float, float], optional
        Figure size.
    save_path : str | Path | None, optional
        Path to save the figure.

    Returns
    -------
    tuple[plt.Figure, NDArray]
        Figure and axes array.
    """
    fig, axes = plt.subplots(
        3, 1, figsize=figsize, height_ratios=[3, 3, 1], sharex=True, dpi=300,
        constrained_layout=True
    )

    # Plot velocities
    if vlos_corr is not None:
        axes[0].plot(
            R_kin, vlos_corr, 'o', color='C0', markersize=4, alpha=0.7, label='Corrected'
        )
        axes[0].plot(
            R_kin, vlos, 'o', color='C1', markersize=4, alpha=0.7, label='Uncorrected'
        )
    else:
        axes[0].plot(R_kin, vlos, 'o', color='C0', markersize=4, alpha=0.7)

    # Plot velocity dispersion profiles
    if obs_vdisp is not None:
        yerr = [
            obs_vdisp['velsig'] - obs_vdisp['velsig_lo'],
            obs_vdisp['velsig_hi'] - obs_vdisp['velsig'],
        ]
        axes[1].errorbar(
            obs_vdisp['R_mid'],
            obs_vdisp['velsig'],
            yerr=yerr,
            fmt='o',
            color='C0',
            label='Corrected',
            alpha=0.7,
            capsize=5,
        )

    if obs_vdisp_uncorr is not None:
        yerr_uncorr = [
            obs_vdisp_uncorr['velsig'] - obs_vdisp_uncorr['velsig_lo'],
            obs_vdisp_uncorr['velsig_hi'] - obs_vdisp_uncorr['velsig'],
        ]
        axes[1].errorbar(
            obs_vdisp_uncorr['R_mid'],
            obs_vdisp_uncorr['velsig'],
            yerr=yerr_uncorr,
            fmt='o',
            color='C1',
            label='Uncorrected',
            alpha=0.7,
            capsize=5,
        )

    # Plot radial distribution histogram
    axes[2].hist(
        R_kin, bins=20, histtype='stepfilled', color='k', alpha=0.3, density=False
    )

    # Labels
    axes[0].set_ylabel(r'$v_\mathrm{los}$ [km/s]', fontsize=16)
    axes[1].set_ylabel(r'$\sigma_\mathrm{los}$ [km/s]', fontsize=16)
    axes[2].set_ylabel(r'$N_\mathrm{tracer}$', fontsize=16)
    axes[2].set_xlabel(r'Projected radius $R$ [kpc]', fontsize=16)

    # Legend
    if vlos_corr is not None:
        handles = [
            mlines.Line2D(
                [], [], color='C0', marker='o', linestyle='None',
                markersize=6, label='Corrected'
            ),
            mlines.Line2D(
                [], [], color='C1', marker='o', linestyle='None',
                markersize=6, label='Uncorrected'
            ),
        ]
        axes[0].legend(handles=handles, fontsize=12, loc='upper right')

    if title is not None:
        axes[0].set_title(title, fontsize=16)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes


def plot_profiles_comparison(
    profile_data_list: list[dict],
    obs_vdisp_list: list[dict] | None = None,
    R_kin_list: list[NDArray[np.floating]] | None = None,
    rhalf_kpc: float | None = None,
    mass_wolf: float | None = None,
    mass_wolf_err: tuple[float, float] | None = None,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (25, 6),
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, NDArray]:
    """
    Plot comparison of multiple profile fits.

    Parameters
    ----------
    profile_data_list : list[dict]
        List of profile data dictionaries from compute_profile_samples().
    obs_vdisp_list : list[dict] | None, optional
        List of observed velocity dispersion profiles.
    R_kin_list : list[NDArray[np.floating]] | None, optional
        List of kinematic tracer radii arrays.
    rhalf_kpc : float | None, optional
        Half-light radius in kpc.
    mass_wolf : float | None, optional
        Wolf mass estimate.
    mass_wolf_err : tuple[float, float] | None, optional
        Wolf mass error.
    labels : list[str] | None, optional
        Labels for each dataset.
    colors : list[str] | None, optional
        Colors for each dataset.
    title : str | None, optional
        Figure title.
    figsize : tuple[float, float], optional
        Figure size.
    save_path : str | Path | None, optional
        Path to save the figure.

    Returns
    -------
    tuple[plt.Figure, NDArray]
        Figure and axes array.
    """
    n_datasets = len(profile_data_list)
    if colors is None:
        colors = [f'C{i}' for i in range(n_datasets)]
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(n_datasets)]

    r_vec = profile_data_list[0]['r']
    log_r = np.log10(r_vec)

    fig, axes = plt.subplots(
        2, 4, figsize=figsize, dpi=300, height_ratios=[3, 1], sharex=True
    )

    # Plot profiles for each dataset
    for i, (profile_data, color) in enumerate(zip(profile_data_list, colors)):
        plot_percentiles(log_r, np.log10(profile_data['rho']), axes[0, 0], color=color)
        plot_percentiles(
            log_r, np.log10(profile_data['mass']) + 7, axes[0, 1], color=color
        )
        plot_percentiles(log_r, profile_data['beta'], axes[0, 2], color=color)
        plot_percentiles(log_r, profile_data['sigma_los'], axes[0, 3], color=color)

    # Plot tracer distributions
    if R_kin_list is not None:
        for ax in axes[1, :]:
            for R_kin, color in zip(R_kin_list, colors):
                ax.hist(np.log10(R_kin), bins=10, color=color, alpha=0.5, density=True)

    # Plot observed velocity dispersions
    if obs_vdisp_list is not None:
        markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p']
        data_colors = ['darkblue', 'darkred', 'darkgreen', 'darkorange']
        for i, obs_vdisp in enumerate(obs_vdisp_list):
            yerr = [
                obs_vdisp['velsig'] - obs_vdisp['velsig_lo'],
                obs_vdisp['velsig_hi'] - obs_vdisp['velsig'],
            ]
            xerr = [
                np.log10(obs_vdisp['R_mid'] / obs_vdisp['R_lo']),
                np.log10(obs_vdisp['R_hi'] / obs_vdisp['R_mid']),
            ]
            axes[0, 3].errorbar(
                np.log10(obs_vdisp['R_mid']),
                obs_vdisp['velsig'],
                yerr=yerr,
                xerr=xerr,
                fmt=markers[i % len(markers)],
                color=data_colors[i % len(data_colors)],
                alpha=0.7,
                markersize=8,
            )

    # Labels
    axes[0, 0].set_ylabel(r'$\log_{10} \rho(r)$ [M$_\odot$ kpc$^{-3}$]')
    axes[0, 1].set_ylabel(r'$\log_{10} M(r)$ [M$_\odot$]')
    axes[0, 2].set_ylabel(r'$\beta (r)$')
    axes[0, 3].set_ylabel(r'$\sigma_\mathrm{los} (r)$ [km s$^{-1}$]')

    for ax in axes[1, :]:
        ax.set_xlabel(r'$\log_{10} r$ [kpc]')
        ax.set_ylabel(r'$\mathrm{PDF}_\mathrm{tracers}$')

    # Axis limits
    axes[0, 0].set_xlim(-2, 1)
    axes[0, 0].set_ylim(5, 10)
    axes[0, 1].set_ylim(5, 10)
    axes[0, 3].set_ylim(0, 20)

    # Mark half-light radius
    if rhalf_kpc is not None:
        for ax in axes.ravel():
            ax.axvline(np.log10(rhalf_kpc), color='k', linestyle='--')
        for ax in axes[0].ravel():
            ax.annotate(
                r'$R_{\star, 1/2}$',
                xy=(0.5, 0.9),
                xycoords='axes fraction',
                xytext=(5, 0),
                textcoords='offset points',
                color='k',
                fontsize=20,
                va='center',
            )

    # Plot Wolf mass
    if mass_wolf is not None:
        axes[0, 1].axhline(np.log10(mass_wolf), color='k', linestyle='-')
        if mass_wolf_err is not None:
            axes[0, 1].fill_between(
                [log_r[0], log_r[-1]],
                np.log10(mass_wolf - mass_wolf_err[0]),
                np.log10(mass_wolf + mass_wolf_err[1]),
                color='k',
                alpha=0.5,
            )
        axes[0, 1].annotate(
            r'$M_{\mathrm{Wolf}}$',
            xy=(0.9, 0.3),
            xycoords='axes fraction',
            xytext=(-5, 0),
            textcoords='offset points',
            color='k',
            fontsize=20,
            va='center',
            ha='right',
        )

    # Legends
    handles_ci = [
        mlines.Line2D([], [], color='k', lw=3, label='Median'),
        mpatches.Patch(color='gray', alpha=0.5, label=r'68\% CI'),
    ]
    axes[0, 0].legend(
        handles=handles_ci, loc='lower left', fontsize=20, markerfirst=True
    )

    model_handles = [
        mpatches.Patch(color=c, alpha=0.75, label=lbl)
        for c, lbl in zip(colors, labels)
    ]
    axes[0, 3].legend(
        handles=model_handles,
        loc='center left',
        title=title if title else '',
        fontsize=18,
        title_fontsize=18,
        markerfirst=True,
        bbox_to_anchor=(1.05, 0.9),
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes
