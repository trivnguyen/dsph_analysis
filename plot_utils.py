import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from analysis import utils

mpl.style.use('/mnt/home/tnguyen/default.mplstyle')

def plot_projection(data, save_path=None, title=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(data['pos2d'][:, 0], data['pos2d'][:, 1], s=20)
    axes[0].set_xlabel('x [kpc]')
    axes[0].set_ylabel('y [kpc]')
    axes[1].hist(data['pos2d'][:, 0], bins=20, histtype='step')
    axes[1].hist(data['pos2d'][:, 1], bins=20, histtype='step')
    axes[1].set_xlabel('x/y [kpc]')
    axes[2].hist(data['vel_los'], bins=20, histtype='step')
    axes[2].set_xlabel(r'$v_{los}$ [km/s]')
    if title is not None:
        fig.suptitle(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes

def plot_lp(lp_result, data, save_path=None, title=None):
    # Extract the best fit parameters
    posterior = lp_result.posterior
    ln_a = np.log(posterior['a'].values)
    ln_L = np.log(posterior['L'].values)

    # Rdata = np.linalg.norm(data['pos2d'], axis=1)
    Rdata = data['rad2d']

    # Generate the Plummer profile using the best fit parameters
    R = np.linspace(np.min(Rdata), np.max(Rdata), 1000)
    ln_Plummer = utils.ln_Plummer2d(np.log(R), ln_L[:, None], ln_a[:, None])
    ln_Plummer_q = np.quantile(ln_Plummer, [0.5, 0.16, 0.84, 0.025, 0.975], axis=0)

    # Calculate the surface density data
    Sigma, Sigma_lo, Sigma_hi, logRbins_lo, logRbins_hi = utils.calc_Sigma_data(
        Rdata, alpha=0.32, return_bounds=True)
    Rbins_lo = 10**logRbins_lo
    Rbins_hi = 10**logRbins_hi
    Rbins_ce = 0.5 * (Rbins_lo + Rbins_hi)
    sig_lo = Sigma - Sigma_lo
    sig_hi = Sigma_hi - Sigma

    # Start plotting
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(R, np.exp(ln_Plummer_q[0]), color='C0', label='Med')
    ax.fill_between(R, np.exp(ln_Plummer_q[1]), np.exp(ln_Plummer_q[2]), color='C0', alpha=0.3, label='68\% confidence')
    ax.fill_between(R, np.exp(ln_Plummer_q[3]), np.exp(ln_Plummer_q[4]), color='C0', alpha=0.1, label='95\% confidence')
    ax.errorbar(Rbins_ce, Sigma, yerr=[sig_lo, sig_hi], color='C1', fmt='o', label='Data')
    ax.set_xlabel(r'$R$ [kpc]')
    ax.set_ylabel(r'Surface density [M$_\odot$/kpc$^2$]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_density_profile(posterior, rce=None, save_path=None, title=None):

    if rce is None:
        rce = np.logspace(-2, 2, 100)

    # calculate the predicted profile
    gamma = posterior['gamma'].values
    ln_rdm = np.log(posterior['rdm'].values)
    ln_rho0 = np.log(posterior['rho0'].values)
    ln_a = np.log(posterior['a'].values)
    ln_rho_posterior = utils.ln_rho_gnfw(
        np.log(rce)[None, :], ln_rho0[:, None], ln_rdm[:, None], gamma[:, None])
    ln_rho_q = np.nanquantile(
        ln_rho_posterior, [0.5, 0.16, 0.84, 0.02, 0.98], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(rce, np.exp(ln_rho_q[0]), label='Med', color='C0')
    ax.fill_between(
        rce, np.exp(ln_rho_q[1]), np.exp(ln_rho_q[2]), color='C0', alpha=0.3, label='68\% confidence')
    ax.fill_between(
        rce, np.exp(ln_rho_q[3]), np.exp(ln_rho_q[4]), color='C0', alpha=0.1, label='95\% confidence')
    ax.axvline(np.median(np.exp(ln_a)), color='k', linestyle='--', alpha=0.5, label=r'$R_{\rm star}$')
    ax.set_xlabel(r'$r$ [kpc]')
    ax.set_ylabel(r'Density [M$_\odot$/kpc$^3$]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_mass_profile(posterior, rce=None, save_path=None, title=None):

    if rce is None:
        rce = np.logspace(-2, 2, 100)

    # calculate the predicted profile
    gamma = posterior['gamma'].values
    ln_rdm = np.log(posterior['rdm'].values)
    ln_rho0 = np.log(posterior['rho0'].values)
    ln_a = np.log(posterior['a'].values)
    ln_mass_posterior = utils.ln_mass_gnfw(
        np.log(rce)[None, :], ln_rho0[:, None], ln_rdm[:, None], gamma[:, None])
    ln_mass_q = np.nanquantile(
        ln_mass_posterior, [0.5, 0.16, 0.84, 0.02, 0.98], axis=0)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(rce, np.exp(ln_mass_q[0]), label='Med', color='C0')
    ax.fill_between(rce, np.exp(ln_mass_q[1]), np.exp(ln_mass_q[2]), color='C0', alpha=0.3, label='68\% confidence')
    ax.fill_between(rce, np.exp(ln_mass_q[3]), np.exp(ln_mass_q[4]), color='C0', alpha=0.1, label='95\% confidence')
    ax.axvline(np.median(np.exp(ln_a)), color='k', linestyle='--', alpha=0.5, label=r'$R_{\rm star}$')
    ax.set_xlabel(r'$r$ [kpc]')
    ax.set_ylabel(r'Mass [M$_\odot$]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_vel_anisotropy(posterior, rce=None, save_path=None, title=None):

    if rce is None:
        rce = np.logspace(-2, 2, 100)

    # calculate the predicted profile
    beta_0 = posterior['beta_0'].values
    ln_ra = np.log(posterior['ra'].values)
    ln_a = np.log(posterior['a'].values)
    beta_posterior = utils.betaOM(np.log(rce), beta_0[:, None], ln_ra[:, None])
    beta_q = np.nanquantile(beta_posterior, [0.5, 0.16, 0.84, 0.02, 0.98], axis=0)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(rce, beta_q[0], color='C0', label='Med')
    ax.fill_between(rce, beta_q[1], beta_q[2], color='C0', alpha=0.3, label='68\% confidence')
    ax.fill_between(rce, beta_q[3], beta_q[4], color='C0', alpha=0.1, label='95\% confidence')
    ax.set_xlabel(r'$r$ [kpc]')
    ax.set_ylabel(r'$\beta(r)$')
    ax.set_xscale('log')
    ax.legend()
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_vel_dispersion(posterior, rce=None, rint=None, save_path=None, title=None):

    if rce is None:
        rce = np.logspace(-2, 2, 100)

    velsig_fit_posterior = []
    for i in range(len(posterior)):
        gamma_i = posterior['gamma'].iloc[i]
        ln_rdm_i = posterior['ln_rdm'].iloc[i]
        ln_rho0_i = posterior['ln_rho0'].iloc[i]
        beta_0_i = posterior['beta_0'].iloc[i]
        ln_ra_i = posterior['ln_ra'].iloc[i]
        ln_a_i = posterior['ln_a'].iloc[i]
        ln_L_i = posterior['ln_L'].iloc[i]
        velsig_fit_i = np.sqrt(
            utils.calc_velsig2_los(
                np.log(r_data), gamma_i, ln_rdm_i, ln_rho0_i, beta_0_i, ln_ra_i, ln_a_i, ln_L_i,
                rint=rint))
        velsig_fit_posterior.append(velsig_fit_i)
    velsig_fit_posterior = np.array(velsig_fit_posterior)
    velsig_fit_q = np.nanquantile(velsig_fit_posterior, [0.5, 0.16, 0.84, 0.02, 0.98], axis=0)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(r_data, velsig_fit_q[0], color='C0', label='Med')
    ax.fill_between(r_data, velsig_fit_q[1], velsig_fit_q[2], alpha=0.3, color='C0', label='68\% confidence')
    ax.fill_between(r_data, velsig_fit_q[3], velsig_fit_q[4], alpha=0.1, color='C0', label='95\% confidence')
    ax.errorbar(r_data, velsig_data, fmt='-o', color='C1', label='Data')
    ax.set_xlabel(r'$r$ [kpc]')
    ax.set_ylabel(r'$\sigma$ [km/s]')
    ax.legend()
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax
