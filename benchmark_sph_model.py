"""
Absolute accuracy benchmark for sph_model.GeneralizedOMJeans.

Comparing the solver against a tighter integration of its own methods only
bounds quadrature error: it reuses nu, M, beta, gbeta and I, so a wrong
integrating factor or projection kernel would pass. These three checks each
test something that one cannot.

Tier 1 - closed form. gNFW with gam = bet = 2 is exactly rho_s (r/r_s)^-2,
    so M(r) = 4 pi rho_s r_s^2 r. With a Plummer tracer and constant
    anisotropy, sigma_r^2 is elementary and sigma_los follows from one
    projection, evaluated here at 30 digits.
Tier 2 - independent reimplementation at 18 digits for a realistic NFW halo
    with full Osipkov-Merritt anisotropy. Every ingredient is written from
    its definition, M(r) is validated against the integral of rho, and the
    identity dln g/dln r = 2 beta(r) is checked directly.
Tier 3 - agama's distribution-function moments. The only check that never
    uses the Jeans equation: agama integrates a Cuddeford-Osipkov-Merritt
    DF f(E, L) in the same potential for <v_r^2>, <v_t^2> and the projected
    <v_los^2>. Note its QuasiSpherical DF has beta -> 1, so tier 3 covers
    two_to_betainf = 2 only.

Run:  python -m dsph_analysis.benchmark_sph_model  (a few minutes,
      from ~/modules or anywhere with it on PYTHONPATH)

Results, 2026-09-10, worst relative error in sigma2_los:
    tier 1  5.6e-05      tier 2  2.3e-05      tier 3  1.5e-03
Tier 3's floor is agama's own integration tolerance: its DF reproduces the
Plummer nu it was built from to only 1.2e-04. For reference, the library's
GeneralizedOMJeans scores 3.9e-02 on the same models.
"""

import sys

import mpmath as mp
import numpy as np
from scipy import constants

from .sph_model import _TO_KM2_S2
from .sph_model import GeneralizedOMJeans


def _model(theta, rh):
    """The model on a grid wide enough that bounds do not bite."""
    m = GeneralizedOMJeans(theta, min_rgrid=1e-4 * rh,
                           max_rgrid=1e4 * rh, n_grid=200)
    m.N_NODES = 40
    return m


def tier1(rho_s=3.0, r_s=1.0, rh=0.25):
    """Closed-form isothermal halo + Plummer tracer + constant beta."""
    mp.mp.dps = 30
    pref = 4 * mp.pi * rho_s * r_s ** 2 * constants.G * _TO_KM2_S2

    def sigma2_r(r, b):
        y = mp.mpf(r) / rh
        if b == 0:
            s = mp.sqrt(1 + y ** 2)
            br = (-y ** 2 * s - mp.mpf(4) / 3 * s
                  + (1 + y ** 2) ** 2 / 2 * mp.log((s + 1) / (s - 1)))
            shape = (1 + y ** 2) ** mp.mpf(2.5) * br / (1 + y ** 2) ** 2
        else:
            j = mp.quad(lambda x: (1 + x ** 2) ** mp.mpf(-2.5), [y, mp.inf])
            shape = j * (1 + y ** 2) ** mp.mpf(2.5) / y
        return pref * shape

    def sigma2_los(R, b):
        R = mp.mpf(R)

        def nu(r):
            return (1 + (r / rh) ** 2) ** mp.mpf(-2.5) * 3 / (4 * mp.pi
                                                              * rh ** 3)

        def f(u):
            r = R * mp.cosh(u)
            return ((1 - b * (R / r) ** 2) * nu(r) * sigma2_r(r, b)
                    * R * mp.cosh(u))

        surf = (1 + (R / rh) ** 2) ** mp.mpf(-2) / (mp.pi * rh ** 2)
        return 2 / surf * mp.quad(f, [0, 2, 6, 12, 25])

    out = {}
    for b in (mp.mpf(0), mp.mpf("0.5")):
        ttb = float(2.0 ** float(b))
        m = _model([np.log10(rho_s), np.log10(r_s), 1.0, 2.0, 2.0, 0.0,
                    ttb, ttb, rh, 0.0, 0.0, 0.0], rh)
        radii = [0.02, 0.1, 0.25, 1.0, 4.0]
        err = [abs(m.sigma2_los(np.array([R]))[0] / float(sigma2_los(R, b))
                   - 1) for R in radii]
        out[float(b)] = (radii, err)
        print(f"  tier 1, beta = {float(b):+.1f}: max rel = {max(err):.2e}")
    return out


def tier2(rh=0.25):
    """18-digit reimplementation, realistic NFW with full OM anisotropy."""
    mp.mp.dps = 18
    rho_s, r_s, r_a = mp.mpf("3.1622776601683795"), mp.mpf(1), mp.mpf("0.5")
    alp, bet, gam = mp.mpf(1), mp.mpf(3), mp.mpf(1)
    b0, binf = mp.mpf("-0.2"), mp.mpf(1)
    rh = mp.mpf(str(rh))
    G = mp.mpf(constants.G) * mp.mpf(_TO_KM2_S2)

    def rho(r):
        return (rho_s * (r / r_s) ** (-gam)
                * (1 + (r / r_s) ** alp) ** (-(bet - gam) / alp))

    def mass(r):
        rn = r / r_s
        return (4 * mp.pi * rho_s * r_s ** 3 / (3 - gam) * rn ** (3 - gam)
                * mp.hyp2f1((3 - gam) / alp, (bet - gam) / alp,
                            1 + (3 - gam) / alp, -(rn ** alp)))

    def nu(r):
        return 3 / (4 * mp.pi * rh ** 3) * (1 + (r / rh) ** 2) ** mp.mpf(-2.5)

    def beta(r):
        return b0 + (binf - b0) * r ** 2 / (r ** 2 + r_a ** 2)

    def gbeta(r):
        return r ** (2 * b0) * (1 + r ** 2 / r_a ** 2) ** (binf - b0)

    def sigma2_r(r):
        w = nu(r) * gbeta(r)
        return mp.quad(lambda s: G * mass(s) / s ** 2 * nu(s) * gbeta(s),
                       [r, 2 * r, 10 * r, 100 * r, mp.inf]) / w

    def sigma2_los(R):
        def f(u):
            r = R * mp.cosh(u)
            return ((1 - beta(r) * (R / r) ** 2) * nu(r) * sigma2_r(r)
                    * R * mp.cosh(u))
        surf = (1 + (R / rh) ** 2) ** -2 / (mp.pi * rh ** 2)
        return 2 / surf * mp.quad(f, [0, 1, 3, 8, 16])

    wg = max(abs(float(mp.diff(lambda x: mp.log(gbeta(mp.e ** x)),
                               mp.log(mp.mpf(r))) - 2 * beta(mp.mpf(r))))
             for r in ("0.03", "0.2", "1.0", "5.0"))
    print(f"  tier 2, dln g/dln r - 2 beta: max |diff| = {wg:.2e}")
    wm = max(abs(float(mass(mp.mpf(r)) / (4 * mp.pi * mp.quad(
        lambda s: rho(s) * s ** 2, [0, mp.mpf(r) / 10, mp.mpf(r)])) - 1))
        for r in ("0.02", "0.3", "3.0"))
    print(f"  tier 2, M(r) closed form vs integral of rho: {wm:.2e}")

    m = _model([float(mp.log10(rho_s)), 0.0, 1.0, 3.0, 1.0,
                float(mp.log10(r_a)), float(2 ** b0), float(2 ** binf),
                float(rh), 0.0, 0.0, 0.0], float(rh))
    radii = [0.02, 0.25, 1.0, 3.0]
    err = [abs(m.sigma2_los(np.array([R]))[0] / float(sigma2_los(mp.mpf(
        str(R)))) - 1) for R in radii]
    print(f"  tier 2, sigma2_los: max rel = {max(err):.2e}")
    return radii, err


def tier3(rho_s=10 ** 7.5, r_s=1.0, rh=0.25, r_a=0.5, beta0=-0.2):
    """agama DF moments; independent of the Jeans equation entirely."""
    import agama
    import astropy.units as u

    agama.setUnits(mass=1 * u.Msun, length=1 * u.kpc, velocity=1 * u.km / u.s)
    pot = agama.Potential(type="Spheroid", alpha=1.0, beta=3.0, gamma=1.0,
                          scaleRadius=r_s, densityNorm=rho_s)
    tracer = agama.Density(type="Plummer", mass=1.0, scaleRadius=rh)
    df = agama.DistributionFunction(type="QuasiSpherical", potential=pot,
                                    density=tracer, beta0=beta0, r_a=r_a)
    gm = agama.GalaxyModel(pot, df)

    m = _model([np.log10(rho_s / 1e7), np.log10(r_s), 1.0, 3.0, 1.0,
                np.log10(r_a), 2.0 ** beta0, 2.0, rh, 0.0, 0.0, 0.0], rh)
    radii = np.array([0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
    pts3 = np.column_stack([radii, np.zeros_like(radii),
                            np.zeros_like(radii)])
    v2 = gm.moments(pts3, dens=False, vel=False, vel2=True)
    beta_df = 1.0 - (v2[:, 1] + v2[:, 2]) / (2.0 * v2[:, 0])
    nu_df = gm.moments(pts3, dens=True, vel=False, vel2=False)
    print(f"  tier 3, nu:   agama DF vs Plummer, max rel = "
          f"{np.abs(m.nu(radii) / nu_df - 1).max():.2e}  (agama's floor)")
    print(f"  tier 3, beta: max |diff| = "
          f"{np.abs(m.beta(radii) - beta_df).max():.2e}")
    print(f"  tier 3, sigma_r^2: max rel = "
          f"{np.abs(m.sigma2_r_vec(radii) / v2[:, 0] - 1).max():.2e}")

    v2p = gm.moments(np.column_stack([radii, np.zeros_like(radii)]),
                     dens=False, vel=False, vel2=True)
    err = np.abs(m.sigma2_los(radii) / v2p[:, 2] - 1)
    print(f"  tier 3, sigma_los^2: max rel = {err.max():.2e}, "
          f"median = {np.median(err):.2e}")
    return list(radii), list(err)


def main():
    print("tier 1: closed-form isothermal + Plummer + constant beta")
    t1 = tier1()
    print("\ntier 2: 18-digit reimplementation, NFW + full OM")
    t2 = tier2()
    print("\ntier 3: agama distribution-function moments")
    t3 = tier3()
    return t1, t2, t3


if __name__ == "__main__":
    sys.exit(main() and 0)
