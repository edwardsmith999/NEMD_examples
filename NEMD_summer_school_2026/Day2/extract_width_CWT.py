#!/usr/bin/env python3

import argparse
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf


def slab_erf(z, rho_v, rho_l, z_left, z_right, width):
    """Double-erf density profile for a liquid slab with two interfaces."""
    return rho_v + 0.5 * (rho_l - rho_v) * (
        erf((z - z_left) / (np.sqrt(2.0) * width))
        - erf((z - z_right) / (np.sqrt(2.0) * width))
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a slab density profile to a double-erf form and compare the "
            "fitted width with a naive capillary-wave-theory estimate."
        )
    )
    parser.add_argument( "profile", help="Input profile file, e.g. itim_lab.dat.",)
    parser.add_argument( "--gamma", type=float, required=True, help="Surface tension in reduced LJ units.",)
    parser.add_argument( "--box-width", type=float, required=True, help="Lateral box width L_parallel in reduced LJ units.",)
    parser.add_argument( "--cutoff", type=float, required=True, help="Small-scale/UV cutoff a in reduced LJ units.",)
    parser.add_argument( "--temperature", type=float, default=0.82, help="Reduced temperature T*. Default: 0.82.",)
    parser.add_argument( "-o", "--output", default="lab_density_fit.dat", help="Output file with z, rho, fitted rho. Default: lab_density_fit.dat.",)

    args = parser.parse_args()

    data = np.loadtxt(args.profile, usecols=(0,1))
    z, rho = data[:, 0], data[:, 1]

    finite = np.isfinite(z) & np.isfinite(rho)
    z = z[finite]
    rho = rho[finite]

    # crude initial guesses
    rho_v0 = np.percentile(rho, 5)
    rho_l0 = np.percentile(rho, 95)

    # locate rough slab boundaries from half-density crossing
    rho_half = 0.5 * (rho_l0 + rho_v0)
    above = rho > rho_half
    idx = np.where(above)[0]

    if len(idx) < 2:
        raise RuntimeError(
            "Could not locate the slab boundaries from the half-density crossing."
        )

    z_left0 = z[idx[0]]
    z_right0 = z[idx[-1]]
    width0 = 1.5

    p0 = [rho_v0, rho_l0, z_left0, z_right0, width0]

    bounds = (
        [0.0, 0.0, z.min(), z.min(), 1.0e-6],
        [np.inf, np.inf, z.max(), z.max(), np.inf],
    )

    popt, pcov = curve_fit(
        slab_erf,
        z,
        rho,
        p0=p0,
        bounds=bounds,
        maxfev=10000,
    )

    rho_v, rho_l, z_left, z_right, width = popt

    rho_fit = slab_erf(z, *popt)

    np.savetxt(
        args.output,
        np.c_[z, rho, rho_fit],
        header="z rho_lab rho_double_erf_fit",
    )
    
    w_cwt = np.sqrt(args.temperature / (2.0 * np.pi * args.gamma) * np.log(args.box_width / args.cutoff))

    print("Temperature T*:                 ", args.temperature)
    print("Surface tension gamma*:         ", args.gamma)
    print("Lateral box width L_parallel:   ", args.box_width)
    print("UV cutoff a:                    ", args.cutoff)
    print()
    print("Fitted vapour density:          ", rho_v)
    print("Fitted liquid density:          ", rho_l)
    print("Left interface:                 ", z_left)
    print("Right interface:                ", z_right)
    print()
    print("Apparent fitted width:          ", width)
    print("Naive sharp-interface CWT width:", w_cwt)
    print()
    print(f"Saved fit to: {args.output}")


if __name__ == "__main__":
    main()
