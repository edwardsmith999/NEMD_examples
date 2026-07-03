#!/usr/bin/env python3

import argparse
p = argparse.ArgumentParser()
p.add_argument("trajectory", nargs="?", default="day2_traj.lammpstrj")
p.add_argument("-a", "--alpha", type=float, default=0.8)
p.add_argument("-l", "--layers", type=int, default=5)
p.add_argument("--start", type=int, default=0)
p.add_argument("--stop", type=int, default=-1)
p.add_argument("--stride", type=int, default=1)
p.add_argument("-o", "--output", default="itim")
args = p.parse_args()

import numpy as np
import MDAnalysis as mda
import pytim
from pytim.observables import Profile

# tqdm displays nice progress bars
try:
    from tqdm.auto import tqdm
except:
    def tqdm(x):x



# The trajectory is in reduced LJ units.  cluster_cut removes vapour atoms
# from the cluster used to define the interface; keep it fixed for the school.
u = mda.Universe(args.trajectory,
                 format="LAMMPSDUMP",
                 atom_style="id type x y z",
                 convert_units=True, # we keep reduced (LJ) units
                )
# The LAMMPS trjectory format is a bit loose, and does not include atom names/types,
# so here we need to add them manually.
#
# Notice that the xyz format has the atom names but does
# not include the box edges in the same file, so that option requires some
# workaround as well. Extra (optional) dump fixes include more standard trajectory
# formats like dcd or xtc.
u.add_TopologyAttr("types", ['Ar'] * len(u.atoms))

# Define the parameters for the interfacial atoms identification and run the ITIM code
# on the first frame. Pytim will run the analysis automatically whenever a new frame is
# loaded
inter = pytim.ITIM(u, group=u.atoms,                    # Universe and group to analyse.
                       alpha=args.alpha,                # Radius of the probe sphere
                       max_layers=args.layers,          # How many molecular layers we want to analyse
                       molecular=False,                 # this is necessary for lammps as it writes the whole system as a unique residue
                       cluster_cut=3,                   # Large because when using cluster_threshold
                                                        # the cluster cutoff is used to compute the local
                                                        # density.
                       cluster_threshold_density='auto' # When cluster_threshold_density is used, a DBSCAN-based
                                                        # clustering is used to separate liquid from vapour
                                                        # with the value 'auto', the local density threshold is
                                                        # chosen automatically (see 10.1039/d3sm00176h)
                      )
# We prepare several profiles, two (one in the global reference frame, one intrinsic) for all atoms in the system
# and a list of profiles in the global reference frame for each of the molecular layers.
rho = Profile(direction="z")
rho_intrinsic = Profile(interface=inter) # when interface is passed, compute the intrinsic profile
rho_layers = [Profile(direction="z") for _ in range(args.layers)]


# We run through our frames and sample the profile
for ts in tqdm(u.trajectory[args.start : args.stop : args.stride]):
    inter.center() # this removes com fluctuations of the slab
    rho.sample(u.atoms)
    rho_intrinsic.sample(u.atoms)
    for layer, prof in enumerate(rho_layers, start=1):
        prof.sample(u.atoms[u.atoms.layers == layer])


# a convenience function to extract the mid of the bin
def get(profile):
    lo, hi, val = profile.get_values(binwidth=0.01)
    return 0.5 * (lo + hi), val # we return the middle of each bin

z, r_all = get(rho)
r_layer = [get(prof)[1] for prof in rho_layers]
np.savetxt(args.output + "_lab.dat", np.c_[z, r_all, *r_layer],
           header="z rho_all " + " ".join(f"rho_layer_{i}" for i in range(1, args.layers + 1)))

s, r_int = get(rho_intrinsic)
np.savetxt(args.output + "_intrinsic.dat", np.c_[s, r_int],
           header="distance_from_ITIM_surface rho_intrinsic")
