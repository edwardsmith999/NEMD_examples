#!/bin/bash

#SBATCH --partition=standard
#SBATCH --job-name=Intrinsic_interface_lammps_job
#SBATCH --time=0:30:0
#SBATCH --account=tc075
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --qos=standard
#SBATCH --export=none

# Set the number of threads to 1
export OMP_NUM_THREADS=1

module load lammps

# Launch the parallel job
srun -n 32 --hint=nomultithread --distribution=block:block lmp -var rcut 0 -var dump 10 -var T 0.82 -var Lz_t 48 -var Lx_t 24 -var Ly_t 24 -var tslab 12 -var rho_l 0.746 -var rho_v 0.001 -var fmt lammps -var nprod 100000 < liquid_vapour.in
