#!/bin/bash
#SBATCH --job-name=nodeba3
#SBATCH --mail-user=sc55341
#SBATCH --partition=amd_256
#SBATCH --tasks-per-node=64
#SBATCH --nodes=10
#SBATCH --time=00:15:00
#SBATCH --output=t3.log
#SBATCH --error=t3.err

echo "--------------- PROGRAM OUTPUT ----------------"

julia nodeba.jl

echo "--------------- PROGRAM COMPLETED--------------"

exit
