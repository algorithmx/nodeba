#!/bin/bash
#SBATCH --job-name=nodeba6
#SBATCH --mail-user=sc55341
#SBATCH --partition=amd_256
#SBATCH --tasks-per-node=64
#SBATCH --nodes=20
#SBATCH --time=00:15:00
#SBATCH --output=t6.log
#SBATCH --error=t6.err

echo "--------------- PROGRAM OUTPUT ----------------"

julia nodeba.jl

echo "--------------- PROGRAM COMPLETED--------------"

exit
