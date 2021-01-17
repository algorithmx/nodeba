#!/bin/bash
#SBATCH --job-name=nodeba5
#SBATCH --mail-user=sc55341
#SBATCH --partition=amd_256
#SBATCH --tasks-per-node=64
#SBATCH --nodes=40
#SBATCH --time=00:25:00
#SBATCH --output=t5.log
#SBATCH --error=t5.err

echo "--------------- PROGRAM OUTPUT ----------------"

julia nodeba.jl

echo "--------------- PROGRAM COMPLETED--------------"

exit
