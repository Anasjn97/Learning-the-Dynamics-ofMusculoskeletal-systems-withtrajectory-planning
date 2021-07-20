#!/bin/bash
#SBATCH -J ./test
#SBATCH -o ./test.out
#SBATCH -e ./test.err

#SBATCH -J sinergia

#SBATCH -N 1 # number of nodes

#SBATCH -n 40 # number of cores

#SBATCH --mem 28G  # memory pool for all cores

#SBATCH -t 00-23:30:00 # time (D-HH:MM)

source activate opensim-rl

echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Current working directory is `pwd`"

srun -n 1 python ./ppo_arm_flexor.py -s 10000000 -l -p -o DDPG/ -d

echo "Program finished"
