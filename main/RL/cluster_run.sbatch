#!/bin/bash

#SBATCH -J osimrl
#SBATCH -o slurm.out
#SBATCH -e slurm.err
#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH --mem 1G # memory pool for all cores
#SBATCH --mail-user=anas.jnini@epfl.ch
#SBATCH --mail-type=fail




echo "running on hosts: $SLURM_NODELIST"
echo "running on $SLURM_NNODES nodes."
echo "running $SLURM_NTASKS tasks."
echo "current working directory is `pwd`"

# setup simbody path (Please change SIMBODY_HOME accordingly!)
export SIMBODY_HOME=/shared-dev/stanev/opensim-core/dependencies/simbody
export Simbody_DIR=$SIMBODY_HOME/lib/cmake/simbody
export PATH=$SIMBODY_HOME/libexec/simbody:$PATH
export LD_LIBRARY_PATH=$SIMBODY_DIR/lib:$LD_LIBRARY_PATH

# setup opensim-core path (Please change OPENSIM_HOME accordingly!)
export OPENSIM_HOME=/shared-dev/stanev/opensim-core/
export OpenSim_DIR=$OPENSIM_HOME/lib/cmake/OpenSim
export OpenSim_DIR=$OPENSIM_HOME
export LD_LIBRARY_PATH=$OPENSIM_HOME/lib:$LD_LIBRARY_PATH
export PATH=$OPENSIM_HOME/bin:$PATH

# activate opensim-evn (Please change PATH accordingly!)
source /shared-dev/stanev/opensim-env/bin/activate

# run script and create SLURM_NTASKS subprocesses
srun -n 1 python ppo_arm_flexor.py
echo "Program finished"