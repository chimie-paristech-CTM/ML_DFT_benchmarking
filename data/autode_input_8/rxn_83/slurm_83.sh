#!/bin/bash
#SBATCH --job-name=ade_83
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --qos=qos_cpu-t3
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread  # Disable hyperthreading
#SBATCH --output=ade_83_%j.out
#SBATCH --error=ade_83_%j.err
#SBATCH --account=qev@cpu
module purge
module load xtb/6.4.1
module load gaussian/g16-revC01
module load python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export GAUSS_SCRDIR=$JOBSCRATCH
conda activate autode
export AUTODE_LOG_LEVEL=INFO
export AUTODE_LOG_FILE=ade_83.log
python3 ade_83.py 
python3 irc_validation.py 
