#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8g
#SBATCH -C "c20"
#SBATCH --account=fogelson
#SBATCH --partition=kingspeak
#SBATCH -o slurm-73.out
#SBATCH -e slurm-73.err

# find number of threads for OpenMP (adapted from CHPC code)
# find number of MPI tasks per node
export TPN=$(echo $SLURM_TASKS_PER_NODE | cut -f 1 -d \()
# find number of CPU cores per node
export PPN=$(echo $SLURM_JOB_CPUS_PER_NODE | cut -f 1 -d \()
(( THREADS = PPN / TPN ))
export OMP_NUM_THREADS=$THREADS

echo "$OMP_NUM_THREADS"

cd $HOME/thesis/reg_stokeslets/3d/
module load miniconda3/latest/
conda activate rolling
python motion_integration.py 73 linux > data/out73.txt
