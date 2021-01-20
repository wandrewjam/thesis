#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH -o runner01.out
#SBATCH -e runner01.err

# find number of threads for OpenMP (adapted from CHPC code)
# find number of MPI tasks per node
export TPN=$(echo $SLURM_TASKS_PER_NODE | cut -f 1 -d \()
# find number of CPU cores per node
export PPN=$(echo $SLURM_JOB_CPUS_PER_NODE | cut -f 1 -d \()
(( THREADS = PPN / TPN ))
# export OMP_NUM_THREADS=$THREADS
export OMP_NUM_THREADS=1

echo "$TPN"
echo "$PPN"
echo "$OMP_NUM_THREADS"

cd $HOME/thesis/reg_stokeslets/3d/

module unload miniconda3/latest
module load miniconda3/latest
conda activate rolling

export start=$SECONDS
cat par-files/bd_runner01.txt | parallel --results outdir \
-j $TPN python binding_expt.py

export duration=$(( SECONDS - start ))
echo "Completed in $duration seconds"

wait
