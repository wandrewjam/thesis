#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH -o bd_test044.out
#SBATCH -e bd_test044.err

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
python binding_expt.py bd_run044 > bd_run044.out

export duration=$(( SECONDS - start ))
echo "Completed in $duration seconds"
