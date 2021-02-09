#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --account=fogelson
#SBATCH --partition=kingspeak
#SBATCH -C "c16"
#SBATCH -o runner10.out
#SBATCH -e runner10.err

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

cd $HOME/thesis/reg_stokeslets/

module unload miniconda3/latest
module load miniconda3/latest
conda activate rolling

export start=$SECONDS
cat par-files/bd_runner10.txt | parallel --results output/outdir10 \
-j $TPN python src/3d/binding_expt.py

export duration=$(( SECONDS - start ))
echo "Completed in $duration seconds"

wait
