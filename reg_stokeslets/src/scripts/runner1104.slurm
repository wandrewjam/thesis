#!/bin/bash
#SBATCH --time=46:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --account=fogelson
#SBATCH --partition=lonepeak
#SBATCH -C "c32"
#SBATCH -o runner1104.out
#SBATCH -e runner1104.err

# find number of threads for OpenMP (adapted from CHPC code)
# find number of MPI tasks per node
export TPN=$(echo $SLURM_TASKS_PER_NODE | cut -f 1 -d \()
# find number of CPU cores per node
export PPN=$(echo $SLURM_JOB_CPUS_PER_NODE | cut -f 1 -d \()
(( THREADS = PPN / TPN ))
export OMP_NUM_THREADS=$THREADS
# export OMP_NUM_THREADS=1

echo "$TPN"
echo "$PPN"
echo "$OMP_NUM_THREADS"

cd $HOME/thesis/reg_stokeslets/

module unload miniconda3/latest
module load miniconda3/latest
conda activate rolling

export start=$SECONDS

cat par-files/bd_runner1104.txt | parallel --results output/outdir1104 \
-j $TPN python src/3d/binding_expt.py

export duration=$(( SECONDS - start ))
echo "Completed in $duration seconds"

wait
