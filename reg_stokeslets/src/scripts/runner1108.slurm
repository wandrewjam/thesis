#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --account=owner-guest
#SBATCH --partition=notchpeak-guest
#SBATCH -C "[(rogersa|farmer|cabrera|gnosis|dtn|mansfield|oberg|yrao|chipman|anderegg|fiorella|facelli|uspcasw|pezzolesi|gompert|rutter)&c32]"
#SBATCH -o runner1108.out
#SBATCH -e runner1108.err

# find number of threads for OpenMP (adapted from CHPC code)
# find number of MPI tasks per node
export TPN=$(echo $SLURM_TASKS_PER_NODE | cut -f 1 -d \()
# find number of CPU cores per node
export PPN=$(echo $SLURM_JOB_CPUS_PER_NODE | cut -f 1 -d \()
(( THREADS = PPN / TPN ))
export OMP_NUM_THREADS=$THREADS

echo "$TPN"
echo "$PPN"
echo "$OMP_NUM_THREADS"

cd $HOME/thesis/reg_stokeslets/

module unload miniconda3/latest
module load miniconda3/latest
conda activate rolling

export start=$SECONDS
cat par-files/bd_runner1108.txt | parallel --results output/outdir1108 \
-j $TPN python src/3d/binding_expt.py

export duration=$(( SECONDS - start ))
echo "Completed in $duration seconds"

wait
