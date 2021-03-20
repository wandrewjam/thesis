#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH -o runner2102_1.out
#SBATCH -e runner2102_1.err

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

DATA_PATH="data/bd_run/"
SUFFIX=".npz"
LINES=$(cat par-files/bd_runner2102.txt)

for LINE in $LINES
do
    if [ -f "$DATA_PATH$LINE$SUFFIX" ]; then
	continue
    fi
    echo $LINE >> tmpfile.txt
done

export start=$SECONDS
cat tmpfile.txt | parallel --results output/outdir2102_1 \
-j $TPN python src/3d/binding_expt.py

export duration=$(( SECONDS - start ))
echo "Completed in $duration seconds"

rm tmpfile.txt

wait
