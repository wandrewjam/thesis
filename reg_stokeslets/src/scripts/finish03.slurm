#!/bin/bash
#SBATCH --time=25:00:00
#SBATCH --nodes=3
#SBATCH --ntasks=12
#SBATCH -C 'c16'
#SBATCH --account=fogelson
#SBATCH --partition=kingspeak
#SBATCH -o finish03.out
#SBATCH -e finish03.err

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

DATA_PATH="data/bd_run/"
SUFFIX=".npz"
LINES=$(cat par-files/bd_runner1103.txt)

for LINE in $LINES
do
    if [ -f "$DATA_PATH$LINE$SUFFIX" ]; then
	continue
    fi
    echo $LINE >> tmpfile03.txt
done

export start=$SECONDS
cat tmpfile03.txt | parallel --results output/outdir_finish03 \
-j $TPN python src/3d/binding_expt.py

export duration=$(( SECONDS - start ))
echo "Completed in $duration seconds"

wait

rm tmpfile03.txt
