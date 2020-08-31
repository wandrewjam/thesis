#!/bin/sh
#SBATCH --time=0:12:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --account=fogelson
#SBATCH --partition=kingspeak
#SBATCH -o slurm-72.out
#SBATCH -e slurm-72.err

cd $HOME/thesis/reg_stokeslets/3d/
module load miniconda3/latest/
conda activate rolling
export OMP_NUM_THREADS=1
python motion_integration.py 72 linux > data/out72.txt
