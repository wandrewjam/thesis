#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --account=fogelson
#SBATCH --partition=kingspeak
#SBATCH -o slurm-73.out
#SBATCH -e slurm-73.err

cd $HOME/thesis/reg_stokeslets/3d/
module load miniconda3/latest/
conda activate rolling
export OMP_NUM_THREADS=1
python motion_integration.py 73 linux
