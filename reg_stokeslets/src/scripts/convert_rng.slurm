#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=owner-guest
#SBATCH --partition=notchpeak-guest
#SBATCH -C "chipman|rutter|gnosis|mansfield|uspcasw|cabrera|zpu|farmer|yzhou|zhang|oberg|dtn|fiorella|deo"
#SBATCH -o convert_rng.out
#SBATCH -e convert_rng.err

cd $HOME/thesis/reg_stokeslets/

module unload miniconda3/latest
module load miniconda3/latest
conda activate rolling

python src/3d/convert_rng.py

wait
