#! /bin/bash

cd $HOME/thesis/reg_stokeslets/

module unload miniconda3/latest
module load miniconda3/latest
conda activate rolling

DATA_PATH="data/bd_run/"
SUFFIX=".npz"
LINES=$(cat par-files/bd_runner2107.txt)

for LINE in $LINES
do
    if [ -f "$DATA_PATH$LINE$SUFFIX" ]; then
	continue
    fi
    echo $LINE
done
