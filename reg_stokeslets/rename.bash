#! /bin/bash

cd $HOME/thesis/reg_stokeslets/par-files/
for file in bd_run0[0-9][0-9].txt; do
    git mv -n "$file" "${file:0:6}000${file:6}"
done

for file in bd_runner[

cd $HOME/thesis/reg_stokeslets/data/bd_run/
for file in bd_run03[0-9].*; do
    echo "$file"
done
