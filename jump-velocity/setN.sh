#!/bin/sh
for filename in par-files/*.txt
do
    gsed -i "1s/.*/N $1/" $filename
done
