#!/bin/sh
for filename in *.txt
do
    sed -i "1s/.*/N $1/" $filename
done
