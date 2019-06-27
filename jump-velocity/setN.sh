#!/bin/sh
for filename in par-files/*.txt
do
    sed "1s/.*/N $1/" $filename > $filename.tmp && mv $filename.tmp $filename
done
