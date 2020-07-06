#!/bin/bash

arr=(
    64
    128
    256
    512
    1024
    2048
    4096
)

for x in ${arr[@]}; do
    f=${x}_hetero.sh
    ./gen_workload.py $x > $f
    chmod +x $f
done
