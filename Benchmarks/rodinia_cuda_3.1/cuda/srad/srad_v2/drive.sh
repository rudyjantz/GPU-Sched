#!/bin/bash

if [ $# == 1 ]; then
    NUM_PROCS=$1
else
    NUM_PROCS=1
fi

for i in `seq 1 $NUM_PROCS`; do
    time ./srad 2048 2048 0 127 0 127 0.5 2 &
done

echo "Waiting for jobs to complete..."
wait

echo "Done"
