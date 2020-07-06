#!/bin/bash

if [ $# == 1 ]; then
    NUM_PROCS=$1
else
    NUM_PROCS=1
fi

for i in `seq 1 $NUM_PROCS`; do
    time ./backprop 524288 &
done

echo "Waiting for jobs to complete..."
wait

echo "Done"
