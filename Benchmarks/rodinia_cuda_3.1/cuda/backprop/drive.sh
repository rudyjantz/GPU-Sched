#!/bin/bash

if [ $# == 1 ]; then
    NUM_PROCS=$1
else
    NUM_PROCS=1
fi

for i in `seq 1 $NUM_PROCS`; do
    #time ./backprop 524288 &
    #time ./backprop 1048576 # 2**20, this throws an error: "bpnn kernel error:
    #                        # invalid configuration argument"
    #time ./backprop 2097152 # 2**21, throw same error
    #time ./backprop 4194304 # 2**22
    #time ./backprop 8388608 # 23 -- takes ~1GB
    #time ./backprop 16777216 # 24 -- ~2GB
    #time ./backprop 33554432 # 25 -- ~4.5GB
    time ./backprop 67108864 # 26 -- ~4.7GB
    #time ./backprop 1073741824 # 2**30, this gets killed

done

echo "Waiting for jobs to complete..."
wait

echo "Done"
