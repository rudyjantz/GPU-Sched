#!/bin/bash

if [ $# == 1 ]; then
    NUM_PROCS=$1
else
    NUM_PROCS=1
fi

for i in `seq 1 $NUM_PROCS`; do
    #time ./b+tree.out file ../../data/b+tree/mil.txt command ../../data/b+tree/command.txt &
    time ./b+tree.out file ../../data/b+tree/mil_gt.txt command ../../data/b+tree/command_gt.txt &
done

echo "Waiting for jobs to complete..."
wait

echo "Done"
