#!/bin/bash

if [ $# == 1 ]; then
    NUM_PROCS=$1
else
    NUM_PROCS=1
fi

for i in `seq 1 $NUM_PROCS`; do
    #time ./particlefilter_float -x 128 -y 128 -z 10 -np 1000

    #time ./particlefilter_naive -x 128 -y 128 -z 10 -np 1000 &
    #time ./particlefilter_naive -x 128 -y 128 -z 10 -np 1000000 &
    #time ./particlefilter_naive -x 1024 -y 1024 -z 100 -np 1000 &
    time ./particlefilter_naive -x 128 -y 128 -z 10 -np 10000000 &
done

echo "Waiting for jobs to complete..."
wait

echo "Done"