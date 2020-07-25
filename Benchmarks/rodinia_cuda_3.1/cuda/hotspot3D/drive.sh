#!/bin/bash

if [ $# == 1 ]; then
    NUM_PROCS=$1
else
    NUM_PROCS=1
fi

for i in `seq 1 $NUM_PROCS`; do
    #time ./3D 512 8 100 ../../data/hotspot3D/power_512x8 ../../data/hotspot3D/temp_512x8 output.out &
    #time ./3D 512 8 1000 ../../data/hotspot3D/power_512x8 ../../data/hotspot3D/temp_512x8 output.out &
    #time ./3D 512 16 100 ../../data/hotspot3D/power_512x16 ../../data/hotspot3D/temp_512x16 output.out &
    #time ./3D 512 16 1000 ../../data/hotspot3D/power_512x16 ../../data/hotspot3D/temp_512x16 output.out &
    #time ./3D 512 32 100 ../../data/hotspot3D/power_512x32 ../../data/hotspot3D/temp_512x32 output.out &
    #time ./3D 512 32 1000 ../../data/hotspot3D/power_512x32 ../../data/hotspot3D/temp_512x32 output.out &
    #time ./3D 512 64 100 ../../data/hotspot3D/power_512x64 ../../data/hotspot3D/temp_512x64 output.out &
    time ./3D 512 64 1000 ../../data/hotspot3D/power_512x64 ../../data/hotspot3D/temp_512x64 output.out &
done

echo "Waiting for jobs to complete..."
wait

echo "Done"
