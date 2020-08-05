#!/bin/bash

#set -x

# Structure of this bash script:
#
#   for each scheduling algorithm
#     for each workload
#       Start the scheduler
#       Start the workload driver
#       (Workload driver completes)
#       Stop scheduler
#       Parse workload driver output
#       Move scheduler results to results folder
#

BASE_PATH=/home/rudy/wo/gpu
#BASE_PATH=/home/cc
BEMPS_SCHED_PATH=${BASE_PATH}/GPU-Sched/build/runtime/sched
WORKLOADER_PATH=${BASE_PATH}/GPU-Sched/src/runtime/driver
WORKLOADS_PATH=${BASE_PATH}/GPU-Sched/src/runtime/driver/workloads/test
RESULTS_PATH=results



WORKLOADS=(
    #debug_02.wl
    debug_05.wl
    #k80_small_16jobs_0.wl
    #k80_small_16jobs_1.wl
    #k80_medium_16jobs_0.wl
    #k80_medium_16jobs_1.wl
    #k80_large_16jobs_0.wl
    #k80_large_16jobs_1.wl
    ##random_00.wl
    ##random_01.wl
)

declare -A SCHED_ALG_TO_NUM_PROCS=(
    [single-assignment]=1
    #[single-assignment]=2
    #[cg]=6 # XXX Do not change this without changing JOBS_PER_GPU in sched.cpp
    #[mgb]=12
)

mkdir -p results


for SCHED_ALG in "${!SCHED_ALG_TO_NUM_PROCS[@]}"; do
    for WORKLOAD in ${WORKLOADS[@]}; do
        WORKLOAD_NO_EXT=`basename $WORKLOAD .wl`
        EXPERIMENT_BASENAME=${RESULTS_PATH}/${WORKLOAD_NO_EXT}.${SCHED_ALG}
        NUM_PROCESSES=${SCHED_ALG_TO_NUM_PROCS[$SCHED_ALG]}

        echo "Launching scheduler for ${EXPERIMENT_BASENAME}"
        ${BEMPS_SCHED_PATH}/bemps_sched ${SCHED_ALG} \
          &> ${EXPERIMENT_BASENAME}.sched-log &
        SCHED_PID=$!
        echo "Scheduler is running with pid ${SCHED_PID}"

        echo "Launching workoader for ${EXPERIMENT_BASENAME} with ${NUM_PROCESSES} processes"
        ${WORKLOADER_PATH}/workloader.py \
          ${NUM_PROCESSES} \
          ${WORKLOADS_PATH}/${WORKLOAD}  \
          &> ${EXPERIMENT_BASENAME}.workloader-log &
        WORKLOADER_PID=$!
        echo "Workloader is running with pid ${WORKLOADER_PID}"

        echo "Waiting for workloader to complete"
        wait ${WORKLOADER_PID}

        echo "Workloader done"
        echo "Killing scheduler"
        kill -2 ${SCHED_PID}
        sleep 1 # maybe a good idea before moving sched-stats.out
        mv ./sched-stats.out ${EXPERIMENT_BASENAME}.sched-stats

        echo "Completed experiment for ${EXPERIMENT_BASENAME}"
    done
done

echo "All experiments complete"
echo "Exiting normally"
