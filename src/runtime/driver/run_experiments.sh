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


BEMPS_SCHED_PATH=/home/cc/GPU-Sched/build/runtime/sched
WORKLOADER_PATH=/home/cc/GPU-Sched/src/runtime/driver
WORKLOADS_PATH=/home/cc/GPU-Sched/src/runtime/driver/workloads/test
RESULTS_PATH=results



WORKLOADS=(
    debug_02.wl
    #small_16jobs_0.wl
    #small_16jobs_1.wl
    #medium_16jobs_0.wl
    #medium_16jobs_1.wl
    #large_16jobs_0.wl
    #large_16jobs_1.wl
    ##random_00.wl
    ##random_01.wl
)

declare -A SCHED_ALG_TO_NUM_PROCS=(
    [single-assignment]=2
    [cg]=6
    [mgb]=6
)



for SCHED_ALG in "${!SCHED_ALG_TO_NUM_PROCS[@]}"; do
    for WORKLOAD in ${WORKLOADS[@]}; do
        WORKLOAD_NO_EXT=`basename $WORKLOAD .wl`
        EXPERIMENT_BASENAME=${RESULTS_PATH}/${WORKLOAD_NO_EXT}.${SCHED_ALG}
        NUM_PROCESSES=${SCHED_ALG_TO_NUM_PROCS[$SCHED_ALG]}

        echo "Launching scheduler for ${EXPERIMENT_BASENAME}"
        ${BEMPS_SCHED_PATH}/bemps_sched ${SCHED_ALG} \
          &> ${EXPERIMENT_BASENAME}.sched-log &
        SCHED_PID=$!

        echo "Launching workoader for ${EXPERIMENT_BASENAME} with ${NUM_PROCESSES} processes"
        ${WORKLOADER_PATH}/workloader.py \
          ${NUM_PROCESSES} \
          ${WORKLOADS_PATH}/${WORKLOAD}  \
          &> ${EXPERIMENT_BASENAME}.workloader-log
        WORKLOADER_PID=$!

        echo "Waiting for workload to complete"
        wait ${WORKLOADER_PID}

        echo "Workload done."
        echo "Killing scheduler with pid "${SCHED_PID}
        kill -2 ${SCHED_PID}
        sleep 1 # maybe a good idea before moving sched-stats.out
        mv ${BEMPS_SCHED_PATH}/sched-stats.out ${EXPERIMENT_BASENAME}.sched-stats

        echo "Completed experiment for ${EXPERIMENT_BASENAME}. See ${RESULTS_PATH}"
    done
done

echo "All experiments complete"
echo "Exiting normally"
