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

#BASE_PATH=/home/rudy/wo/gpu
BASE_PATH=/home/cc
BEMPS_SCHED_PATH=${BASE_PATH}/GPU-Sched/build/runtime/sched
WORKLOADER_PATH=${BASE_PATH}/GPU-Sched/src/runtime/driver
WORKLOADS_PATH=${BASE_PATH}/GPU-Sched/src/runtime/driver/workloads/test
RESULTS_PATH=results



WORKLOADS=(
    all_jobs_0.wl
    #debug_02.wl
    #debug_05.wl
    #debug_06.wl
    #debug_07.wl
    #k80_small_16jobs_0.wl
    #k80_small_16jobs_1.wl
    #k80_medium_16jobs_0.wl
    #k80_medium_16jobs_1.wl
    #k80_large_16jobs_0.wl
    #k80_large_16jobs_1.wl
    ##random_00.wl
    ##random_01.wl
)

SINGLE_ASSIGNMENT_ARGS_ARR=(
    1
    #2
)
CG_ARGS_ARR=(
    6 # XXX Do not change this without changing SCHED_JOBS_PER_GPU in sched.cpp
    16 # XXX Do not change this without changing SCHED_JOBS_PER_GPU in sched.cpp
)
MGB_ARGS_ARR=(
    6
    16.10 # num procs and threshold K
    48.10 # num procs and threshold K
)


declare -A SCHED_ALG_TO_ARGS_ARR=(
    [single-assignment]="SINGLE_ASSIGNMENT_ARGS_ARR"
    #[cg]="CG_ARGS_ARR"
    #[mgb]="MGB_ARGS_ARR"
)




mkdir -p results


for WORKLOAD in ${WORKLOADS[@]}; do
    for SCHED_ALG in "${!SCHED_ALG_TO_ARGS_ARR[@]}"; do

        #echo ${SCHED_ALG}
        ARGS_ARR_STR=${SCHED_ALG_TO_ARGS_ARR[$SCHED_ALG]}
        #echo $ARGS_ARR_STR
        eval ARGS_ARR=\${${ARGS_ARR_STR}[@]}
        for ARGS in ${ARGS_ARR[@]}; do
            #echo $ARGS
            WORKLOAD_NO_EXT=`basename $WORKLOAD .wl`
            #ARGS=${SCHED_ALG_TO_ARGS[$SCHED_ALG]}
            EXPERIMENT_BASENAME=${RESULTS_PATH}/${WORKLOAD_NO_EXT}.${SCHED_ALG}.${ARGS}

            # yet another hack: cg needs to know jobs-per-cpu. And we need to
            # be able to control it from this bash driver. So pass it along.
            SCHED_ARGS=""
            if [ "${SCHED_ALG}" == "cg" ]; then
                SCHED_ARGS=${ARGS}
            fi

            echo "Launching scheduler for ${EXPERIMENT_BASENAME}"
            ${BEMPS_SCHED_PATH}/bemps_sched ${SCHED_ALG} ${SCHED_ARGS} \
              &> ${EXPERIMENT_BASENAME}.sched-log &
            SCHED_PID=$!
            echo "Scheduler is running with pid ${SCHED_PID}"

            # FIXME Adding a hacky sleep. We have an unsafe assumption, though we
            # have yet to see a problem manifest: The scheduler needs to initialize
            # the shared memory (bemps_sched_init()) before benchmarks run and try
            # to open it (bemps_init()). When using mgbd (mgb with dynamic job
            # pressure), the workloader itself could also fail without a sufficient
            # delay here.
            sleep 1

            echo "Launching workoader for ${EXPERIMENT_BASENAME}"
            ${WORKLOADER_PATH}/workloader.py \
              ${WORKLOADS_PATH}/${WORKLOAD}  \
              ${SCHED_ALG} \
              ${ARGS} \
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
done

echo "All experiments complete"
echo "Exiting normally"
