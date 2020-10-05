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
#BASE_PATH=/home/cc
BASE_PATH=/home/ubuntu
BEMPS_SCHED_PATH=${BASE_PATH}/GPU-Sched/build/runtime/sched
WORKLOADER_PATH=${BASE_PATH}/GPU-Sched/src/runtime/driver
#WORKLOADS_PATH=${BASE_PATH}/GPU-Sched/src/runtime/driver/workloads/test
#WORKLOADS_PATH=${BASE_PATH}/GPU-Sched/src/runtime/driver/workloads/ppopp21
WORKLOADS_PATH=${BASE_PATH}/GPU-Sched/src/runtime/driver/workloads/ppopp21-volta
#WORKLOADS_PATH=${BASE_PATH}/GPU-Sched/src/runtime/driver/workloads/ppopp21-rebuttal/p100
#WORKLOADS_PATH=${BASE_PATH}/GPU-Sched/src/runtime/driver/workloads/ppopp21-rebuttal/v100
RESULTS_PATH=results



WORKLOADS=(
    #all_jobs_0.wl
    #all_jobs_1.wl
    #all_jobs_leftover_1.wl
    #debug_02.wl
    #debug_05.wl
    #debug_06.wl
    #debug_07.wl
    #debug_08.wl
    #k80_small_16jobs_0.wl
    #k80_small_16jobs_1.wl
    #k80_medium_16jobs_0.wl
    #k80_medium_16jobs_1.wl
    #k80_large_16jobs_0.wl
    #k80_large_16jobs_1.wl
    ##random_00.wl
    ##random_01.wl
    #p100_small_16jobs_3.wl
    #p100_medium_16jobs_3.wl
    #p100_large_16jobs_3.wl
    #p100_random_16jobs_3.wl
    #p100_small_16jobs_4.wl  # sa.2, cg.6, mgb.24.10
    #p100_medium_16jobs_4.wl # sa.2, cg.4, mgb.16
    #p100_large_16jobs_4.wl # sa.2, cg.3, mgb.8
    #p100_large_16jobs_5.wl # sa.2, cg.3, mgb.8
    # ppopp21 WORKLOADS_PATH
    #p100_16_16jobs_0.wl
    #p100_25_16jobs_0.wl
    #p100_33_16jobs_0.wl
    #p100_50_16jobs_0.wl
    #p100_16_32jobs_0.wl
    #p100_25_32jobs_0.wl
    #p100_33_32jobs_0.wl
    #p100_50_32jobs_0.wl
    # nvprof versions of _0.wl
    # assume root b/f getting nvprof number; cc needed it; I believe aws didn't:
    #   sudo su
    #   source /home/cc/setenv.sh
    #   export PATH=/usr/local/cuda-10.1/bin/:$PATH
    #   ./run_experiments.sh
    #p100_16_16jobs_1.wl # _1.wl is nvprof on cmds 1, 2, 3, ...
    #p100_25_16jobs_1.wl
    #p100_33_16jobs_1.wl
    #p100_50_16jobs_1.wl
    #p100_16_32jobs_1.wl # _1.wl has insufficient mem for 32 jobs when using mps
    #p100_25_32jobs_1.wl
    #p100_33_32jobs_1.wl
    #p100_50_32jobs_1.wl
    #p100_16_32jobs_2.wl # _2.wl is nvprof on cmds 1, 3, 5, ...
    #p100_16_32jobs_3.wl # _3.wl is nvprof on cmds 2, 4, 6, ...
    #p100_25_32jobs_2.wl
    #p100_25_32jobs_3.wl
    #p100_33_32jobs_2.wl
    #p100_33_32jobs_3.wl
    #p100_50_32jobs_2.wl
    #p100_50_32jobs_3.wl
    # ppopp21 extra last-minute runs
    #v100_50_64jobs_0.wl
    #v100_50_128jobs_0.wl
    #v100_50_256jobs_0.wl
    #v100_50_512jobs_0.wl
    #v100_50_1024jobs_0.wl
    # ppopp21 rebuttal WORKLOADS_PATH
    #p100_v2-90_32jobs_0.wl # lost the original workload
    #p100_v2-90_64jobs_0.wl # lost the original workload
    #hand_picked_8jobs_1.wl # hand-picked _0.wl is starter file, _1.wl is shuffled
    #hand_picked_16jobs_1.wl
    #hand_picked_32jobs_1.wl
    #hand_picked_64jobs_1.wl
    #hand_picked_8jobs_2.wl # _2.wl is with nvprof
    #hand_picked_16jobs_2.wl
    #hand_picked_32jobs_2.wl
    #hand_picked_64jobs_2.wl
    # _10.wl is semi-convenient naming. These are similar to the original _0.wl
    # files. But now we modified them so no jobs are over 10GB. (sched.cpp
    # was modified so v100 had 10GB max, as well)
    v100_50_32jobs_10.wl
    v100_50_64jobs_10.wl
    v100_50_128jobs_10.wl
    #v100_50_256jobs_10.wl
)

SINGLE_ASSIGNMENT_ARGS_ARR=(
    #1
    #2 # <-- 2-GPU system
    4 # <-- 4-GPU system
)
CG_ARGS_ARR=(
    #2 # <-- Don't use for 4-GPU system. Don't use for 2-GPU system unless sanity checking. This is equivalent to single-assignment.
    #3 # <-- Don't use for 4-GPU system
    4 # <-- Don't use for 4-GPU system unless sanity checking. This is equivalent to single-assignment.
    #5
    #6
    #7
    #8
    #9
    #10
    #11
    #12
    #24
)
MGB_ARGS_ARR=(
    #6
    #8
    #10 # <-- ultimately used for ppopp21 2xp100 results
    #12
    #14
    #16 # <-- ultimately used for ppopp21 4xv100 results
    #18
    #20
    #24
    32
    #24.10 # num procs . max jobs waiting for GPU
    #48.10 # num procs . max jobs waiting for GPU
)


declare -A SCHED_ALG_TO_ARGS_ARR=(
    [single-assignment]="SINGLE_ASSIGNMENT_ARGS_ARR"
    #[cg]="CG_ARGS_ARR"
    #[mgb_basic]="MGB_ARGS_ARR"
    #[mgb_simple_compute]="MGB_ARGS_ARR"
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
