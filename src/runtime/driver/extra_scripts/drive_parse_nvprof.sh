#!/bin/bash

set -x
set -e


#RESULTS_FLD=../results-nvprof-2020.08.12
#RESULTS_FLD=../results-nvprof-2020.08.13
#RESULTS_FLD=../results/nvprof
RESULTS_FLD=../results/hand-picked/nvprof


#NUM_JOBS=(
#    16jobs
#    32jobs
#)
NUM_JOBS=(
    8jobs
    16jobs
    32jobs
    64jobs
)

#MIXES=(
#    50
#    33
#    25
#    16
#)
MIXES=(
    picked
)

for NJ in ${NUM_JOBS[@]}; do
    for MIX in ${MIXES[@]}; do

        # FIXME: can dump the raw profile with a command like this:
        #./parse_nvprof.py ${RESULTS_FLD}/p100_${MIX}_${NJ}_1.single-assignment.2.workloader-log \
        #  &> ${NJ}-${MIX}-sa.txt

        #./parse_nvprof.py \
        #  ${RESULTS_FLD}/p100_${MIX}_${NJ}_1.single-assignment.2.workloader-log \
        #  ${RESULTS_FLD}/p100_${MIX}_${NJ}_1.mgb.10.workloader-log \
        #  &> ${NJ}-${MIX}-result.txt

        #./parse_nvprof.py \
        #  ${RESULTS_FLD}/p100_${MIX}_${NJ}_1.single-assignment.4.workloader-log \
        #  ${RESULTS_FLD}/p100_${MIX}_${NJ}_1.mgb_basic.16.workloader-log \
        #  &> ${NJ}-${MIX}-result.txt

        #./parse_nvprof.py \
        #  ${RESULTS_FLD}/p100_${MIX}_${NJ}_1.single-assignment.4.workloader-log \
        #  ${RESULTS_FLD}/p100_${MIX}_${NJ}_1.mgb.16.workloader-log \
        #  &> ${NJ}-${MIX}-result.txt



        #./parse_nvprof.py \
        #  ${RESULTS_FLD}/hand_${MIX}_${NJ}_2.single-assignment.2.workloader-log \
        #  ${RESULTS_FLD}/hand_${MIX}_${NJ}_2.mgb_basic.10.workloader-log \
        #  &> ${NJ}-${MIX}-result.txt

        #./parse_nvprof.py \
        #  ${RESULTS_FLD}/hand_${MIX}_${NJ}_2.single-assignment.4.workloader-log \
        #  ${RESULTS_FLD}/hand_${MIX}_${NJ}_2.mgb_basic.16.workloader-log \
        #  &> ${NJ}-${MIX}-result.txt

        ./parse_nvprof.py \
          ${RESULTS_FLD}/hand_${MIX}_${NJ}_2.single-assignment.4.workloader-log \
          ${RESULTS_FLD}/hand_${MIX}_${NJ}_2.mgb.16.workloader-log \
          &> ${NJ}-${MIX}-result.txt


    done
done
