#!/bin/bash

BASE_FLD=/home/cc/GPU-Sched/src/runtime/driver
SUFFIX=sched-stats


# 16 jobs
#RESULTS_FLD=results-2020.08.12/16jobs
#FILES=(
#    ${BASE_FLD}/${RESULTS_FLD}/p100_50_16jobs_0.single-assignment.2.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_50_16jobs_0.cg.3.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_50_16jobs_0.mgb.10.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_33_16jobs_0.single-assignment.2.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_33_16jobs_0.cg.3.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_33_16jobs_0.mgb.10.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_25_16jobs_0.single-assignment.2.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_25_16jobs_0.cg.2.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_25_16jobs_0.mgb.10.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_16_16jobs_0.single-assignment.2.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_16_16jobs_0.cg.5.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_16_16jobs_0.mgb.10.${SUFFIX}
#)
# 32 jobs
RESULTS_FLD=results-2020.08.12/32jobs
FILES=(
    ${BASE_FLD}/${RESULTS_FLD}/p100_50_32jobs_0.single-assignment.2.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_50_32jobs_0.cg.3.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_50_32jobs_0.mgb.10.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_33_32jobs_0.single-assignment.2.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_33_32jobs_0.cg.2.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_33_32jobs_0.mgb.10.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_25_32jobs_0.single-assignment.2.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_25_32jobs_0.cg.2.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_25_32jobs_0.mgb.10.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_16_32jobs_0.single-assignment.2.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_16_32jobs_0.cg.3.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_16_32jobs_0.mgb.10.${SUFFIX}
)



echo "sched-stats"
for FILE in ${FILES[@]}; do
    echo `basename ${FILE}`
    cat ${FILE}
    echo
done
