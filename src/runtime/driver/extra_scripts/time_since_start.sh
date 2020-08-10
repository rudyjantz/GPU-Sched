#!/bin/bash


FILES=(
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_small_16jobs_3.single-assignment.2.workloader-log
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_small_16jobs_3.cg.6.workloader-log
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_small_16jobs_3.mgb.24.10.workloader-log
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_medium_16jobs_3.single-assignment.2.workloader-log
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_medium_16jobs_3.cg.4.workloader-log
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_medium_16jobs_3.mgb.16.workloader-log
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_large_16jobs_3.single-assignment.2.workloader-log
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_large_16jobs_3.cg.3.workloader-log
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_large_16jobs_3.mgb.8.workloader-log
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_random_16jobs_3.single-assignment.2.workloader-log
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_random_16jobs_3.cg.6.workloader-log
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_random_16jobs_3.mgb.8.workloader-log
)



echo "TIME_SINCE_START"
echo "Sortedj-based-on-job-index"
for FILE in ${FILES[@]}; do
    echo `basename ${FILE}`
    grep "TIME_SINCE_START" ${FILE} | awk '{print $4" "$5}' | sort -n
done

echo
echo
echo
echo

echo "Sorted-based-on-time-since-start"
for FILE in ${FILES[@]}; do
    echo `basename ${FILE}`
    grep "TIME_SINCE_START" ${FILE} | awk '{print $5" "$4}' | sort -n | awk '{print $2" "$1}'
done
