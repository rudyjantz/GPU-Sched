#!/bin/bash


FILES=(
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_small_16jobs_3.single-assignment.2.sched-stats
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_small_16jobs_3.cg.6.sched-stats
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_small_16jobs_3.mgb.24.10.sched-stats
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_medium_16jobs_3.single-assignment.2.sched-stats
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_medium_16jobs_3.cg.4.sched-stats
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_medium_16jobs_3.mgb.16.sched-stats
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_large_16jobs_3.single-assignment.2.sched-stats
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_large_16jobs_3.cg.3.sched-stats
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_large_16jobs_3.mgb.8.sched-stats
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_random_16jobs_3.single-assignment.2.sched-stats
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_random_16jobs_3.cg.6.sched-stats
    /home/cc/GPU-Sched/src/runtime/driver/results-2020.08.10-10.30am/p100_random_16jobs_3.mgb.8.sched-stats
)



echo "sched-stats"
for FILE in ${FILES[@]}; do
    echo `basename ${FILE}`
    cat ${FILE}
    echo
done
