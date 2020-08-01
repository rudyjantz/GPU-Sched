#!/usr/bin/env python3

import sys

#
# Examples of what we're looking for:
#
# bmark times:
#   Worker 0: TOTAL_BENCHMARK_TIME 0 2.99345064163208
#   Worker 1: TOTAL_BENCHMARK_TIME 2 1.2312407493591309
# total time:
#   Worker 0: TOTAL_EXPERIMENT_TIME 4.400961637496948
#



if len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    filename = 'outch'

bmark_times = []
total_time = 0


with open(filename) as f:
    for line in f:
        if 'TOTAL_BENCHMARK_TIME' in line:
            line = line.strip().split()
            bmark_times.append((line[3], line[4]))
        elif 'TOTAL_EXPERIMENT_TIME' in line:
            line = line.strip().split()
            total_time = line[3]

throughput = len(bmark_times) / float(total_time)

sorted_bmark_times = sorted(bmark_times)
for t in sorted_bmark_times:
    print('{} {}'.format(t[0], t[1]))
print('total {}'.format(total_time))
print('throughput {}'.format(throughput))
