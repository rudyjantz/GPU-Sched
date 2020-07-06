#!/usr/bin/env python3

import sys
import random

BMARK_PATH = "/home/cc/wo/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda"
DATA_PATH  = "/home/cc/wo/GPU-Sched/Benchmarks/rodinia_cuda_3.1/data"

bmark_cmds = {
    "btree": "time {}/b+tree/b+tree.out file {}/b+tree/mil.txt command {}/b+tree/command.txt &".format(BMARK_PATH, DATA_PATH, DATA_PATH),
    "bprop": "time {}/backprop/backprop 524288 &".format(BMARK_PATH),
    "hot":   "time {}/hotspot3D/3D 512 8 100 {}/hotspot3D/power_512x8 {}/hotspot3D/temp_512x8 output.out &".format(BMARK_PATH, DATA_PATH, DATA_PATH),
    "part":  "time {}/particlefilter/particlefilter_naive -x 128 -y 128 -z 10 -np 1000 &".format(BMARK_PATH),
    "srad":  "time {}/srad/srad_v2/srad 2048 2048 0 127 0 127 0.5 2 &".format(BMARK_PATH)
}

bmark_keys = list(bmark_cmds)


def usage_and_exit():
    print()
    print('Usage:')
    print('    {} <num_procs> [which_bmark]'.format(sys.argv[0]))
    print()
    print('If which_bmark is unspecified, then the generated workload is ' \
          'heterogeneous and random.')
    print('which_bmark can be one of: ')
    print('    btree (for b+tree)')
    print('    bprop (for backprop)')
    print('    hot (for hotspot3D)')
    print('    part (for particlefilter_naive)')
    print('    srad (for srad_v2)')
    print()
    sys.exit(1)



def parse_args():
    if len(sys.argv) > 3 or len(sys.argv) < 2:
        usage_and_exit()
    try:
        num_procs = int(sys.argv[1])
    except:
        usage_and_exit()
    which = ""
    if len(sys.argv) == 3:
        which = sys.argv[2]
        if which not in bmark_cmds:
            usage_and_exit()
    return num_procs, which


def print_preamble(num_procs):
    print("#!/bin/bash")

    print()



def print_conclusion():
    print()

    print('echo "Waiting for jobs to complete..."')
    print("wait")
    print()
    print('echo "Done"')



num_procs, which = parse_args()

print_preamble(num_procs)


if which == "":
    print("# Generated script for {} random, heterogeneous commands\n".format(num_procs))
    for i in range(num_procs):
        bmark = random.choice(bmark_keys)
        print(bmark_cmds[bmark])
else:
    print("# Generated script {} commands for {}\n".format(num_procs, which))
    for i in range(num_procs):
        print(bmark_cmds[which])


print_conclusion()
