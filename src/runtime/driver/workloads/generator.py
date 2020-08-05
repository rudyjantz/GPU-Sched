#!/usr/bin/env python3

import sys
import random
import time

BASE_PATH = '/home/rudy/wo/gpu'
BASE_PATH = '/home/cc/'
RODINIA_BMARK_PATH = BASE_PATH+'/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda'
RODINIA_DATA_PATH  = BASE_PATH+'/GPU-Sched/Benchmarks/rodinia_cuda_3.1/data'


GPU_TO_MEM = {
    'gtx1080':  8 * 1024 * 1024 * 1024, #  8 GB
    'k80':     12 * 1024 * 1024 * 1024, # 12 GB
    'p100':    16 * 1024 * 1024 * 1024, # 16 GB
}


# all_jobs is a list of tuples:
#   (max kernel memory size, benchmark command that produces it)
all_jobs = [
    # ~20 MB
    (  20523432, 'b+tree/b+tree.out file {}/b+tree/mil.txt ' \
                 'command {}/b+tree/command.txt'.format(RODINIA_DATA_PATH,
                                                        RODINIA_DATA_PATH)),
    (  48000000, 'particlefilter/particlefilter_naive ' \
                 '-x 128 -y 128 -z 10 -np 1000000'),
    (  50331648, 'hotspot/hotspot 2048 2 2 {}/hotspot/gt_temp_2048 ' \
                 '{}/hotspot/gt_power_2048 ' \
                 '{}/hotspot/output.out'.format(RODINIA_DATA_PATH,
                                                RODINIA_DATA_PATH,
                                                RODINIA_DATA_PATH)),
    # ~ 100 MB
    ( 100663296, 'srad/srad_v2/srad 2048 2048 0 127 0 127 0.5 2'),
    ( 210834320, 'b+tree/b+tree.out file {}/b+tree/mil_gt.txt command ' \
                 '{}/b+tree/command_gt.txt'.format(RODINIA_DATA_PATH,
                                                   RODINIA_DATA_PATH)),
    ( 201326592, 'hotspot3D/3D 512 64 1000 {}/hotspot3D/power_512x64 ' \
                 '{}/hotspot3D/temp_512x64 ' \
                 '{}/hotspot3D/output.out'.format(RODINIA_DATA_PATH,
                                                  RODINIA_DATA_PATH,
                                                  RODINIA_DATA_PATH)),
    ( 402653184, 'srad/srad_v2/srad 4096 4096 0 127 0 127 0.5 2'),
    ( 589299988, 'backprop/backprop 4194304'),
    # ~ 1 GB
    (1176502548, 'backprop/backprop 8388608'),
    (1610612736, 'srad/srad_v2/srad 8192 8192 0 127 0 127 0.5 2'),
    (2350907668, 'backprop/backprop 16777216'),
    (4699717908, 'backprop/backprop 33554432'),
    (6442450944, 'srad/srad_v2/srad 16384 16384 0 127 0 127 0.5 2'),
    (7856000000, 'lavaMD/lavaMD -boxes1d 100'),
    (9397338388, 'backprop/backprop 67108864'),
    #(25769803776, 'srad/srad_v2/srad 32768 32768 0 127 0 127 0.5 2'), # seems halved to 12GB
]
small_jobs = []
medium_jobs = []
large_jobs = []



job_size_to_jobs = {
    'small': small_jobs,
    'medium': medium_jobs,
    'large': large_jobs,
}

job_bufs = [
    small_jobs,
    medium_jobs,
    large_jobs
]


HELP_GPU = """gpu
This field represents the GPU you intend to use for the workload you're
generating. The size of the workload ultimately depends on the capability of
the GPU (i.e. a "small" workload for a powerful GPU may be "large" for a weak
GPU). This field must be one of "gtx0180", "k80", or "p100"."""

HELP_NUM_JOBS = """num_jobs
The number of jobs in the workload."""

HELP_JOB_SIZE = """job_size
The job_size represents the sizes of the jobs that are part of a workload.
It must be one of the following values:
  - small
  - medium
  - large
  - random
They carry the following meanings:
  - small: 100% of the jobs are small
  - medium: 100% of the jobs are medium
  - large: 100% of the jobs are large
  - random: ~33% of the jobs are of each size (small, medium, or large).
            The jobs from each group are randomly chosen.
            The order of jobs over the entire workload is random.
Here, "small", "medium", "large" refer to the max memory footprint
possible for a given job. Job sizes:
  - small: all kernels < 10% GPU memory
  - medium: no large kernels, and at least 1 kernel with 10-50% GPU memory
  - large: at least 1 kernel with > 50% GPU memory"""

def usage_and_exit():
    print()
    print()
    print('Usage:')
    print('    {} <gpu> <job_size> <num_jobs> [output_filename]'.format(sys.argv[0]))
    print()
    print(HELP_GPU)
    print()
    print(HELP_JOB_SIZE)
    print()
    print(HELP_NUM_JOBS)
    print()
    print()
    sys.exit(1)


def parse_args():
    if len(sys.argv) < 4 or len(sys.argv) > 5 :
        usage_and_exit()
    gpu = sys.argv[1]
    if gpu not in GPU_TO_MEM:
        usage_and_exit()
    job_size = sys.argv[2]
    if job_size not in ['small', 'medium', 'large', 'random']:
        usage_and_exit()
    num_jobs = sys.argv[3]
    if not num_jobs.isdigit():
        usage_and_exit()
    output_filename = 'example_jobs_00.wl'
    if len(sys.argv) == 5:
        output_filename = sys.argv[4]
    return gpu, job_size, int(num_jobs), output_filename


def construct_job_arrays(which_gpu):
    print('Constructing the job arrays...')
    total_avail_mem  = GPU_TO_MEM[which_gpu]
    small_threshold  = total_avail_mem * .1
    medium_threshold = total_avail_mem * .5
    for j in all_jobs:
        B = j[0]
        if B < small_threshold:
            small_jobs.append(j[1])
        elif B < medium_threshold:
            medium_jobs.append(j[1])
        else:
            large_jobs.append(j[1])
    print('small jobs:')
    for j in small_jobs:
        print('  {}'.format(j))
    print('medium jobs:')
    for j in medium_jobs:
        print('  {}'.format(j))
    print('large jobs:')
    for j in large_jobs:
        print('  {}'.format(j))
    print()



# Generates a workload with 1/3 small, 1/3 medium, and 1/3 large jobs.
# The jobs from each group are randomly chosen.
# The order of jobs over the entire workload is random.
def generate_random_workload():
    workload = []
    random.seed(int(time.time()))
    for i in range(num_jobs):
        # Go to next list of jobs, e.g. small_jobs or medium_jobs
        jobs = job_bufs[i % len(job_bufs)]
        # Pick a job from that list
        job = jobs[random.randint(0, len(jobs)-1)]
        # Append 
        workload.append(job)
    random.shuffle(workload)
    return workload


def generate_workload():
    workload = []
    jobs = job_size_to_jobs[job_size]
    for i in range(num_jobs):
        job = jobs[random.randint(0, len(jobs)-1)]
        workload.append(job)
    return workload


gpu, job_size, num_jobs, output_filename = parse_args()
construct_job_arrays(gpu)

if job_size == 'random':
    workload = generate_random_workload()
else:
    workload = generate_workload()


print('Generating the workload...')
fp = open(output_filename, 'w')
for j in workload:
    full_cmd = RODINIA_BMARK_PATH + '/' + j
    print(full_cmd)
    fp.write(full_cmd+'\n')
fp.close()
print()
print('Jobs written to {}'.format(output_filename))
print()
