#!/usr/bin/env python3

import sys
import random
import time

small_jobs = [
    'a',
    'b',
]
medium_jobs = [
    'c',
    'd',
]
large_jobs = [
    'e',
    'f'
]

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
    print('    {} <num_jobs> <job_size> [output_filename]'.format(sys.argv[0]))
    print()
    print(HELP_NUM_JOBS)
    print()
    print(HELP_JOB_SIZE)
    print()
    print()
    sys.exit(1)


def parse_args():
    if len(sys.argv) < 3 or len(sys.argv) > 4 :
        usage_and_exit()
    num_jobs = sys.argv[1]
    if not num_jobs.isdigit():
        usage_and_exit()
    job_size = sys.argv[2]
    if job_size not in ['small', 'medium', 'large', 'random']:
        usage_and_exit()
    output_filename = 'example_jobs_00.wl'
    if len(sys.argv) == 4:
        output_filename = sys.argv[3]
    return int(num_jobs), job_size, output_filename


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


num_jobs, job_size, output_filename = parse_args()

if job_size == 'random':
    workload = generate_random_workload()
else:
    workload = generate_workload()


fp = open(output_filename, 'w')
for j in workload:
    print(j)
    fp.write(j+'\n')
fp.close()
print('\nJobs written to {}\n'.format(output_filename))
