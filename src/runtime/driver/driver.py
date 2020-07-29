#!/usr/bin/env python3
import multiprocessing
import subprocess
import sys
import time
import queue
import os



def print_flush(s):
    print(s, flush=True)


def usage_and_exit():
    print()
    print()
    print_flush('Usage:')
    print_flush('    {} <num_processes> <workload_size> <job_size> <variation>'.format(sys.argv[0]))
    print()
    print_flush('Args:')
    print_flush('  num_processes: The number of worker processes for this driver.')
    print()
    print_flush('  workload_size: The size of the workload, i.e. how long it will\n' \
          '  run, which is based off the number of jobs that it has.')
    print()
    print_flush('  job_size: The size of each job within the workload.')
    print()
    print_flush('  variation: A number from 00-99 for the specific file/variation.')
    print()
    print_flush(HELP_NUM_PROCESSES)
    print()
    print_flush(HELP_WORKLOAD_SIZE)
    print()
    print_flush(HELP_JOB_SIZE)
    print()
    print_flush(HELP_VARIATION)
    print()
    print()
    exit(1)


HELP_NUM_PROCESSES = """num_processes
The number of worker processes for this driver determines the number of
simultaneous benchmarks that are running in the expierment. This must be
carefully chosen for each particular experiment, along with the scheduler,
the number of GPU devices, the number of CPU cores, etc."""

HELP_WORKLOAD_SIZE = """workload_size
The workload_size represents how long the job will run, which is based off the
number of jobs that it has. These are determined experimentally by running
randomized workloads that fall within some range. The workload_size must
be one of the following values:
  - debug
  - test
  - ref
They carry the following meanings:
  - debug: manually written with just a few jobs
  - test: roughly 1-3 minutes
  - ref: roughly 10-15 minutes"""

HELP_JOB_SIZE = """job_size
The job_size represents the sizes of the jobs that are part of a workload.
It must be one of the following values:
  - debug
  - small
  - medium
  - large
  - random
They carry the following meanings:
  - debug: manually written and containing any size
  - small: 100% of the jobs are small
  - medium: 100% of the jobs are medium
  - large: 100% of the jobs are large
  - random: ~33% of the jobs are of each size (small, medium, or large)
Here, "small", "medium", "large" refer to the max memory footprint
possible of a given job. Job sizes:
  - small: all kernels < 10% GPU memory
  - medium: no large kernels, and at least 1 kernel with 10-50% GPU memory
  - large: at least 1 kernel with > 50% GPU memory"""

HELP_VARIATION = """variation
The variation just refers to a specific file for the given workload and job sizes.
For example, for small workloads using medium-sized jobs, we might have a
few randomized inputs that we want to experiment with. The variation is the
number corresponding to a specific one."""





def run_benchmark(cmd):
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            shell=True)
    o, e = proc.communicate()
    rc = proc.returncode
    if rc != 0:
        print_flush(rc)
        print_flush(e.decode('utf-8'))
    else:
        print_flush('suc')
    print_flush(o.decode('utf-8'))
    #print_flush(e.decode('utf-8'))



def worker_main(q, jobs_processed, wid):
    print_flush('Worker {}: Starting'.format(wid))
    while True:
        try:
            benchmark_cmd = q.get(block=True, timeout=1)
            print_flush('Worker {}: {}'.format(wid, benchmark_cmd))
            run_benchmark(benchmark_cmd)
            jobs_processed.value = jobs_processed.value + 1
            print_flush('Worker {}: done with benchmark'.format(wid))
        except queue.Empty:
            if jobs_processed.value == jobs_total:
                print_flush('Worker {}: Worklist is empty.'.format(wid))
                break
            print_flush('Worker {}: Worklist is empty. Retrying get().'.format(wid))
            print_flush('Worker {}: jobs_processed.value is {}'.format(wid, jobs_processed.value))
        except Exception as e:
            print_flush('Worker {}: Unexpected error when fetching from worklist. ' \
                  'Raising.'.format(wid))
            raise
    print_flush('Worker {}: Exiting normally'.format(wid))


def read_workload_into_q(q, workload_size, job_size, variation):
    count = 0
    filename = './workloads/{}/{}_{}.wl'.format(workload_size, job_size, variation)
    with open(filename) as f:
        for line in f:
            q.put(line.strip())
            count += 1
    return count


def parse_args():
    if len(sys.argv) != 5:
        usage_and_exit()
    num_processes = int(sys.argv[1])
    workload_size = sys.argv[2]
    if workload_size not in ['debug', 'test', 'ref']:
        usage_and_exit()
    job_size = sys.argv[3]
    if job_size not in ['debug', 'small', 'medium', 'large', 'random']:
        usage_and_exit()
    variation = sys.argv[4]
    print_flush('Starting driver')
    print_flush('  num_processes: {}'.format(num_processes))
    print_flush('  workload_size: {}'.format(workload_size))
    print_flush('  job_size: {}'.format(job_size))
    print_flush('  variation: {}'.format(variation))
    return num_processes, workload_size, job_size, variation




num_processes, workload_size, job_size, variation = parse_args()
q = multiprocessing.Queue()
jobs_total = read_workload_into_q(q, workload_size, job_size, variation)
jobs_processed = multiprocessing.Value('i', 0)

workers = []
for i in range(num_processes):
    p = multiprocessing.Process(target=worker_main, args=(q,jobs_processed,i,))
    workers.append(p)
    p.start()

print_flush('Main process: Waiting for workers...')
for w in workers:
    w.join()
print_flush('Main process: Exiting normally.')
