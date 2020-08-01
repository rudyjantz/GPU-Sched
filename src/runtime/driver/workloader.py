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
    print_flush('    {} <num_processes> <workload_file>'.format(sys.argv[0]))
    print()
    print_flush('Args:')
    print_flush('  num_processes: The number of worker processes for this driver.')
    print_flush('  workload_file: The relative path and name of the .wl workload file.')
    print()
    print()
    exit(1)




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



def worker_main(q, jobs_processed, wid, experiment_start_time):
    print_flush('Worker {}: Starting'.format(wid))
    while True:
        try:
            idx, benchmark_cmd = q.get(block=True, timeout=1)
            print_flush('Worker {}: {} {}'.format(wid, idx, benchmark_cmd))
            bmark_start_time = time.time()
            run_benchmark(benchmark_cmd)
            bmark_end_time = time.time()
            bmark_time = bmark_end_time - bmark_start_time
            time_since_start = bmark_end_time - experiment_start_time
            print_flush('Worker {}: TOTAL_BENCHMARK_TIME {} {}'.format(wid, idx, bmark_time))
            print_flush('Worker {}: TIME_SINCE_START     {} {}'.format(wid, idx, time_since_start))
            with jobs_processed.get_lock():
                jobs_processed.value += 1
                if jobs_processed.value == jobs_total:
                    experiment_total_time = time.time() - experiment_start_time
                    print_flush('Worker {}: TOTAL_EXPERIMENT_TIME {}'.format(wid, experiment_total_time))
                    print_flush('Worker {}: Exiting normally'.format(wid))
                    return
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


def read_workload_into_q(q, workload_file):
    count = 0
    with open(workload_file) as f:
        for line in f:
            q.put((count, line.strip()))
            count += 1
    return count


def parse_args():
    if len(sys.argv) != 3:
        usage_and_exit()
    num_processes = int(sys.argv[1])
    workload_size = sys.argv[2]
    print_flush('Starting driver')
    print_flush('  num_processes: {}'.format(num_processes))
    print_flush('  workload_size: {}'.format(workload_size))
    return num_processes, workload_size



num_processes, workload_file = parse_args()
q = multiprocessing.Queue()
jobs_total = read_workload_into_q(q, workload_file)
jobs_processed = multiprocessing.Value('i', 0)

experiment_start_time = time.time()
workers = []
for i in range(num_processes):
    p = multiprocessing.Process(target=worker_main, args=(q,jobs_processed,i,experiment_start_time,))
    workers.append(p)
    p.start()

print_flush('Main process: Waiting for workers...')
for w in workers:
    w.join()
print_flush('Main process: Exiting normally.')
