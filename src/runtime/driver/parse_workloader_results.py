#!/usr/bin/env python3
import sys
import statistics
from scipy.stats.mstats import gmean




BASE_PATH = '/home/rudy/wo/gpu/bes-gpu/foo/scripts/cc/results-2020.08.01-7.30pm'
SCHED_LOG_SUF  = 'sched-log'
SCHED_STAT_SUF = 'sched-stats'
WRKLDR_LOG_SUF = 'workloader-log'


DEBUG = False


# TODO: move to input files and support command-line args
workloads = [
    'k80_small_16jobs_0',
    'k80_small_16jobs_1',
    'k80_medium_16jobs_0',
    'k80_medium_16jobs_1',
    'k80_large_16jobs_0',
    'k80_large_16jobs_1',
]




def print_debug(s):
    if DEBUG:
        print(s)


def parse_workloader_log(filename):
    # Examples of what we're looking for:
    #
    # bmark times:
    #   Worker 0: TOTAL_BENCHMARK_TIME 0 2.99345064163208
    #   Worker 1: TOTAL_BENCHMARK_TIME 2 1.2312407493591309
    # total time:
    #   Worker 0: TOTAL_EXPERIMENT_TIME 4.400961637496948
    bmark_times = []
    total_time  = 0
    throughput = 0
    print_debug(filename)
    with open(filename) as f:
        for line in f:
            if 'TOTAL_BENCHMARK_TIME' in line:
                line = line.strip().split()
                bmark_times.append( (int(line[3]), float(line[4])) )
            elif 'TOTAL_EXPERIMENT_TIME' in line:
                line = line.strip().split()
                total_time = float(line[3])

    throughput = len(bmark_times) / float(total_time)

    sorted_bmark_times = sorted(bmark_times)
    for t in sorted_bmark_times:
        print_debug('{} {}'.format(t[0], t[1]))
    print_debug('total {}'.format(total_time))
    print_debug('throughput {}'.format(throughput))
    return sorted_bmark_times, total_time, throughput


def report_total_time_and_throughput(workload):
    print('{}'.format(workload))
    print('scheduler total_time throughput')
    print('sa {} {}'.format(sa_total_time, sa_throughput))
    print('cg {} {}'.format(cg_total_time, cg_throughput))
    print('mgb {} {}'.format(mgb_total_time, mgb_throughput))
    print()


def report_avg_speedup_throughput_improvement_and_job_slowdown():
    cg_speedups  = [ t[0] / t[1] for t in zip(sa_total_times, cg_total_times) ]
    mgb_speedups = [ t[0] / t[1] for t in zip(sa_total_times, mgb_total_times) ]
    avg_cg_speedup  = statistics.mean(cg_speedups)
    avg_mgb_speedup = statistics.mean(mgb_speedups)

    cg_throughput_improvements  = [ t[1] / t[0] for t in zip(sa_throughputs, cg_throughputs) ]
    mgb_throughput_improvements = [ t[1] / t[0] for t in zip(sa_throughputs, mgb_throughputs) ]
    avg_cg_throughput_improvement  = statistics.mean(cg_throughput_improvements)
    avg_mgb_throughput_improvement = statistics.mean(mgb_throughput_improvements)

    assert avg_cg_throughput_improvement  == avg_cg_speedup  # could fail due
    assert avg_mgb_throughput_improvement == avg_mgb_speedup # to rounding error
    print('avg_cg_throughput_improvement {}'.format(avg_cg_throughput_improvement))
    print('avg_mgb_throughput_improvement {}'.format(avg_mgb_throughput_improvement))

    cg_job_slowdowns  = [ t[0][1] / t[1][1] for t in zip(cg_job_times, sa_job_times) ]
    mgb_job_slowdowns = [ t[0][1] / t[1][1] for t in zip(mgb_job_times, sa_job_times) ]
    avg_cg_job_slowdown      = statistics.mean(cg_job_slowdowns)
    avg_mgb_job_slowdown     = statistics.mean(mgb_job_slowdowns)
    geomean_cg_job_slowdown  = gmean(cg_job_slowdowns)
    geomean_mgb_job_slowdown = gmean(mgb_job_slowdowns)
    print('mean_cg_job_slowdown {}'.format(avg_cg_job_slowdown))
    print('geomean_cg_job_slowdown {}'.format(geomean_cg_job_slowdown))
    print('mean_mgb_job_slowdown {}'.format(avg_mgb_job_slowdown))
    print('geomean_mgb_job_slowdown {}'.format(geomean_mgb_job_slowdown))
    print()

    print('normalized_throughput_improvements sa cg mgb')
    for idx, workload in enumerate(workloads):
        print('{} {} {} {}'.format(workload, 1, cg_throughput_improvements[idx],
                                  mgb_throughput_improvements[idx]))
    print('{} {} {} {}'.format('average', 1, avg_cg_throughput_improvement,
                                  avg_mgb_throughput_improvement))
    print()

    print('normalized_job_slowdowns sa cg mgb')
    for idx, workload in enumerate(workloads):
        print('{} {} {} {}'.format(workload, 1, cg_job_slowdowns[idx], mgb_job_slowdowns[idx]))
    print('{} {} {} {}'.format('average', 1, avg_cg_job_slowdown, avg_mgb_job_slowdown))
    print()



sa_total_times  = []
cg_total_times  = []
mgb_total_times = []
sa_throughputs  = []
cg_throughputs  = []
mgb_throughputs = []
sa_job_times    = []
cg_job_times    = []
mgb_job_times   = []
for workload in workloads:
    sa_filename  = '{}/{}.{}.{}'.format(BASE_PATH, workload, 'single-assignment', WRKLDR_LOG_SUF)
    cg_filename  = '{}/{}.{}.{}'.format(BASE_PATH, workload, 'cg', WRKLDR_LOG_SUF)
    mgb_filename = '{}/{}.{}.{}'.format(BASE_PATH, workload, 'mgb', WRKLDR_LOG_SUF)

    sa_times, sa_total_time, sa_throughput    = parse_workloader_log(sa_filename)
    cg_times, cg_total_time, cg_throughput    = parse_workloader_log(cg_filename)
    mgb_times, mgb_total_time, mgb_throughput = parse_workloader_log(mgb_filename)

    sa_job_times.extend(sa_times)
    cg_job_times.extend(cg_times)
    mgb_job_times.extend(mgb_times)
    sa_total_times.append(sa_total_time)
    cg_total_times.append(cg_total_time)
    mgb_total_times.append(mgb_total_time)
    sa_throughputs.append(sa_throughput)
    cg_throughputs.append(cg_throughput)
    mgb_throughputs.append(mgb_throughput)

    #report_total_time_and_throughput(workload)

report_avg_speedup_throughput_improvement_and_job_slowdown()
