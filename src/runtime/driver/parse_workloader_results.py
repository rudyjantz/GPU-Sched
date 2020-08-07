#!/usr/bin/env python3
import sys
import statistics
from scipy.stats.mstats import gmean
from pprint import pprint




#BASE_PATH = '/home/rudy/wo/gpu/bes-gpu/foo/scripts/cc/results-2020.08.01-7.30pm'
BASE_PATH = '/home/rudy/wo/gpu/GPU-Sched/src/runtime/driver/results'

SCHED_LOG_SUF  = 'sched-log'
SCHED_STAT_SUF = 'sched-stats'
WRKLDR_LOG_SUF = 'workloader-log'


DEBUG = False


# TODO: move to input files and support command-line args
workloads = [
    #'k80_small_16jobs_0',
    #'k80_small_16jobs_1',
    #'k80_medium_16jobs_0',
    #'k80_medium_16jobs_1',
    #'k80_large_16jobs_0',
    #'k80_large_16jobs_1',
    #'k80_large_16jobs_1',
    'debug_05'
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
    # beacon timing (if it's turned on):
    #   12949252580699578 _bemps_dump_stats: count of beacon times: 1
    #   12949252580707683 _bemps_dump_stats: min beacon time (ns): 103063473
    #   12949252580708435 _bemps_dump_stats: max beacon time (ns): 103063473
    #   12949252580709246 _bemps_dump_stats: avg beacon time (ns): 1.03063e+08
    #   12949252580740675 _bemps_dump_stats: count of free times: 1
    #   12949252580741427 _bemps_dump_stats: min free time (ns): 68178
    #   12949252580742148 _bemps_dump_stats: max free time (ns): 68178
    #   12949252580742839 _bemps_dump_stats: avg free time (ns): 68178
    COUNT_BEACON_STR = 'count of beacon times:'
    MIN_BEACON_STR = 'min beacon time (ns):'
    MAX_BEACON_STR = 'max beacon time (ns):'
    AVG_BEACON_STR = 'avg beacon time (ns):'
    COUNT_FREE_STR = 'count of free times:'
    MIN_FREE_STR = 'min free time (ns):'
    MAX_FREE_STR = 'max free time (ns):'
    AVG_FREE_STR = 'avg free time (ns):'

    # beacon times
    bt = {
        'min_beacon': 1<<65,
        'max_beacon': 0,
        'avg_beacon': 0,
        'min_free': 1<<65,
        'max_free': 0,
        'avg_free': 0,
    }
    num_beacons = 0
    num_frees = 0
    beacon_avgs = []
    free_avgs = []

    bmark_times = []
    total_time  = 0
    throughput = 0
    print_debug(filename)
    with open(filename) as f:
        count_beacon_flag = 0 # to help with asserts
        count_free_flag   = 0
        for line in f:
            if 'TOTAL_BENCHMARK_TIME' in line:
                line = line.strip().split()
                bmark_times.append( (int(line[3]), float(line[4])) )
            elif 'TOTAL_EXPERIMENT_TIME' in line:
                total_time = float(line.strip().split()[3])
            elif COUNT_BEACON_STR in line:
                count_beacon = int(line.strip().split()[6])
                num_beacons += count_beacon
                count_beacon_flag += 1
            elif MIN_BEACON_STR in line:
                min_beacon = int(line.strip().split()[6])
                if min_beacon < bt['min_beacon']:
                    bt['min_beacon'] = min_beacon
            elif MAX_BEACON_STR in line:
                max_beacon = int(line.strip().split()[6])
                if max_beacon > bt['max_beacon']:
                    bt['max_beacon'] = max_beacon
            elif AVG_BEACON_STR in line:
                avg_beacon = float(line.strip().split()[6])
                count_beacon_flag -= 1
                assert count_beacon_flag == 0
                beacon_avgs.append((count_beacon, avg_beacon))
            elif COUNT_FREE_STR in line:
                count_free = int(line.strip().split()[6])
                num_frees += count_free
                count_free_flag += 1
            elif MIN_FREE_STR in line:
                min_free = int(line.strip().split()[6])
                if min_free < bt['min_free']:
                    bt['min_free'] = min_free
            elif MAX_FREE_STR in line:
                max_free = int(line.strip().split()[6])
                if max_free > bt['max_free']:
                    bt['max_free'] = max_free
            elif AVG_FREE_STR in line:
                avg_free = float(line.strip().split()[6])
                count_free_flag -= 1
                assert count_free_flag == 0
                free_avgs.append((count_free, avg_free))

    for count, avg in beacon_avgs:
        bt['avg_beacon'] += avg * count / num_beacons
    for count, avg in free_avgs:
        bt['avg_free'] += avg * count / num_frees

    throughput = len(bmark_times) / float(total_time)

    sorted_bmark_times = sorted(bmark_times)
    for t in sorted_bmark_times:
        print_debug('{} {}'.format(t[0], t[1]))
    print_debug('total {}'.format(total_time))
    print_debug('throughput {}'.format(throughput))
    return sorted_bmark_times, total_time, throughput, bt


def report_total_time_and_throughput(workload):
    print('{}'.format(workload))
    print('scheduler total_time throughput')
    print('sa {} {}'.format(sa_total_time, sa_throughput))
    print('cg {} {}'.format(cg_total_time, cg_throughput))
    print('mgb {} {}'.format(mgb_total_time, mgb_throughput))
    print()


def report_beacon_times(sa_bcn_times, cg_bcn_times, mgb_bcn_times):
    print('sa beacon times')
    for k, v in sa_bcn_times.items():
        print('{} {}'.format(k, v))
    print()
    print('cg beacon times')
    for k, v in cg_bcn_times.items():
        print('{} {}'.format(k, v))
    print()
    print('mgb beacon times')
    for k, v in mgb_bcn_times.items():
        print('{} {}'.format(k, v))
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

    sa_times, sa_total_time, sa_throughput, sa_bcn_times     = parse_workloader_log(sa_filename)
    cg_times, cg_total_time, cg_throughput, cg_bcn_times     = parse_workloader_log(cg_filename)
    mgb_times, mgb_total_time, mgb_throughput, mgb_bcn_times = parse_workloader_log(mgb_filename)

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

report_beacon_times(sa_bcn_times, cg_bcn_times, mgb_bcn_times)
report_avg_speedup_throughput_improvement_and_job_slowdown()
