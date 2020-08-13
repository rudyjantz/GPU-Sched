#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <ctype.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <map>
#include <list>
#include <queue>

#include "bemps.hpp"

//#define BEMPS_SCHED_DEBUG

#define SCHED_DEFAULT_BATCH_SIZE 1
#define SCHED_VECTOR_BATCH_SIZE 10
#define SCHED_MGB_BATCH_SIZE    10

// Based on nvidia-smi
//const long P100_PCIE_TOTAL_MEM_KB = 16280L * 1024;
const long TESLA_K80_TOTAL_MEM_KB = 11441L * 1024;
const long GTX_1080_TOTAL_MEM_KB = 8116L * 1024;

const long P100_PCIE_TOTAL_MEM_KB = 14000L * 1024;
const long V100_SXM2_TOTAL_MEM_KB = 14000L * 1024;

#if defined(GPU_RES_SLOP)
#define NUM_GPUS 1
#define INIT_GPU_RES()                               \
  GPU_RES[0].mem_B = GTX_1080_TOTAL_MEM_KB * 1024;   \
  GPU_RES[0].cores = 2560;                           \
  memset(gpu_res_in_use, 0, sizeof(gpu_res_in_use)); \
  dump_gpu_res("slop");

#elif defined(GPU_RES_CC_2)
#define NUM_GPUS 2
#define INIT_GPU_RES()                               \
  GPU_RES[0].mem_B = TESLA_K80_TOTAL_MEM_KB * 1024;  \
  GPU_RES[0].cores = 2496;                           \
  GPU_RES[1].mem_B = TESLA_K80_TOTAL_MEM_KB * 1024;  \
  GPU_RES[1].cores = 2496;                           \
  memset(gpu_res_in_use, 0, sizeof(gpu_res_in_use)); \
  dump_gpu_res("cc_2");

#elif defined(GPU_RES_CC_2P)
#define NUM_GPUS 2
#define INIT_GPU_RES()                               \
  GPU_RES[0].mem_B = P100_PCIE_TOTAL_MEM_KB * 1024;  \
  GPU_RES[0].cores = 3584;                           \
  GPU_RES[1].mem_B = P100_PCIE_TOTAL_MEM_KB * 1024;  \
  GPU_RES[1].cores = 3584;                           \
  memset(gpu_res_in_use, 0, sizeof(gpu_res_in_use)); \
  dump_gpu_res("cc_2p");

#elif defined(GPU_RES_CC_4)
#define NUM_GPUS 4
#define INIT_GPU_RES()                                                     \
  assert(0 &&                                                              \
         "Finish defining GPU_RES_CC_4. It currently holds dummy values"); \
  GPU_RES[0].mem_B = 100000;                                               \
  GPU_RES[0].cores = 2500;                                                 \
  GPU_RES[1].mem_B = 100000;                                               \
  GPU_RES[1].cores = 2500;                                                 \
  GPU_RES[2].mem_B = 100000;                                               \
  GPU_RES[2].cores = 2500;                                                 \
  GPU_RES[3].mem_B = 100000;                                               \
  GPU_RES[3].cores = 2500;                                                 \
  memset(gpu_res_in_use, 0, sizeof(gpu_res_in_use));                       \
  dump_gpu_res("cc_4");

#elif defined(GPU_RES_AWS_4)
#define NUM_GPUS 4
#define INIT_GPU_RES()                                                     \
  GPU_RES[0].mem_B = V100_SXM2_TOTAL_MEM_KB * 1024;                        \
  GPU_RES[0].cores = 5120;                                                 \
  GPU_RES[1].mem_B = V100_SXM2_TOTAL_MEM_KB * 1024;                        \
  GPU_RES[1].cores = 5120;                                                 \
  GPU_RES[2].mem_B = V100_SXM2_TOTAL_MEM_KB * 1024;                        \
  GPU_RES[2].cores = 5120;                                                 \
  GPU_RES[3].mem_B = V100_SXM2_TOTAL_MEM_KB * 1024;                        \
  GPU_RES[3].cores = 5120;                                                 \
  memset(gpu_res_in_use, 0, sizeof(gpu_res_in_use));                       \
  dump_gpu_res("aws_4");

#else
#define NUM_GPUS 0
#define INIT_GPU_RES() \
  assert(0 &&          \
         "Must define a valid GPU resource (e.g. GPU_RES_SLOP, \
               GPU_RES_CC_2, etc.)");
#endif

#ifdef BEMPS_SCHED_DEBUG
#define BEMPS_SCHED_LOG(str)                                          \
  do {                                                                \
    std::cout << get_time_ns() << " " << __FUNCTION__ << ": " << str; \
    std::cout.flush();                                                \
  } while (0)
#else
#define BEMPS_SCHED_LOG(str) \
  do {                       \
  } while (0)
#endif


const int SCHED_ALIVE_COUNT_MAX = 30; // roughly 5s, assuming 100ms timer and 0 beacons
int SCHED_ALIVE_COUNT = 0;
#define ALIVE_MSG()                              \
  do {                                                \
    if (SCHED_ALIVE_COUNT == SCHED_ALIVE_COUNT_MAX) { \
      BEMPS_SCHED_LOG("alive\n");                     \
      SCHED_ALIVE_COUNT = 0;                          \
    } else {                                          \
      SCHED_ALIVE_COUNT++;                            \
    }                                                 \
  } while(0)


#define SCHED_NUM_STOPWATCHES 1
typedef enum {
  SCHED_STOPWATCH_AWAKE = 0 // time the scheduler spends awake and processing
} sched_stopwatch_e;


typedef enum {
  SCHED_ALG_ZERO_E = 0,
  SCHED_ALG_ROUND_ROBIN_E,
  SCHED_ALG_ROUND_ROBIN_BEACONS_E,
  SCHED_ALG_VECTOR_E,
  SCHED_ALG_SINGLE_ASSIGNMENT_E,
  SCHED_ALG_CG_E,
  SCHED_ALG_MGB_E
} sched_alg_e;

struct gpu_res_s {
  long mem_B;
  long cores;
};

typedef struct {
  int num_beacons;
  int num_frees;
  int max_len_boomers;
  int max_age;
  int max_observed_batch_size;
} sched_stats_t;


bemps_stopwatch_t sched_stopwatches[SCHED_NUM_STOPWATCHES];


//
// single-assignment scheduler
//
std::map<pid_t, int> pid_to_device_id;
std::vector<int> avail_device_ids;


//
// C:G scheduler
//
typedef struct {
    int device_id;
    int count;
} device_id_count_t;
std::map<pid_t, device_id_count_t *> pid_to_device_id_counts;
std::list<device_id_count_t *> avail_device_id_counts;
int CG_JOBS_PER_GPU = 0; // set by command-line args



bemps_shm_t *bemps_shm_p;
sched_stats_t stats;

std::list<bemps_shm_comm_t *> boomers;

sched_alg_e which_scheduler;
int max_batch_size;

struct gpu_res_s GPU_RES[NUM_GPUS];
struct gpu_res_s gpu_res_in_use[NUM_GPUS];

void usage_and_exit(char *prog_name) {
  printf("\n");
  printf("Usage:\n");
  printf("    %s [which_scheduler] [jobs_per_gpu]\n", prog_name);
  printf("\n");
  printf(
      "    which_scheduler is one of zero, round-robin,\n"
      "    round-robin-beacons, vector, single-assignment, cg, or mgb\n"
      "    \n"
      "    jobs_per_gpu is required and only valid for cg; it is an int that\n"
      "    specifies the maximum number of jobs that can be run a GPU\n");
  printf("\n");
  printf("\n");
  exit(1);
}

static inline long long get_time_ns(void) {
  struct timespec ts = {0};
  if (clock_gettime(CLOCK_MONOTONIC, &ts)) {
    fprintf(stderr, "ERROR: get_time_ns failed\n");
    return 0;
  }
  return (((long long)ts.tv_sec * 1000000000L) + (long long)(ts.tv_nsec));
}

static inline void dump_gpu_res(const char *which_env) {
  int i;
  BEMPS_SCHED_LOG("Running with the following GPU resources (" << which_env
                                                               << ")\n");
  for (i = 0; i < NUM_GPUS; i++) {
    BEMPS_SCHED_LOG("  GPU " << i << "\n");
    BEMPS_SCHED_LOG("  mem_B: " << GPU_RES[i].mem_B << "\n");
    BEMPS_SCHED_LOG("  cores: " << GPU_RES[i].cores << "\n");
  }
}

static inline void set_wakeup_time_ns(struct timespec *ts_p) {
  struct timespec now;

  // Must use CLOCK_REALTIME when passing to pthread_cond_timedwait
  if (clock_gettime(CLOCK_REALTIME, &now)) {
    fprintf(stderr, "ERROR: set_wakeup_time_ns failed\n");
    return;
  }

  // won't overflow
  //BEMPS_SCHED_LOG("BEMP_SCHED_TIMEOUT_NS: " << BEMPS_SCHED_TIMEOUT_NS << "\n");
  ts_p->tv_nsec = now.tv_nsec + BEMPS_SCHED_TIMEOUT_NS;
  ts_p->tv_sec = now.tv_sec + ts_p->tv_nsec / 1000000000UL;
  ts_p->tv_nsec = ts_p->tv_nsec % 1000000000UL;

  //BEMPS_SCHED_LOG("now s:  " << now.tv_sec << "\n");
  //BEMPS_SCHED_LOG("ts_p s: " << ts_p->tv_sec << "\n");
}

void consume_beacon(bemps_beacon_t *beacon_p) {
  // TODO: Need to implement the scheduling algorithm, which ultimately
  // decides how we want to consume the beacons. One idea might be to
  // first group the beacons (while we advance the tail to the head),
  // and then assign device IDs in a separate loop.
  BEMPS_SCHED_LOG("mem_B(" << beacon_p->mem_B << ")"
                           << " cores(" << beacon_p->cores << ")"
                           << "\n");
}

void dump_stats(void) {
#define STATS_LOG(str)    \
  do {                    \
    stats_file << str;    \
    stats_file.flush();   \
    BEMPS_SCHED_LOG(str); \
  } while (0)
  std::ofstream stats_file;
  bemps_stopwatch_t *sa;

  stats_file.open("sched-stats.out");
  sa = &sched_stopwatches[SCHED_STOPWATCH_AWAKE];

  BEMPS_SCHED_LOG("Caught interrupt. Exiting.\n");
  STATS_LOG("num_beacons: " << stats.num_beacons << "\n");
  STATS_LOG("num_frees: " << stats.num_frees << "\n");
  STATS_LOG("max_len_boomers: " << stats.max_len_boomers << "\n");
  STATS_LOG("max_age: " << stats.max_age << "\n");
  STATS_LOG("max_batch_size: " << max_batch_size << "\n");
  STATS_LOG("max_observed_batch_size: "<<stats.max_observed_batch_size<< "\n");
// clockwatch is only active if BEMPS_STATS is defined
#ifdef BEMPS_STATS
  STATS_LOG("count-of-awake-times: " << sa->n << "\n");
  STATS_LOG("min-awake-time-(ns): " << sa->min << "\n");
  STATS_LOG("max-awake-time-(ns): " << sa->max << "\n");
  STATS_LOG("avg-awake-time-(ns): " << sa->avg << "\n");
#endif

  stats_file.close();
}

void sigint_handler(int unused) {
  BEMPS_SCHED_LOG("Caught interrupt. Exiting.\n");
  dump_stats();
  exit(0);
}

struct MemFootprintCompare {
  bool operator()(const bemps_shm_comm_t *lhs, const bemps_shm_comm_t *rhs) {
    // sort in increasing order. priority queue .top() and pop() will
    // therefore return the largest values
    // return lhs->beacon.mem_B < rhs->beacon.mem_B;

    // sort in decreasing order. iterating use vector begin() and end().
    // return lhs->beacon.mem_B > rhs->beacon.mem_B;

    // sort in decreasing order. iterate with index.
    return lhs->beacon.mem_B > rhs->beacon.mem_B;

    // sort in increasing order. iterate with index. use back() and pop_back()
    // return lhs->beacon.mem_B < rhs->beacon.mem_B;
  }
} mem_footprint_compare;

struct AvailDevicesCompare {
  bool operator()(const device_id_count_t *lhs, const device_id_count_t *rhs) {
    // sort in decreasing order. iterate with index.
    return lhs->count > rhs->count;
  }
} avail_devices_compare;



// Our custom scheduler, multi-GPU with beacons
void sched_mgb(void) {
  int tmp_dev_id;
  int *head_p;
  int *tail_p;
  int *jobs_running_on_gpu;
  int *jobs_waiting_on_gpu;
  int assigned;
  struct timespec ts;
  int boomers_len;
  int i;
  bemps_shm_comm_t *comm;
  int batch_size;

  head_p = &bemps_shm_p->gen->beacon_q_head;
  tail_p = &bemps_shm_p->gen->beacon_q_tail;
  jobs_running_on_gpu = &bemps_shm_p->gen->jobs_running_on_gpu;
  jobs_waiting_on_gpu = &bemps_shm_p->gen->jobs_waiting_on_gpu;

  while (1) {
    set_wakeup_time_ns(&ts);

    // wait until we get a signal or time out
    pthread_mutex_lock(&bemps_shm_p->gen->lock);
    // TODO spurious wakeups ? shouldn't make a big difference to wake up
    // randomly from time to time before the batch is ready
    pthread_cond_timedwait(&bemps_shm_p->gen->cond, &bemps_shm_p->gen->lock,
                           &ts);
    pthread_mutex_unlock(&bemps_shm_p->gen->lock);
    bemps_stopwatch_start(&sched_stopwatches[SCHED_STOPWATCH_AWAKE]);

    ALIVE_MSG();

    // First loop: Catch the scheduler's tail back up with the beacon
    // queue's head. If we see a free-beacon, then reclaim that resource.
    //BEMPS_SCHED_LOG("*head_p: " << (*head_p) << "\n");
    //BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
    batch_size = 0;
    while (*tail_p != *head_p) {
      BEMPS_SCHED_LOG("*head_p: " << (*head_p) << "\n");
      BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");

      comm = &bemps_shm_p->comm[*tail_p];
      while (comm->state != BEMPS_BEACON_STATE_BEACON_FIRED_E) {
        // TODO probably want to track a stat for this case
        BEMPS_SCHED_LOG("WARNING: Scheduler hit a beacon before FIRED "
                        << "was set. (Not a bug, but unless we're "
                        << "flooded with beacons, this should be rare."
                        << "\n");
        BEMPS_SCHED_LOG("WARNING: *tail_p: " << (*tail_p) << "\n");
        // FIXME sync shouldn't hurt, but may not help?
        __sync_synchronize();
      }

      if (comm->exit_flag) {
        BEMPS_SCHED_LOG("seeing exit flag\n");
        comm->exit_flag = 0;
      } else {
        assert(comm->beacon.mem_B);
        BEMPS_SCHED_LOG("First loop seeing mem_B: " << comm->beacon.mem_B
                                                    << "\n");
        if (comm->beacon.mem_B < 0) {
          BEMPS_SCHED_LOG("Received free-beacon for pid " << comm->pid << "\n");
          stats.num_frees++;
          tmp_dev_id = comm->sched_notif.device_id;
          // Add (don't subtract), because mem_B is negative already
          long tmp_bytes_to_free = comm->beacon.mem_B;
          long tmp_cores_to_free = comm->beacon.cores;
          BEMPS_SCHED_LOG("Freeing " << tmp_bytes_to_free << " bytes "
                          << "from device " << tmp_dev_id << "\n");
          BEMPS_SCHED_LOG("Freeing " << tmp_cores_to_free << " cores "
                          << "from device " << tmp_dev_id << "\n");
          gpu_res_in_use[tmp_dev_id].mem_B += tmp_bytes_to_free;
          gpu_res_in_use[tmp_dev_id].cores += tmp_cores_to_free;
          --*jobs_running_on_gpu;
        } else {
          stats.num_beacons++;
          boomers.push_back(comm);
          batch_size++; // batch size doesn't include free() beacons
          ++*jobs_waiting_on_gpu;
        }
      }

      *tail_p = (*tail_p + 1) & (BEMPS_BEACON_BUF_SZ - 1);
    }

    if (batch_size > stats.max_observed_batch_size) {
      stats.max_observed_batch_size = batch_size;
    }

    // Second loop: Walk the boomers. This time handle regular beacons, and
    // attempt to assign them to a device. The boomers are sorted by memory
    // footprint, highest to lowest.
    boomers.sort(mem_footprint_compare);
    boomers_len = boomers.size();
    if (boomers_len > stats.max_len_boomers) {
      stats.max_len_boomers = boomers_len;
    }
    if (boomers_len > 0) {
      BEMPS_SCHED_LOG("boomers_len: " << boomers_len << "\n");
    }
    for (i = 0; i < boomers_len; i++) {
      assigned = 0;
      comm = boomers.front();
      boomers.pop_front();

      if (comm->age > stats.max_age) {
        stats.max_age = comm->age;
      }

      // The target device for a process must have memory available for it,
      // and it should be the device with the least cores currently in use.
      long curr_min_cores = LONG_MAX;
      int target_dev_id = 0;
      for (tmp_dev_id = 0; tmp_dev_id < NUM_GPUS; tmp_dev_id++) {
        BEMPS_SCHED_LOG("Checking device " << tmp_dev_id << "\n"
                        << "  Total avail bytes: " << GPU_RES[tmp_dev_id].mem_B << "\n"
                        << "  In-use bytes: " << gpu_res_in_use[tmp_dev_id].mem_B << "\n"
                        << "  Trying-to-fit bytes: " << comm->beacon.mem_B << "\n"
                        << "  In-use cores: " << gpu_res_in_use[tmp_dev_id].cores << "\n"
                        << "  Trying-to-add cores: " << comm->beacon.cores << "\n");
        if (((gpu_res_in_use[tmp_dev_id].mem_B + comm->beacon.mem_B) <
             GPU_RES[tmp_dev_id].mem_B)) {
          if (gpu_res_in_use[tmp_dev_id].cores < curr_min_cores) {
              curr_min_cores = gpu_res_in_use[tmp_dev_id].cores;
              target_dev_id = tmp_dev_id;
              assigned = 1;
          }
        } else {
            BEMPS_SCHED_LOG("Couldn't fit " << comm->beacon.mem_B << "\n");
        }
      }

      if (!assigned) {
        // FIXME: need to add stats, and possibly a way to reserve a
        // GPU to prevent starving.
        comm->age++;
        boomers.push_back(comm);
        // don't adjust jobs-waiting-on-gpu. it was incremented when job first
        // went into the boomers list
      } else {
        long tmp_bytes_to_add = comm->beacon.mem_B;
        long tmp_cores_to_add = comm->beacon.cores;
        BEMPS_SCHED_LOG("Adding " << tmp_bytes_to_add << " bytes "
                        << "to device " << target_dev_id << "\n");
        BEMPS_SCHED_LOG("Adding " << tmp_cores_to_add << " cores "
                        << "to device " << target_dev_id << "\n");
        gpu_res_in_use[target_dev_id].mem_B += tmp_bytes_to_add;
        gpu_res_in_use[target_dev_id].cores += tmp_cores_to_add;
        BEMPS_SCHED_LOG("sem_post for pid(" << comm->pid << ") "
                                            << "on device(" << target_dev_id
                                            << ")\n");
        // FIXME Is this SCHEDULER_READ state helping at all?
        comm->state = BEMPS_BEACON_STATE_SCHEDULER_READ_E;
        comm->sched_notif.device_id = target_dev_id;
        comm->state = BEMPS_BEACON_STATE_SCHEDULED_E;
        sem_post(&comm->sched_notif.sem);
        ++*jobs_running_on_gpu;
        --*jobs_waiting_on_gpu;
      }
    }
    bemps_stopwatch_end(&sched_stopwatches[SCHED_STOPWATCH_AWAKE]);
  }
}


void sched_cg(void) {

  int i;
  int rc;
  int device_id;
  int *head_p;
  int *tail_p;
  struct timespec ts;
  bemps_shm_comm_t *comm;
  device_id_count_t *dc;

  for(i = 0; i < NUM_GPUS; i++){
    dc = (device_id_count_t *) malloc(sizeof(device_id_count_t));
    dc->device_id = i;
    dc->count = CG_JOBS_PER_GPU;
    avail_device_id_counts.push_back(dc);
  }

  device_id = 0;
  head_p = &bemps_shm_p->gen->beacon_q_head;
  tail_p = &bemps_shm_p->gen->beacon_q_tail;

  while (1) {
    set_wakeup_time_ns(&ts);

    // wait until we get a signal or time out
    pthread_mutex_lock(&bemps_shm_p->gen->lock);
    // TODO spurious wakeups ? shouldn't make a big difference to wake up
    // randomly from time to time before the batch is ready
    rc = pthread_cond_timedwait(&bemps_shm_p->gen->cond,
                                &bemps_shm_p->gen->lock, &ts);
    //BEMPS_SCHED_LOG("rc from timedwait: " << rc << "\n");
    //BEMPS_SCHED_LOG("strerror of rc: " << strerror(rc) << "\n");
    pthread_mutex_unlock(&bemps_shm_p->gen->lock);

    //BEMPS_SCHED_LOG("Woke up\n");
    bemps_stopwatch_start(&sched_stopwatches[SCHED_STOPWATCH_AWAKE]);

    ALIVE_MSG();

    // catch the scheduler's tail back up with the beacon queue's head
    while (*tail_p != *head_p) {
      BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
      BEMPS_SCHED_LOG("*head_p: " << (*head_p) << "\n");

      comm = &bemps_shm_p->comm[*tail_p];
      while (comm->state != BEMPS_BEACON_STATE_BEACON_FIRED_E) {
        // TODO probably want to track a stat for this case
        BEMPS_SCHED_LOG("WARNING: Scheduler hit a beacon before FIRED "
                        << "was set. (Not a bug, but unless we're "
                        << "flooded with beacons, this should be rare."
                        << "\n");
        // FIXME sync shouldn't hurt, but may not help?
        __sync_synchronize();
      }

      if (comm->exit_flag) {
        dc = pid_to_device_id_counts[comm->pid];
        BEMPS_SCHED_LOG("pid(" << comm->pid << ") exiting.\n");
        BEMPS_SCHED_LOG("recycling device_id(" << dc->device_id << ").\n");
        dc->count++;
        assert(dc->count <= CG_JOBS_PER_GPU); // error could mean problem with driver
        avail_device_id_counts.sort(avail_devices_compare);
        pid_to_device_id_counts.erase(comm->pid);
        comm->exit_flag = 0;
        comm->pid = 0;
      } else {
        if (pid_to_device_id_counts.find(comm->pid) == pid_to_device_id_counts.end()) {
          // Not found: We're seeing this pid for the first time.
          // This should be a proper beacon (not a free)
          assert(comm->beacon.mem_B > 0);
          dc = avail_device_id_counts.front();
          dc->count--;
          assert(dc->count >= 0); // error could mean a problem with driver
          avail_device_id_counts.sort(avail_devices_compare);
          pid_to_device_id_counts[comm->pid] = dc;
        } else {
          // Found: Do nothing.
          // assert that at least one process (the one that sent this beacon)
          // is assigned to this device (i.e. count should be < max)
          assert(pid_to_device_id_counts[comm->pid]->count < CG_JOBS_PER_GPU);
        }

        comm->state = BEMPS_BEACON_STATE_SCHEDULER_READ_E;
        comm->sched_notif.device_id = pid_to_device_id_counts[comm->pid]->device_id;
        comm->state = BEMPS_BEACON_STATE_SCHEDULED_E;
        sem_post(&comm->sched_notif.sem);
      }

      *tail_p = (*tail_p + 1) & (BEMPS_BEACON_BUF_SZ - 1);
    }
    bemps_stopwatch_end(&sched_stopwatches[SCHED_STOPWATCH_AWAKE]);
  }
}


void sched_single_assignment(void) {
  int i;
  int rc;
  int device_id;
  int *head_p;
  int *tail_p;
  struct timespec ts;
  bemps_shm_comm_t *comm;

  for(i = 0; i < NUM_GPUS; i++){
    avail_device_ids.push_back(i);
  }

  device_id = 0;
  head_p = &bemps_shm_p->gen->beacon_q_head;
  tail_p = &bemps_shm_p->gen->beacon_q_tail;

  while (1) {
    set_wakeup_time_ns(&ts);

    // wait until we get a signal or time out
    pthread_mutex_lock(&bemps_shm_p->gen->lock);
    // TODO spurious wakeups ? shouldn't make a big difference to wake up
    // randomly from time to time before the batch is ready
    rc = pthread_cond_timedwait(&bemps_shm_p->gen->cond,
                                &bemps_shm_p->gen->lock, &ts);
    //BEMPS_SCHED_LOG("rc from timedwait: " << rc << "\n");
    //BEMPS_SCHED_LOG("strerror of rc: " << strerror(rc) << "\n");
    pthread_mutex_unlock(&bemps_shm_p->gen->lock);

    //BEMPS_SCHED_LOG("Woke up\n");
    ALIVE_MSG();

    // catch the scheduler's tail back up with the beacon queue's head
    while (*tail_p != *head_p) {
      BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
      BEMPS_SCHED_LOG("*head_p: " << (*head_p) << "\n");

      comm = &bemps_shm_p->comm[*tail_p];
      while (comm->state != BEMPS_BEACON_STATE_BEACON_FIRED_E) {
        // TODO probably want to track a stat for this case
        BEMPS_SCHED_LOG("WARNING: Scheduler hit a beacon before FIRED "
                        << "was set. (Not a bug, but unless we're "
                        << "flooded with beacons, this should be rare."
                        << "\n");
        // FIXME sync shouldn't hurt, but may not help?
        __sync_synchronize();
      }

      if (comm->exit_flag) {
        BEMPS_SCHED_LOG("pid(" << comm->pid << ") exiting.\n");
        BEMPS_SCHED_LOG("recycling device_id(" << pid_to_device_id[comm->pid]
                        << ").\n");
        avail_device_ids.push_back(pid_to_device_id[comm->pid]);
        pid_to_device_id.erase(comm->pid);
        comm->exit_flag = 0;
        comm->pid = 0;
      } else {
        if (pid_to_device_id.find(comm->pid) == pid_to_device_id.end()) {
          // Not found: We're seeing this pid for the first time.
          // This should be a proper beacon (not a free)
          assert(comm->beacon.mem_B > 0);
          // No avail device could mean there's an issue with the driver
          assert(avail_device_ids.size() > 0);
          pid_to_device_id[comm->pid] = avail_device_ids.back();
          avail_device_ids.pop_back();
        } else {
          // Found: Do nothing.
        }

        comm->state = BEMPS_BEACON_STATE_SCHEDULER_READ_E;
        comm->sched_notif.device_id = pid_to_device_id[comm->pid];
        comm->state = BEMPS_BEACON_STATE_SCHEDULED_E;
        sem_post(&comm->sched_notif.sem);
      }

      *tail_p = (*tail_p + 1) & (BEMPS_BEACON_BUF_SZ - 1);
    }
  }
}


void sched_vector(void) {
  int tmp_dev_id;
  int *head_p;
  int *tail_p;
  int assigned;
  struct timespec ts;
  int boomers_len;
  int i;
  bemps_shm_comm_t *comm;
  int batch_size;
  /*std::priority_queue<bemps_shm_comm_t *,
                      std::vector<bemps_shm_comm_t *>,
                      CustomCompare> pq;*/
  // std::vector<bemps_shm_comm_t *> boomers_sorted;

  head_p = &bemps_shm_p->gen->beacon_q_head;
  tail_p = &bemps_shm_p->gen->beacon_q_tail;

  while (1) {
    set_wakeup_time_ns(&ts);

    // wait until we get a signal or time out
    pthread_mutex_lock(&bemps_shm_p->gen->lock);
    // TODO spurious wakeups ? shouldn't make a big difference to wake up
    // randomly from time to time before the batch is ready
    pthread_cond_timedwait(&bemps_shm_p->gen->cond, &bemps_shm_p->gen->lock,
                           &ts);
    pthread_mutex_unlock(&bemps_shm_p->gen->lock);

    // First loop: Catch the scheduler's tail back up with the beacon
    // queue's head. If we see a free-beacon, then reclaim that resource.
    BEMPS_SCHED_LOG("*head_p: " << (*head_p) << "\n");
    BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
    batch_size = 0;
    while (*tail_p != *head_p) {
      BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");

      comm = &bemps_shm_p->comm[*tail_p];
      while (comm->state != BEMPS_BEACON_STATE_BEACON_FIRED_E) {
        // TODO probably want to track a stat for this case
        BEMPS_SCHED_LOG("WARNING: Scheduler hit a beacon before FIRED "
                        << "was set. (Not a bug, but unless we're "
                        << "flooded with beacons, this should be rare."
                        << "\n");
        BEMPS_SCHED_LOG("WARNING: *tail_p: " << (*tail_p) << "\n");
        // FIXME sync shouldn't hurt, but may not help?
        __sync_synchronize();
      }

      BEMPS_SCHED_LOG("First loop seeing mem_B: " << comm->beacon.mem_B
                                                  << "\n");

      assert(comm->beacon.mem_B);
      if (comm->beacon.mem_B < 0) {
        stats.num_frees++;

        BEMPS_SCHED_LOG("Received free-beacon for pid " << comm->pid << "\n");
        tmp_dev_id = comm->sched_notif.device_id;
        // Add (don't subtract), because mem_B is negative already
        gpu_res_in_use[tmp_dev_id].mem_B += comm->beacon.mem_B;
        gpu_res_in_use[tmp_dev_id].cores += comm->beacon.cores;
      } else {
        stats.num_beacons++;
        boomers.push_back(comm);
        // pq.push(comm);
        batch_size++;
      }

      *tail_p = (*tail_p + 1) & (BEMPS_BEACON_BUF_SZ - 1);
    }

    if (batch_size > stats.max_observed_batch_size) {
      stats.max_observed_batch_size = batch_size;
    }

    // Second loop: Walk the boomers. This time handle regular beacons,
    // and attempt to assign them a device
    // std::sort(boomers.begin(), boomers.end(), mem_footprint_compare);
    boomers.sort(mem_footprint_compare);
    boomers_len = boomers.size();
    if (boomers_len > stats.max_len_boomers) {
      stats.max_len_boomers = boomers_len;
    }
    BEMPS_SCHED_LOG("boomers_len: " << boomers_len << "\n");
    for (i = 0; i < boomers_len; i++) {
      assigned = 0;
      comm = boomers.front();
      boomers.pop_front();

      if (comm->age > stats.max_age) {
        stats.max_age = comm->age;
      }

      for (tmp_dev_id = 0; tmp_dev_id < NUM_GPUS; tmp_dev_id++) {
        /*if (((gpu_res_in_use[tmp_dev_id].mem_B + comm->beacon.mem_B) <
             GPU_RES[tmp_dev_id].mem_B) &&
            ((gpu_res_in_use[tmp_dev_id].cores + comm->beacon.cores) <
             GPU_RES[tmp_dev_id].cores)) {*/
        if (((gpu_res_in_use[tmp_dev_id].mem_B + comm->beacon.mem_B) <
             GPU_RES[tmp_dev_id].mem_B)) {
          gpu_res_in_use[tmp_dev_id].mem_B += comm->beacon.mem_B;
          gpu_res_in_use[tmp_dev_id].cores += comm->beacon.cores;
          assigned = 1;
          break;
        }
      }

      if (!assigned) {
        // FIXME: need to add stats, and possibly a way to reserve a
        // GPU to prevent starving.
        comm->age++;
        boomers.push_back(comm);
      } else {
        BEMPS_SCHED_LOG("sem_post for pid(" << comm->pid << ") "
                                            << "on device(" << tmp_dev_id
                                            << ")\n");
        // FIXME Is this SCHEDULER_READ state helping at all?
        comm->state = BEMPS_BEACON_STATE_SCHEDULER_READ_E;
        comm->sched_notif.device_id = tmp_dev_id;
        comm->state = BEMPS_BEACON_STATE_SCHEDULED_E;
        sem_post(&comm->sched_notif.sem);
      }
    }
  }
}

void sched_round_robin(void) {
  int device_id;
  int tmp_dev_id;
  int *head_p;
  int *tail_p;
  int bcn_idx;
  int assigned;
  struct timespec ts;
  int boomers_len;
  int i;
  bemps_shm_comm_t *comm;

  device_id = 0;
  head_p = &bemps_shm_p->gen->beacon_q_head;
  tail_p = &bemps_shm_p->gen->beacon_q_tail;

  while (1) {
    set_wakeup_time_ns(&ts);

    // wait until we get a signal or time out
    pthread_mutex_lock(&bemps_shm_p->gen->lock);
    // TODO spurious wakeups ? shouldn't make a big difference to wake up
    // randomly from time to time before the batch is ready
    pthread_cond_timedwait(&bemps_shm_p->gen->cond, &bemps_shm_p->gen->lock,
                           &ts);
    pthread_mutex_unlock(&bemps_shm_p->gen->lock);

    // Loop zero: Handle old beacons that haven't been scheduled yet.
    boomers_len = boomers.size();
    if (boomers_len > stats.max_len_boomers) {
      stats.max_len_boomers = boomers_len;
    }
    BEMPS_SCHED_LOG("boomers_len: " << boomers_len << "\n");
    for (i = 0; i < boomers_len; i++) {
      assigned = 0;
      tmp_dev_id = device_id;
      comm = boomers.front();
      boomers.pop_front();

      if (comm->age > stats.max_age) {
        stats.max_age = comm->age;
      }

      while (1) {
        /*if (((gpu_res_in_use[tmp_dev_id].mem_B + comm->beacon.mem_B) <
             GPU_RES[tmp_dev_id].mem_B) &&
            ((gpu_res_in_use[tmp_dev_id].cores + comm->beacon.cores) <
             GPU_RES[tmp_dev_id].cores)) {*/
        if (((gpu_res_in_use[tmp_dev_id].mem_B + comm->beacon.mem_B) <
             GPU_RES[tmp_dev_id].mem_B)) {
          gpu_res_in_use[tmp_dev_id].mem_B += comm->beacon.mem_B;
          gpu_res_in_use[tmp_dev_id].cores += comm->beacon.cores;
          assigned = 1;
          break;
        }

        tmp_dev_id = (tmp_dev_id + 1) & (NUM_GPUS - 1);
        if (tmp_dev_id == device_id) {
          break;
        }
      }

      if (!assigned) {
        // FIXME: need to add stats, and possibly a way to reserve a
        // GPU to prevent starving.
        comm->age++;
        boomers.push_back(comm);
      } else {
        // FIXME Is this SCHEDULER_READ state helping at all?
        comm->state = BEMPS_BEACON_STATE_SCHEDULER_READ_E;
        comm->sched_notif.device_id = device_id;
        comm->state = BEMPS_BEACON_STATE_SCHEDULED_E;
        sem_post(&comm->sched_notif.sem);

        device_id = (device_id + 1) & (NUM_GPUS - 1);
      }
    }

    // First loop: Catch the scheduler's tail back up with the beacon
    // queue's head. If we see a free-beacon, then reclaim that resource.
    bcn_idx = *tail_p;
    BEMPS_SCHED_LOG("*head_p: " << (*head_p) << "\n");
    BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
    while (*tail_p != *head_p) {
      BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");

      comm = &bemps_shm_p->comm[*tail_p];
      while (comm->state != BEMPS_BEACON_STATE_BEACON_FIRED_E) {
        // TODO probably want to track a stat for this case
        BEMPS_SCHED_LOG("WARNING: Scheduler hit a beacon before FIRED "
                        << "was set. (Not a bug, but unless we're "
                        << "flooded with beacons, this should be rare."
                        << "\n");
        BEMPS_SCHED_LOG("WARNING: *tail_p: " << (*tail_p) << "\n");
        // FIXME sync shouldn't hurt, but may not help?
        __sync_synchronize();
      }

      BEMPS_SCHED_LOG("First loop seeing mem_B: " << comm->beacon.mem_B
                                                  << "\n");

      assert(comm->beacon.mem_B);
      if (comm->beacon.mem_B < 0) {
        stats.num_frees++;

        BEMPS_SCHED_LOG("Received free-beacon for pid " << comm->pid << "\n");
        tmp_dev_id = comm->sched_notif.device_id;
        // Add (don't subtract), because mem_B is negative already
        gpu_res_in_use[tmp_dev_id].mem_B += comm->beacon.mem_B;
        gpu_res_in_use[tmp_dev_id].cores += comm->beacon.cores;
      }

      *tail_p = (*tail_p + 1) & (BEMPS_BEACON_BUF_SZ - 1);
    }

    // Second loop: Walk the queue again. This time handle regular beacons,
    // and attempt to assign them a device
    BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
    BEMPS_SCHED_LOG("bcn_idx: " << (bcn_idx) << "\n");
    while (bcn_idx != *tail_p) {
      BEMPS_SCHED_LOG("Second loop bcn_idx: " << (bcn_idx) << "\n");

      assigned = 0;
      tmp_dev_id = device_id;
      comm = &bemps_shm_p->comm[bcn_idx];

      if (comm->beacon.mem_B < 0) {
        bcn_idx = (bcn_idx + 1) & (BEMPS_BEACON_BUF_SZ - 1);
        continue;
      }
      stats.num_beacons++;

      while (1) {
        /*if (((gpu_res_in_use[tmp_dev_id].mem_B + comm->beacon.mem_B) <
             GPU_RES[tmp_dev_id].mem_B) &&
            ((gpu_res_in_use[tmp_dev_id].cores + comm->beacon.cores) <
             GPU_RES[tmp_dev_id].cores)) {*/
        if (((gpu_res_in_use[tmp_dev_id].mem_B + comm->beacon.mem_B) <
             GPU_RES[tmp_dev_id].mem_B)) {
          gpu_res_in_use[tmp_dev_id].mem_B += comm->beacon.mem_B;
          gpu_res_in_use[tmp_dev_id].cores += comm->beacon.cores;
          assigned = 1;
          BEMPS_SCHED_LOG("  assigned\n");
          break;
        }else{
          BEMPS_SCHED_LOG("  not assigned\n");
        }

        tmp_dev_id = (tmp_dev_id + 1) & (NUM_GPUS - 1);
        if (tmp_dev_id == device_id) {
          break;
        }
      }

      if (!assigned) {
        comm->age++;
        boomers.push_back(comm);
      } else {
        BEMPS_SCHED_LOG("sem_post for pid(" << comm->pid << ") "
                                            << "on device(" << device_id
                                            << ")\n");
        // FIXME Is this SCHEDULER_READ state helping at all?
        comm->state = BEMPS_BEACON_STATE_SCHEDULER_READ_E;
        comm->sched_notif.device_id = device_id;
        comm->state = BEMPS_BEACON_STATE_SCHEDULED_E;
        sem_post(&comm->sched_notif.sem);

        device_id = (device_id + 1) & (NUM_GPUS - 1);
      }

      bcn_idx = (bcn_idx + 1) & (BEMPS_BEACON_BUF_SZ - 1);
    }
  }
}


void sched_no_beacons(int is_round_robin) {
  int rc;
  int device_id;
  int *head_p;
  int *tail_p;
  struct timespec ts;
  bemps_shm_comm_t *comm;

  device_id = 0;
  head_p = &bemps_shm_p->gen->beacon_q_head;
  tail_p = &bemps_shm_p->gen->beacon_q_tail;

  while (1) {
    set_wakeup_time_ns(&ts);

    // wait until we get a signal or time out
    pthread_mutex_lock(&bemps_shm_p->gen->lock);
    // TODO spurious wakeups ? shouldn't make a big difference to wake up
    // randomly from time to time before the batch is ready
    rc = pthread_cond_timedwait(&bemps_shm_p->gen->cond,
                                &bemps_shm_p->gen->lock, &ts);
    BEMPS_SCHED_LOG("rc from timedwait: " << rc << "\n");
    BEMPS_SCHED_LOG("strerror of rc: " << strerror(rc) << "\n");
    pthread_mutex_unlock(&bemps_shm_p->gen->lock);

    BEMPS_SCHED_LOG("Woke up\n");

    // catch the scheduler's tail back up with the beacon queue's head
    while (*tail_p != *head_p) {
      BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
      BEMPS_SCHED_LOG("*head_p: " << (*head_p) << "\n");

      comm = &bemps_shm_p->comm[*tail_p];
      while (comm->state != BEMPS_BEACON_STATE_BEACON_FIRED_E) {
        // TODO probably want to track a stat for this case
        BEMPS_SCHED_LOG("WARNING: Scheduler hit a beacon before FIRED "
                        << "was set. (Not a bug, but unless we're "
                        << "flooded with beacons, this should be rare."
                        << "\n");
        // FIXME sync shouldn't hurt, but may not help?
        __sync_synchronize();
      }

      comm->state = BEMPS_BEACON_STATE_SCHEDULER_READ_E;
      comm->sched_notif.device_id = device_id;
      comm->state = BEMPS_BEACON_STATE_SCHEDULED_E;
      sem_post(&comm->sched_notif.sem);

      if (is_round_robin) {
        device_id = (device_id + 1) & (NUM_GPUS - 1);
      }
      *tail_p = (*tail_p + 1) & (BEMPS_BEACON_BUF_SZ - 1);
    }
  }
}

void sched(void) {
  if (which_scheduler == SCHED_ALG_ZERO_E) {
    BEMPS_SCHED_LOG("Starting zero scheduler\n");
    sched_no_beacons(0);
  } else if (which_scheduler == SCHED_ALG_ROUND_ROBIN_E) {
    BEMPS_SCHED_LOG("Starting round robin scheduler\n");
    sched_no_beacons(1);
  } else if (which_scheduler == SCHED_ALG_ROUND_ROBIN_BEACONS_E) {
    BEMPS_SCHED_LOG("Starting round robin beacons scheduler\n");
    sched_round_robin();
  } else if (which_scheduler == SCHED_ALG_VECTOR_E) {
    BEMPS_SCHED_LOG("Starting vector scheduler\n");
    sched_vector();
  } else if (which_scheduler == SCHED_ALG_SINGLE_ASSIGNMENT_E) {
    BEMPS_SCHED_LOG("Starting single asssignment scheduler\n");
    sched_single_assignment();
  } else if (which_scheduler == SCHED_ALG_CG_E) {
    BEMPS_SCHED_LOG("Starting C:G scheduler\n");
    BEMPS_SCHED_LOG("  CG_JOBS_PER_GPU: " << CG_JOBS_PER_GPU << "\n");
    sched_cg();
  } else if (which_scheduler == SCHED_ALG_MGB_E) {
    BEMPS_SCHED_LOG("Starting mgb scheduler\n");
    sched_mgb();
  } else {
    fprintf(stderr, "ERROR: Invalid scheduling algorithm\n");
    exit(2);
  }
  fprintf(stderr, "ERROR: Scheduler loop returned\n");
  exit(3);
}

void parse_args(int argc, char **argv) {
  int i;
  int num_workers;

  max_batch_size = SCHED_DEFAULT_BATCH_SIZE;

  if (argc > 3) {
    usage_and_exit(argv[0]);
  }

  if (argc == 1) {
    which_scheduler = SCHED_ALG_ZERO_E;
    return;
  }

  if (strncmp(argv[1], "zero", 5) == 0) {
    which_scheduler = SCHED_ALG_ZERO_E;
  } else if (strncmp(argv[1], "round-robin", 12) == 0) {
    which_scheduler = SCHED_ALG_ROUND_ROBIN_E;
  } else if (strncmp(argv[1], "round-robin-beacons", 20) == 0) {
    which_scheduler = SCHED_ALG_ROUND_ROBIN_BEACONS_E;
  } else if (strncmp(argv[1], "vector", 7) == 0) {
    which_scheduler = SCHED_ALG_VECTOR_E;
    max_batch_size = SCHED_VECTOR_BATCH_SIZE;
  } else if (strncmp(argv[1], "single-assignment", 18) == 0) {
    which_scheduler = SCHED_ALG_SINGLE_ASSIGNMENT_E;
  } else if (strncmp(argv[1], "cg", 3) == 0) {
    which_scheduler = SCHED_ALG_CG_E;
    if (argc != 3) {
      usage_and_exit(argv[0]);
    }
    for (i = 0; i < strlen(argv[2]); i++) {
      if (!isdigit(argv[2][i])) {
        usage_and_exit(argv[0]);
      }
    }
    num_workers = atoi(argv[2]);
    CG_JOBS_PER_GPU = num_workers / NUM_GPUS;
    if (num_workers % NUM_GPUS) {
      CG_JOBS_PER_GPU++;
    }
  } else if (strncmp(argv[1], "mgb", 4) == 0) {
    which_scheduler = SCHED_ALG_MGB_E;
    max_batch_size = SCHED_MGB_BATCH_SIZE;
  } else {
    usage_and_exit(argv[0]);
  }

  if (which_scheduler != SCHED_ALG_CG_E && argc != 2) {
    usage_and_exit(argv[0]);
  }
}

int main(int argc, char **argv) {
  BEMPS_SCHED_LOG("BEMPS SCHEDULER\n");
  signal(SIGINT, sigint_handler);

  BEMPS_SCHED_LOG("Parsing args\n");
  parse_args(argc, argv);

  BEMPS_SCHED_LOG("Initializing shared memory.\n");
  bemps_shm_p = bemps_sched_init(max_batch_size);

  INIT_GPU_RES();

  sched();

  return 0;
}
