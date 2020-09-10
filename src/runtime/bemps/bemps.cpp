#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

// #include <bemps/bemps.hpp>
#include <iostream>
#include <map>

#include "bemps.hpp"

//#define BEMPS_DEBUG

#ifdef BEMPS_DEBUG
#define BEMPS_LOG(str)                                                 \
  do {                                                                 \
    std::cout << _get_time_ns() << " " << __FUNCTION__ << ": " << str; \
    std::cout.flush();                                                 \
  } while (0)
#else
#define BEMPS_LOG(str) \
  do {                 \
  } while (0)
#endif

#ifdef BEMPS_STATS
#define BEMPS_STATS_LOG(str)    \
  do {                          \
    std::cout << _get_time_ns() << " " << __FUNCTION__ << ": " << str; \
    std::cout.flush();                                                 \
  } while (0)
#else
#define BEMPS_STATS_LOG(str) \
  do {                 \
  } while (0)
#endif


#define ABS(x) (((x) ^ ((x) >> 31)) - ((x) >> 31))

#define BEMPS_NUM_STOPWATCHES 2

typedef enum {
  BEMPS_STOPWATCH_BEACON = 0,
  BEMPS_STOPWATCH_FREE
} bemps_stopwatch_e;



pid_t pid;  // linux process id
bool beacon_initialized = false;

std::map<int, int> bemps_tid_to_q_idx;

// base of the bemps shared memory
bemps_shm_t bemps_shm;


bemps_stopwatch_t bemps_stopwatches[BEMPS_NUM_STOPWATCHES];

// Programmer-friendly pointers into the application's shared memory elements
// long long           *timestamp_p;
// bemps_sched_notif_t *sched_notif_p;

static inline long long _get_time_ns(void) {
  struct timespec tv = {0};
  if (clock_gettime(CLOCK_MONOTONIC, &tv) != 0) {
    fprintf(stderr, "ERROR: _get_time_ns failed\n");
    return 0;
  }
  return (((long long)tv.tv_sec * 1000000000L) + (long long)(tv.tv_nsec));
}

void _bemps_dump_stats(void) {
  bemps_stopwatch_t *sb;
  bemps_stopwatch_t *sf;
  sb = &bemps_stopwatches[BEMPS_STOPWATCH_BEACON];
  sf = &bemps_stopwatches[BEMPS_STOPWATCH_FREE];
  BEMPS_STATS_LOG("Dumping bemps stats\n");
  BEMPS_STATS_LOG("count of beacon times: " << sb->n << "\n");
  BEMPS_STATS_LOG("min beacon time (ns): " << sb->min << "\n");
  BEMPS_STATS_LOG("max beacon time (ns): " << sb->max << "\n");
  BEMPS_STATS_LOG("avg beacon time (ns): " << sb->avg << "\n");
  BEMPS_STATS_LOG("count of free times: " << sf->n << "\n");
  BEMPS_STATS_LOG("min free time (ns): " << sf->min << "\n");
  BEMPS_STATS_LOG("max free time (ns): " << sf->max << "\n");
  BEMPS_STATS_LOG("avg free time (ns): " << sf->avg << "\n");
}

void bemps_stopwatch_start(bemps_stopwatch_t *s) {
#ifdef BEMPS_STATS
  s->ts = _get_time_ns();
#endif
}
void bemps_stopwatch_end(bemps_stopwatch_t *s) {
#ifdef BEMPS_STATS
  long long diff;
  ++(s->n);
  diff = _get_time_ns() - s->ts;
  if (diff < s->min || s->min == 0LL) {
    s->min = diff;
  } // not else-if. edge case: 1 beacon. min and max both get updated
  if (diff > s->max) {
    s->max = diff;
  }
  s->avg += (diff - s->avg) / s->n;
#endif
}




static inline int _inc_head(int *head) {
  int q_idx;

  // head++
  q_idx = __sync_fetch_and_add(&bemps_shm.gen->beacon_q_head, 1);

  // XXX Race between the scheduler and this process can occur after the head
  // has incremented (above) but before we write the beacon data. Though
  // we haven't signaled, this can happen because either the scheduler woke up
  // due to timeout, or because it was iterating through the buffer and
  // got to this new item (head was incremented) before it was properly added
  // (which we do below). This is fine, though, because the scheduler won't
  // continue unless the state is BEMPS_BEACON_STATE_BEACON_FIRED_E, which
  // is the last element we update below.

  if (q_idx >= BEMPS_BEACON_BUF_SZ) {
    q_idx &= (BEMPS_BEACON_BUF_SZ - 1);
    // XXX Race on beacon_q_head with other processes is OK.
    // Only the race winner performs a useful AND operation.
    // Other operations are idempotent.
    // No temp needed so we use and-and-fetch.
    // No need for return value either.
    //
    // XXX Race on beacon_q_head between this process and the scheduler is
    // more complicated, but it's also OK. This AND only occurs when
    // the buffer wraps around. A race could happen if the head gets
    // incremented (the head++ code just above), followed by a scheduler
    // time out (or just iteration again over the beacon q), followed by the
    // scheduler dropping into its loop to pull out the beacon. But the
    // scheduler always checks the state is
    // BEMPS_BEACON_STATE_BEACON_FIRED_E. So this is just a special case of
    // the race mentioned above this if-check.
    (void)__sync_and_and_fetch(&bemps_shm.gen->beacon_q_head,
                               (BEMPS_BEACON_BUF_SZ - 1));
  }

  return q_idx;
}

static inline void _check_state(bemps_beacon_state_e tmp_state, int q_idx) {
  if (tmp_state != BEMPS_BEACON_STATE_COMPLETED_E) {
    fprintf(stderr, "Head caught the tail at q_idx(%d) and state(%d)\n", q_idx,
            tmp_state);
    // TODO Figure out what kind of action is even possible here, assuming
    // the experiment/environment where this occurred is reasonable. Is
    // it enough to double the buffer size, or does something more complex
    // need to be done?
    exit(12);
  }
}

static inline int _push_beacon(bemps_beacon_t *beacon_p) {
  int q_idx;
  bemps_shm_comm_t *comm;

  q_idx = _inc_head(&bemps_shm.gen->beacon_q_head);

  comm = &bemps_shm.comm[q_idx];

  _check_state(comm->state, q_idx);

  comm->timestamp_ns = _get_time_ns();
  comm->beacon.mem_B = beacon_p->mem_B;
  comm->beacon.warps = beacon_p->warps;
  comm->beacon.thread_blocks = beacon_p->thread_blocks;
  comm->pid = pid;

  // XXX This should be the last field that we change in the comm structure,
  // because it's used for gating the scheduler and preventing races.
  comm->state = BEMPS_BEACON_STATE_BEACON_FIRED_E;

  // if batch size is reached, then signal the scheduler
  // TODO prettify
  if (bemps_shm.gen->beacon_q_head > bemps_shm.gen->beacon_q_tail) {
    if ((bemps_shm.gen->beacon_q_head - bemps_shm.gen->beacon_q_tail) >=
        bemps_shm.gen->max_batch_size) {
      BEMPS_LOG("Batch size met. Signaling scheduler.\n");
      pthread_mutex_lock(&bemps_shm.gen->lock);
      pthread_cond_signal(&bemps_shm.gen->cond);
      pthread_mutex_unlock(&bemps_shm.gen->lock);
    }
  } else {
    if (((BEMPS_BEACON_BUF_SZ - bemps_shm.gen->beacon_q_tail) +
         bemps_shm.gen->beacon_q_head) >= bemps_shm.gen->max_batch_size) {
      BEMPS_LOG("Batch size met (wraparound case). Signaling scheduler.\n");
      pthread_mutex_lock(&bemps_shm.gen->lock);
      pthread_cond_signal(&bemps_shm.gen->cond);
      pthread_mutex_unlock(&bemps_shm.gen->lock);
    }
  }

  return q_idx;
}

// XXX can probably remove this. See 2020.04.20 notes on callbacks
// void CUDART_CB _kernel_complete_cb(cudaStream_t stream, cudaError_t status,
// void *data)
//{
//    int q_idx;
//    q_idx = (intptr_t) data;
//    bemps_shm.comm[q_idx].state = BEMPS_BEACON_STATE_COMPLETED_E;
//    printf("kernel_complete_cb from bemps: %d\n", q_idx);
//}
// static inline
// void _set_cuda_callback(int bemps_tid)
//{
//    int rc;
//    int q_idx;
//
//    q_idx = bemps_tid_to_q_idx[bemps_tid];
//    bemps_tid_to_q_idx.erase(bemps_tid);
//
//    // XXX Assumption: The application isn't using CUDA streams.
//    // Here we use the default stream.
//    rc = cudaStreamAddCallback(0,
//                               _kernel_complete_cb,
//                               (void *) (intptr_t) q_idx,
//                               0);
//    if(rc != CUDA_SUCCESS){
//        fprintf(stderr, "ERROR set device failed (%d)\n", rc);
//        // FIXME: error handling. shouldn't exit from this library.
//        exit(11);
//    }
//}
// void bemps_set_cuda_callback(int bemps_tid)
//{
//    // Alternative design: Instead of taking a bemps_tid as the argument,
//    // we could take the q_idx. This means that bemps_beacon() would have to
//    // return the q_idx, and the process would have to manage it at runtime.
//    // Assuming the process calls bemps_beacon() in the same order that the
//    // kernels are called, it could just add q_idx to an internal queue. Then
//    when setting
//    // the callback function, it would dequeue. This assumption may not always
//    // hold, though, which means the process would have to manage which
//    function
//    // is associatd with which q_idx. Rather than complicate the runtime of
//    // the process, the current design opts for complicating the compilation
//    // slightly. A bemps_tid is assigned at compile time for each
//    bemps_beacon()
//    // call and its corresponding bemps_set_cuda_callback().
//    // This guarantees that the ID for a bemps_beacon() call is associated
//    // correctly with the bemps_set_cuda_callback() call. We could use
//    something
//    // like a kernel name instead of a unique beacon ID, but that could fail
//    // if the kernel name is called from more than one place.
//    _set_cuda_callback(bemps_tid);
//}

static inline void _wait_for_sched(bemps_shm_comm_t *comm) {
  // comm->sched_notif.device_id = -1;
  // BEMPS_LOG("device id val:  " << sched_notif_p->device_id << "\n");
  // while(sched_notif_p->device_id == -1){
  //    // busy wait... it's either that or we need locking and cond signaling
  //    __sync_synchronize();
  //}

  // TODO: Initialize all of these variables at scheduler start
  // pthread_mutex_lock(&comm->sched_notif.lock);
  // pthread_cond_wait(&comm->sched_notif.cond);
  // pthread_mutex_unlock(&comm->sched_notif.lock);

  // TODO ? This will block the process. We may need to make a thread pool
  // for this.
  sem_wait(&comm->sched_notif.sem);
}

static inline void _set_device(int device_id) {
  int rc;
  rc = cudaSetDevice(device_id);
  if (rc != CUDA_SUCCESS) {
    fprintf(stderr, "ERROR set device failed (%d)\n", rc);
    // FIXME: error handling. shouldn't exit from this library.
    exit(10);
  }
}

static inline void _reset_comm(bemps_shm_comm_t *comm) {
  // FIXME use memset, but restructure it so we don't blow over sem?
  //       or reset only the fields that "matter" (e.g. state and age, and
  //         maybe the device id)?
  comm->timestamp_ns = 0L;
  comm->pid = 0;
  comm->age = 0;
  comm->exit_flag = 0;
  comm->state = BEMPS_BEACON_STATE_COMPLETED_E;
  comm->beacon.mem_B = 0;
  comm->beacon.warps = 0;
  comm->beacon.thread_blocks = 0;
  comm->sched_notif.device_id = -1;
}

void bemps_beacon(int bemps_tid, bemps_beacon_t *beacon) {
  int q_idx;
  bemps_shm_comm_t *comm;

  BEMPS_STATS_LOG("pid " << pid << " , "
               << "bemps_tid " << bemps_tid << " , "
               << "mem_B " << beacon->mem_B << " , "
               << "warps " << beacon->warps << " , "
               << "thread_blocks " << beacon->thread_blocks << "\n");

  q_idx = _push_beacon(beacon);
  bemps_tid_to_q_idx[bemps_tid] = q_idx;

  comm = &bemps_shm.comm[q_idx];

  _wait_for_sched(comm);

  _set_device(comm->sched_notif.device_id);
}
/** a simple wrapper to bemps_beacon
 * added by Chao
 */
extern "C" {
void bemps_begin(int id, int gx, int gy, int gz, int bx, int by, int bz,
                 int64_t membytes) {
  long num_blocks;
  long threads_per_block;
  long warps;
  bemps_beacon_t beacon;

  bemps_stopwatch_start(&bemps_stopwatches[BEMPS_STOPWATCH_BEACON]);
  if (!beacon_initialized) {
    beacon_initialized = !bemps_init();
  }

  num_blocks        = gx * gy * gz;
  threads_per_block = bx * by * bz;
  BEMPS_STATS_LOG("pid " << pid << " , "
               << "num_blocks " << num_blocks << " , "
               << "threads_per_block " << threads_per_block << "\n");
  warps = num_blocks * threads_per_block / 32;
  if ((num_blocks * threads_per_block) % 32) {
    warps++;
  }
  beacon.mem_B = membytes;
  beacon.warps = warps;
  beacon.thread_blocks = num_blocks;
  bemps_beacon(id, &beacon);
  bemps_stopwatch_end(&bemps_stopwatches[BEMPS_STOPWATCH_BEACON]);
}
}

void bemps_set_device(int bemps_tid) {
  int q_idx;
  bemps_shm_comm_t *comm;
  int device_id;

  q_idx = bemps_tid_to_q_idx[bemps_tid];
  // bemps_tid_to_q_idx.erase(bemps_tid);
  comm = &bemps_shm.comm[q_idx];
  device_id = comm->sched_notif.device_id;
  BEMPS_LOG("bemps_tid(" << bemps_tid << ")"
                         << "q_idx(" << q_idx << ")"
                         << "device_id(" << device_id << ")\n");
  _set_device(device_id);
}

extern "C" {
void bemps_free(int bemps_tid) {
  int orig_beacon_q_idx;
  int free_beacon_q_idx;
  bemps_shm_comm_t *comm;
  long mem_B;
  long warps;
  long thread_blocks;
  int device_id;

  bemps_stopwatch_start(&bemps_stopwatches[BEMPS_STOPWATCH_FREE]);

  //
  // Look up what was used by this BEMPS task.
  // Mark that buffer entry as completed
  //
  orig_beacon_q_idx = bemps_tid_to_q_idx[bemps_tid];
  comm = &bemps_shm.comm[orig_beacon_q_idx];
  mem_B = comm->beacon.mem_B;
  warps = comm->beacon.warps;
  thread_blocks = comm->beacon.thread_blocks;
  device_id = comm->sched_notif.device_id;
  bemps_tid_to_q_idx.erase(bemps_tid);
  _reset_comm(comm);

  BEMPS_LOG("pid " << pid << " , "
          << "freeing device_id " << device_id << " , "
          << "bemps_tid " << bemps_tid << "\n");

  //
  // Notify the scheduler of the resources that have freed up.
  //
  free_beacon_q_idx = _inc_head(&bemps_shm.gen->beacon_q_head);
  comm = &bemps_shm.comm[free_beacon_q_idx];
  _check_state(comm->state, free_beacon_q_idx);

  comm->timestamp_ns = _get_time_ns();
  comm->beacon.mem_B = -1L * mem_B;
  comm->beacon.warps = -1L * warps;
  comm->beacon.thread_blocks = -1L * thread_blocks;
  comm->sched_notif.device_id = device_id;
  comm->pid = pid;

  // XXX The negative value for the beacon resources indicates that this
  // is a free-beacon

  // XXX This should be the last field that we change in the comm structure,
  // because it's used for gating the scheduler and preventing races.
  comm->state = BEMPS_BEACON_STATE_BEACON_FIRED_E;

  // Don't need to signal the scheduler (w.r.t. batch size)
  // Don't need to wait for the scheduler
  // Don't need to capture the q_idx in bemps_tid_to_q_idx
  // Don't need to set device

  bemps_stopwatch_end(&bemps_stopwatches[BEMPS_STOPWATCH_FREE]);
}
}

void _send_beacon_at_exit(void) {
  int orig_beacon_q_idx;
  int exit_beacon_q_idx;
  bemps_shm_comm_t *comm;

  BEMPS_LOG("pid(" << pid << ") "
                   << "setting exit_flag to 1"
                   << "\n");

  exit_beacon_q_idx = _inc_head(&bemps_shm.gen->beacon_q_head);
  comm = &bemps_shm.comm[exit_beacon_q_idx];
  comm->pid = pid;
  comm->exit_flag = 1;

  // XXX This should be the last field that we change in the comm structure,
  // because it's used for gating the scheduler and preventing races.
  comm->state = BEMPS_BEACON_STATE_BEACON_FIRED_E;

  _bemps_dump_stats();
}

int bemps_init(void) {
  int fd;
  int err;

  fd = shm_open("/bemps", O_RDWR, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    fprintf(stderr, "ERROR bemps_init: shm_open failed\n");
    return 1;
  }

  size_t shm_sz = (sizeof(bemps_shm_gen_t))                            // gen
                  + (sizeof(bemps_shm_comm_t) * BEMPS_BEACON_BUF_SZ);  // comm

  bemps_shm.gen = (bemps_shm_gen_t *)mmap(NULL, shm_sz, PROT_READ | PROT_WRITE,
                                          MAP_SHARED, fd, 0);
  if (bemps_shm.gen == MAP_FAILED) {
    err = errno;
    fprintf(stderr, "ERROR bemps_init: mmap failed (%d)\n", err);
    fprintf(stderr, "  %s\n", strerror(err));
    exit(4);
  }

  BEMPS_LOG("bemps_shm mmap:  " << bemps_shm.gen << "\n");
  BEMPS_LOG("shm_sz:              " << shm_sz << "\n");

  bemps_shm.comm = (bemps_shm_comm_t *)(((char *)(bemps_shm.gen)) +
                                        (sizeof(bemps_shm_gen_t)));
  pid = getpid();

  BEMPS_LOG("pid:                 " << pid << "\n");
  BEMPS_LOG("bemps_shm.gen:       " << bemps_shm.gen << "\n");
  BEMPS_LOG("bemps_shm.comm:      " << bemps_shm.comm << "\n");

  atexit(_send_beacon_at_exit);

  return 0;
}

// FIXME: error handling. shouldn't exit from this library.
bemps_shm_t *bemps_sched_init(int max_batch_size) {
  int i;
  int fd;
  int rc;
  int err;
  size_t shm_sz;

  fd = shm_open("/bemps", O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    fprintf(stderr, "ERROR bemps_sched_init: shm_open failed\n");
    exit(1);
  }

  shm_sz = (sizeof(bemps_shm_gen_t))                            // gen
           + (sizeof(bemps_shm_comm_t) * BEMPS_BEACON_BUF_SZ);  // comm

  rc = ftruncate(fd, shm_sz);
  if (rc == -1) {
    fprintf(stderr, "ERROR bemps_sched_init: ftruncate failed\n");
    exit(2);
  }

  bemps_shm.gen = (bemps_shm_gen_t *)mmap(NULL, shm_sz, PROT_READ | PROT_WRITE,
                                          MAP_SHARED, fd, 0);
  if (bemps_shm.gen == MAP_FAILED) {
    err = errno;
    fprintf(stderr, "ERROR bemps_sched_init: mmap failed \n");
    fprintf(stderr, "  %s\n", strerror(err));
    exit(3);
  }

  memset(bemps_shm.gen, 0, shm_sz);

  bemps_shm.gen->max_batch_size = max_batch_size;

  bemps_shm.comm = (bemps_shm_comm_t *)(((char *)(bemps_shm.gen)) +
                                        (sizeof(bemps_shm_gen_t)));

  BEMPS_LOG("bemps_shm.gen:      " << bemps_shm.gen << "\n");
  BEMPS_LOG("bemps_shm.comm:     " << bemps_shm.comm << "\n");

  for (i = 0; i < BEMPS_BEACON_BUF_SZ; i++) {
    sem_init(&bemps_shm.comm[i].sched_notif.sem,
             1,   // share across processes
             0);  // starting value
    _reset_comm(&bemps_shm.comm[i]);
  }

  pthread_mutexattr_init(&bemps_shm.gen->lockattr);
  pthread_condattr_init(&bemps_shm.gen->condattr);

  pthread_mutexattr_setpshared(&bemps_shm.gen->lockattr,
                               PTHREAD_PROCESS_SHARED);
  pthread_condattr_setpshared(&bemps_shm.gen->condattr, PTHREAD_PROCESS_SHARED);

  pthread_mutex_init(&bemps_shm.gen->lock, &bemps_shm.gen->lockattr);
  pthread_cond_init(&bemps_shm.gen->cond, &bemps_shm.gen->condattr);

  return &bemps_shm;
}
