#ifndef BEMPS_HPP
#define BEMPS_HPP

#include <semaphore.h>

#define BEMPS_BEACON_BUF_SZ 0x1000

#define BEMPS_SCHED_TIMEOUT_NS 100000000UL  // 100ms
//#define BEMPS_SCHED_TIMEOUT_NS 5000000000UL // 5s
//#define BEMPS_SCHED_TIMEOUT_NS 10000000000UL // 10s

// The beacon type for an application calling the bemps library.
typedef struct {
  long mem_B;
  long cores;
  // TODO: constant memory
} bemps_beacon_t;

typedef struct {
  sem_t sem;  // for signaling the process to launch its kernel
  int device_id;
} bemps_sched_notif_t;

typedef enum {
  BEMPS_BEACON_STATE_COMPLETED_E = 0,
  BEMPS_BEACON_STATE_BEACON_FIRED_E,
  BEMPS_BEACON_STATE_SCHEDULER_READ_E,
  BEMPS_BEACON_STATE_SCHEDULED_E
} bemps_beacon_state_e;

typedef struct {
  long long timestamp_ns;
  pid_t pid;
  int age;
  int exit_flag;
  bemps_beacon_state_e state;
  bemps_beacon_t beacon;
  bemps_sched_notif_t sched_notif;
} bemps_shm_comm_t;

// A single type that captures all shared memory pointers between the bemps
// library and the scheduler
typedef struct {
  // The head index into the circuluar buffer of beacons
  int beacon_q_head;

  // The tail index into the circuluar buffer of beacons
  int beacon_q_tail;

  // For batching algorithms, the size of the batch before we signal the
  // scheduler.
  int max_batch_size;

  // Lock and cond var for signaling the scheduler
  pthread_cond_t cond;
  pthread_mutex_t lock;

  // attributes probably don't need to be in shm...
  pthread_condattr_t condattr;
  pthread_mutexattr_t lockattr;

} bemps_shm_gen_t;

typedef struct {
  bemps_shm_gen_t *gen;
  bemps_shm_comm_t *comm;
} bemps_shm_t;

/*
 * Notify BEMPS of an upcoming GPU task.
 */
void bemps_beacon(int bemps_tid, bemps_beacon_t *bemps_beacon);

extern "C" {
void bemps_begin(int id, int gx, int gy, int gz, int bx, int by, int bz,
                 int64_t memsize);
}

/*
 * Set the callback for the upcoming cuda call.
 */
// void bemps_set_cuda_callback(int bemps_tid);

/*
 * Set the GPU device based on the bemps task id.
 */
void bemps_set_device(int bemps_tid);

/*
 * Free scheduler and GPU resources for some bemps task.
 */
extern "C" {
void bemps_free(int bemps_tid);
}

/*
 * Initialization function for the BEMPS scheduler.
 */
bemps_shm_t *bemps_sched_init(int max_batch_size);

/*
 * Initialization function for processes that are using BEMPS.
 */
int bemps_init(void);

#endif
