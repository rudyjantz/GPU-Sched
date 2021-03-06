
GPU-Sched is a collection of libraries and a scheduler. The libraries include
LLVM passes, as well as runtime libraries for programs built with these passes.
The scheduler is a single binary that includes a few prototypes for interacting
with applications bound for the GPU, and which could benefit from sharing.



# Requirements
* cmake (tested with 3.15.4)
* llvm (tested with 11.0.0)
* cuda (tested with 11.2)



# Building
    $ git clone git@github.com:rudyjantz/GPU-Sched.git
    $ cd GPU-sched
    $ mkdir build
    $ cd build
    $ cmake ../src -DDEBUG_LAZY=1
    $ make



# Compiling applications with GPU-Sched's passes

One Rodinia example for compiling with the GPU-Sched toolchain is backprop:

    Benchmarks/rodinia_cuda_3.1/cuda/backprop/Makefile
    Benchmarks/rodinia_cuda_3.1/common/make.config

For the darknet example, refer to 

    Benchmarks/darknet/Makefile

In brief, you'll need to build with the libWrapperPass or libGPUBeaconPass
(either with opt or clang), e.g.

    opt -load libWrapperPass.so -WP <foo.bc >foo_mod.bc

or:

    clang -Xclang -load -Xclang libGPUBeaconPass.so foo.ii -c

And you'll need to link with the lazy runtime library and bemps:

    clang -llazy -lbemps



# Running the scheduler

    $ ./bemps_sched -h

    Usage:
        ./bemps_sched <which_scheduler> [jobs_per_gpu]

        which_scheduler is one of:
          zero, single-assignment, cg, mgb_basic, or mgb

        jobs_per_gpu is required and only valid for cg; it is an int that
        specifies the maximum number of jobs that can be run a GPU



# Example

Terminal 1:

    $ cd example
    $ ../build/runtime/sched/bemps_sched zero

Terminal 2:

    $ cd example
    $ LD_LIBRARY_PATH=../build/runtime/bemps bash workload.sh
