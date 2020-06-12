export

.PHONY: all mkdirs clean bemps sched

ROOT_PRJDIR     := $(CURDIR)
ROOT_SRCDIR     := $(ROOT_PRJDIR)/src
ROOT_OBJDIR     := $(ROOT_PRJDIR)/obj

BEMPS_SRCDIR    := $(ROOT_SRCDIR)/bemps
SCHED_SRCDIR    := $(ROOT_SRCDIR)/sched

BEMPS_OBJDIR    := $(ROOT_OBJDIR)/bemps
SCHED_OBJDIR    := $(ROOT_OBJDIR)/sched

CUDA_DIR        := /usr/local/cuda
CUDA_INC        := $(CUDA_DIR)/include
CUDA_LIB        := $(CUDA_DIR)/lib64

IFLAGS          := -I$(ROOT_SRCDIR)


ifeq ($(GPU_RES),)
	GPU_RES := GPU_RES_SLOP
endif

#debug: CFLAGS := $(CFLAGS_DBG)
#debug: all
#.PHONY: debug
CC         := gcc
CXX        := g++
CFLAGS     := -g -Wall -O3  # FIXME load the warning flags: -Werror, etc.
CFLAGS_DBG := -g -Wall -O0 -DBEMPS_DBG -DSCHED_DBG

CFLAGS += $(IFLAGS) $(LFLAGS)



all: mkdirs bemps sched

bemps:
	make -C $(BEMPS_SRCDIR)

sched:
	make -C $(SCHED_SRCDIR)

mkdirs: $(BEMPS_OBJDIR) $(SCHED_OBJDIR)


$(ROOT_OBJDIR)/%:
	mkdir -p $@


clean:
	rm -rf $(ROOT_OBJDIR)
