#makefile

include ../../common/make.config

NVCC := $(CUDA_DIR)/bin/nvcc
CC := clang
FLAGS := -O1 -Xclang -load -Xclang $(BEMPS_PASS) 

INCLUDE := $(CUDA_DIR)/include

all: naive float

naive: ex_particle_CUDA_naive_seq.cu
	$(NVCC) --cuda $< -o /tmp/ex_particle_CUDA_naive_seq.ii
	$(CC) $(FLAGS) /tmp/ex_particle_CUDA_naive_seq.ii -I$(INCLUDE) -L$(CUDA_LIB_DIR) -L$(BEMPS_LIB_DIR) -lcudart -lm -lrt -lbemps -o particlefilter_naive
	
float: ex_particle_CUDA_float_seq.cu
	$(NVCC) --cuda $< -o /tmp/ex_particle_CUDA_float_seq.ii
	$(CC) $(FLAGS) /tmp/ex_particle_CUDA_float_seq.ii -I$(INCLUDE) -L$(CUDA_LIB_DIR) -L$(BEMPS_LIB_DIR) -lcudart -lm -lrt -lbemps -o particlefilter_float

clean:
	rm particlefilter_naive particlefilter_float
