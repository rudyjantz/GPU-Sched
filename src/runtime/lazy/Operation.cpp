
#include "Operation.h"
#include <iostream>

cudaError_t MallocOp::perform() {
    uint64_t* fake_addr = (uint64_t *)*ptr_holder;
    cudaError_t err = cudaMalloc(ptr_holder, devMem->size);
    devMem->ptr = *ptr_holder;
#if DEBUG 
    fprintf(stderr, "Perform Actual cudaMalloc (holder: %p, fake addr: %p, valid addr: %p)\n",  ptr_holder, fake_addr, devMem->ptr); 
#endif
    return err;
}

cudaError_t MemcpyOp::perform() {
#if DEBUG
    fprintf(stderr, "Perform Actual cudaMemcpy (src: %p, dst: %p, size %ld)\n", src, devMem->ptr, size);
#endif
    return cudaMemcpy(devMem->ptr, src, size, cudaMemcpyHostToDevice);
}
