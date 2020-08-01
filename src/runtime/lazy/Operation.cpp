
#include "Operation.h"
#include <iostream>

cudaError_t MallocOp::perform() {
    uint64_t* fake_addr = (uint64_t *)*ptr_holder;
    cudaError_t err = cudaMalloc(ptr_holder, devMem->size);
    devMem->ptr = *ptr_holder;
    // fprintf(stderr, "Perform Actual cudaMalloc (holder: %p, fake addr: %p, valid addr: %p)\n",  ptr_holder, fake_addr, devMem->ptr); 
    return err;
}

cudaError_t MemcpyOp::perform() {
    // std::cerr << "Perform Actual cudaMemcpy (src: " << src << ", dst: " << devMem->ptr << ", size: " << size << ")\n";
    return cudaMemcpy(devMem->ptr, src, size, cudaMemcpyHostToDevice);
}
