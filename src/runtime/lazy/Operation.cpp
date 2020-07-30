
#include "Operation.h"
#include <iostream>

cudaError_t MallocOp::perform() {
    std::cerr << "Perform Actual cudaMalloc (fake addr: " << (uint64_t *)*ptr_holder << ", size: " << devMem->size << ")\n";
    cudaError_t err = cudaMalloc(ptr_holder, devMem->size);
    devMem->ptr = *ptr_holder;
    return err;
}

cudaError_t MemcpyOp::perform() {
    std::cerr << "Perform Actual cudaMemcpy (src: " << src << ", dst: " << devMem->ptr << ", size: " << size << ")\n";
    return cudaMemcpy(devMem->ptr, src, size, cudaMemcpyHostToDevice);
}
