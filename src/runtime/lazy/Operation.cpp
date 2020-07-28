
#include "Operation.h"

cudaError_t MallocOp::perform() {
    cudaError_t err = cudaMalloc(ptr_holder, devMem->size);
    devMem->ptr = *ptr_holder;
    return err;
}

cudaError_t MemcpyOp::perform() {
    return cudaMemcpy(devMem->ptr, src, size, cudaMemcpyHostToDevice);
}
