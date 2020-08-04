
#include "Operation.h"

#include <iostream>

cudaError_t MallocOp::perform() {
  uint64_t* fake_addr = (uint64_t*)devMem->ptr;
  cudaError_t err = cudaMalloc(&(devMem->ptr), devMem->size);
#if DEBUG
  fprintf(stderr, "Perform Actual cudaMalloc (fake addr: %p, valid addr: %p)\n",
          fake_addr, devMem->ptr);
#endif
  return err;
}

cudaError_t MemcpyOp::perform() {
  cudaError_t err = cudaMemcpy(devMem->ptr, src, size, kind);
#if DEBUG
  fprintf(stderr,
          "Perform Actual cudaMemcpy (dst: %p, src: %p, size: %ld, kind: %d, Success: %d)\n",
          devMem->ptr, src, size, kind, err == cudaSuccess );
#endif
  return err;
}

cudaError_t MemcpyToSymbolOp::perform() {
  return cudaMemcpyToSymbol(symbol, buf, count, offset, kind);
  free(buf);
}
