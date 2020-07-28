#ifndef __OPERATION_H_
#define __OPERATION_H_

#include <cuda_runtime.h>

#include <iostream>

typedef enum oper { CUDA_MALLOC, CUDA_MEMCPY } opr_t;

// MOject is to represent a memory region
struct MObject {
  void *ptr;
  size_t size;
  MObject(void *ptr, size_t s) : ptr(ptr), size(s) {}
};

class Operation {
 private:
  opr_t op;

 protected:
  MObject *devMem;

 public:
  Operation(opr_t op, MObject *obj) : op(op), devMem(obj) {}
  bool isMalloc() { return op == CUDA_MALLOC; }
  bool isMemcpy() { return op == CUDA_MEMCPY; }
  virtual cudaError_t perform() = 0;
};

class MallocOp : public Operation {
 private:
  void **ptr_holder;

 public:
  MallocOp(void **holder, MObject *obj)
      : ptr_holder(holder), Operation(CUDA_MALLOC, obj) {}
  cudaError_t perform() override;
};

// here we only interested in host to device copy
class MemcpyOp : public Operation {
 private:
  void *src;
  size_t size;
  MObject *dst;

 public:
  MemcpyOp(void *src, MObject *dst, size_t s)
      : src(src), size(s), Operation(CUDA_MEMCPY, dst) {}
  cudaError_t perform() override;
};

#endif