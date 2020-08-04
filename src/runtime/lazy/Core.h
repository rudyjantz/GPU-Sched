#ifndef _CORE_H_
#define _CORE_H_

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../bemps/bemps.hpp"
#include "Operation.h"

using namespace std;

class Runtime {
 private:
  bool issue;  // need to issue an beacon ?
  unordered_map<uint64_t, MObject*> MemObjects;
  unordered_map<uint64_t, std::vector<Operation*>> CudaMemOps;

  unordered_set<uint64_t> ActiveObjects;
  unordered_map<uint64_t, uint64_t> AllocatedMap;
  unordered_map<uint64_t, uint64_t> ReverseAllocatedMap;
  std::vector<Operation*> DeviceDependentOps;

 public:
  Runtime() : issue(true) {}
  bool toIssue() { return issue; }
  void enableIssue() { issue = true; }
  void disableIssue() { issue = false; }

  bool isAllocated(void* ptr);
  void* getValidAddrforFakeAddr(void* ptr);

  void registerMallocOp(void** holder, size_t size);
  void registerMemcpyOp(void* dst, void* src, size_t size,
                        enum cudaMemcpyKind k);
  void registerMemcpyToSymbleOp(char* symble, void* src, size_t s, size_t o,
                                enum cudaMemcpyKind kind);
  int64_t getAggMemSize();
  cudaError_t prepare();
  cudaError_t free(void* devPtr);
};

#endif