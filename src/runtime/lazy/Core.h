#ifndef _CORE_H_
#define _CORE_H_

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cassert>

#include "../bemps/bemps.hpp"
#include "Operation.h"

using namespace std;

class Runtime {
 private:
  bool issue;  // need to issue an beacon ?
  unordered_set<uint64_t> ActiveObjects;
  unordered_map<uint64_t, uint64_t> AllocatedMap;
  unordered_map<uint64_t, uint64_t> ReverseAllocatedMap;
  unordered_map<uint64_t, MObject*> Objects;
  unordered_map<uint64_t, std::vector<Operation*>> Pendding;

 public:
  Runtime() : issue(true) {}
  bool isAllocated(void* ptr) {
    return AllocatedMap.count((uint64_t)(ptr)) > 0;
  }
  void* getValidAddrforFakeAddr(void* ptr) {
    assert(isAllocated(ptr));
    uint64_t fake_addr = (uint64_t)ptr;
    return (void *)AllocatedMap[fake_addr];
  }
  bool toIssue() { return issue; }
  void enableIssue() { issue = true; }
  void disableIssue() { issue = false; }
  void registerMallocOp(void** holder, size_t size);
  void registerMemcpyOp(void* dst, void* src, size_t size);
  int64_t getAggMemSize();
  cudaError_t prepare();
  cudaError_t free(void* devPtr);
};

#endif