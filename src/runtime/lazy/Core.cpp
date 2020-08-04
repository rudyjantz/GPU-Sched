// #include "lazy.h"

#include "Core.h"

// the fake virtual address for device, actually this
// virtual address segment is reserved for linux, so
// it is safe to assume it is not valid GPU address
static uint64_t next_fake_addr = 0xffff800000000000;

bool is_fake_addr(void* ptr) {
  return (uint64_t)ptr >= 0xffff800000000000;
}


bool Runtime::isAllocated(void* ptr) {
  return AllocatedMap.count((uint64_t)(ptr)) > 0;
}

void* Runtime::getValidAddrforFakeAddr(void* ptr) {
  assert(isAllocated(ptr));
  uint64_t fake_addr = (uint64_t)ptr;
  return (void*)AllocatedMap[fake_addr];
}

void Runtime::registerMallocOp(void** holder, size_t size) {
  uint64_t fake_addr = next_fake_addr++;
  auto* obj = new MObject((void*)fake_addr, size);
  auto* op = new MallocOp(obj);

  MemObjects[fake_addr] = obj;
  CudaMemOps[fake_addr].push_back(op);

  // return an fake address
  *holder = (void*)fake_addr;
}

void Runtime::registerMemcpyOp(void* dst, void* src, size_t size, enum cudaMemcpyKind k) {
  uint64_t fake_addr; 
  if (is_fake_addr(dst)) fake_addr = (uint64_t)dst;
  else fake_addr = (uint64_t)src;
  auto* obj = MemObjects[fake_addr];
  auto* op = new MemcpyOp(src, obj, size, k);
  CudaMemOps[fake_addr].push_back(op);
}

void Runtime::registerMemcpyToSymbleOp(char* sym, void* src, size_t s, size_t o,
                                       enum cudaMemcpyKind k) {
  assert(k == cudaMemcpyHostToDevice);
  auto* op = new MemcpyToSymbolOp(sym, src, s, o, k);
  DeviceDependentOps.push_back(op);
}

int64_t Runtime::getAggMemSize() {
  size_t tot = 0;
  for (auto obj : MemObjects) tot += obj.second->size;
  return (int64_t)tot;
}

cudaError_t Runtime::prepare() {
  cudaError_t err = cudaSuccess;
  // perform actual memory alloc operations
  for (auto it: CudaMemOps) {
    auto op = it.second[0];
    err = op->perform();
    delete op; 
    if (err != cudaSuccess) return err;
  }

  // perform depended memory operations
  for (auto it : CudaMemOps)
    for (auto i = 1; i < it.second.size(); i++) {
      auto op = it.second[i];
      err = op->perform();
      delete op;
      if (err != cudaSuccess) return err;
    }

  // update the core map structure to track the status of GPU mem.
  // and cleanup the MObjects
  for (auto obj : MemObjects) {
    uint64_t fake_addr = obj.first;
    uint64_t valid_addr = (uint64_t)(obj.second->ptr);

    ActiveObjects.insert(valid_addr);
    AllocatedMap[fake_addr] = valid_addr;
    ReverseAllocatedMap[valid_addr] = fake_addr;
    delete obj.second;
  }

  // perform other device dependent operations, e.g., cudaMemcpyToSytmble
  for (auto op: DeviceDependentOps) {
    op->perform();
    delete op;
  }

  CudaMemOps.clear();
  MemObjects.clear();
  DeviceDependentOps.clear();
  return err;
}

cudaError_t Runtime::free(void* ptr) {
  uint64_t valid_addr = (uint64_t)(ptr);
  uint64_t fake_addr = ReverseAllocatedMap[valid_addr];
  ReverseAllocatedMap.erase(valid_addr);
  AllocatedMap.erase(fake_addr);
  ActiveObjects.erase(valid_addr);
  if (ActiveObjects.empty()) enableIssue();
  // std::cerr << "perform cudaFree (toIssue for next kernel launch: " << issue
  // << ")\n";
  return cudaFree(ptr);
}
