// #include "lazy.h"

#include "Core.h"

// the fake virtual address for device, actually this
// virtual address segment is reserved for linux, so
// it is safe to assume it is not valid GPU address
static uint64_t next_fake_addr = 0xffff800000000000;

void Runtime::registerMallocOp(void** holder, size_t size) {
  uint64_t fake_addr = next_fake_addr++;
  auto* obj = new MObject((void*)fake_addr, size);
  auto* op = new MallocOp(holder, obj);

  Objects[fake_addr] = obj;
  Pendding[fake_addr].push_back(op);

  // return an fake address
  *holder = (void*)fake_addr;
}

void Runtime::registerMemcpyOp(void* dst, void* src, size_t size) {
  uint64_t fake_addr = (uint64_t)dst;
  auto* obj = Objects[fake_addr];
  auto* op = new MemcpyOp(src, obj, size);
  Pendding[fake_addr].push_back(op);
}

int64_t Runtime::getAggMemSize() {
  size_t tot = 0;
  for (auto obj : Objects) tot += obj.second->size;
  return (int64_t)tot;
}

cudaError_t Runtime::prepare() {
  cudaError_t err = cudaSuccess;
  for (auto it : Pendding)
    for (auto op : it.second) {
      err = op->perform();
      delete op;
      if (err != cudaSuccess) return err;
    }

  for (auto obj : Objects) {
    ActiveObjects.insert((uint64_t)obj.second->ptr);
    delete obj.second;
  }

  Pendding.clear();
  Objects.clear();
  return err;
}

cudaError_t Runtime::free(void* ptr) {
  ActiveObjects.erase((uint64_t)ptr);
  if (ActiveObjects.empty()) enableIssue();
  return cudaFree(ptr);
}


