
#ifndef _LAZY_CUDA_OP_H_
#define _LAZY_CUDA_OP_H_

#include "Core.h"

static Runtime R;

extern "C" cudaError_t cudaMallocWrapper(void** devPtr, size_t size) {
  R.registerMallocOp(devPtr, size);
  return cudaSuccess;
}

extern "C" cudaError_t cudaMemcpyWrapper(void* dst, const void* src, size_t count,
                              enum cudaMemcpyKind kind) {
  if (kind != cudaMemcpyHostToDevice) return cudaMemcpy(dst, src, count, kind);

  R.registerMemcpyOp(dst, (void*)src, count);
  return cudaSuccess;
}

extern "C" cudaError_t cudaKernelLaunchPrepare(uint64_t gxy, int gz, uint64_t bxy,
                                    int bz) {
  // TODO: here add code to call beacon
  static int id = 0;
#define U32X(v) (int)((v & 0xFFFFFFFF00000000LL) >> 32)
#define U32Y(v) (int)(v & 0xFFFFFFFFLL)
  int gx = U32X(gxy);
  int gy = U32Y(gxy);
  int bx = U32X(bxy);
  int by = U32Y(bxy);
  int64_t membytes = R.getAggMemSize();

  printf("gx: %d, gy: %d, gz: %d, bx: %d, by: %d, bz: %d, mem: %ld, toIssue: %d\n", gx, gy,
         gz, bx, by, bz, membytes, R.toIssue());

  if (R.toIssue()) {
    // bemps_begin(id, gx, gy, gz, bx, by, bz, membytes);
    R.disableIssue();
  }

  return R.prepare();
}

extern "C" cudaError_t cudaFreeWrapper(void* devPtr) { return R.free(devPtr); }

// cudaError_t cudaMemcpyToSymbolWrapper(const char* symbol, const void* src,
//                                       size_t count, size_t offset,
//                                       enum cudaMemcpyKind kind) {
//   return cudaMemcpyToSymbol(symbol, src, count, offset, kind);
// }

// ​cudaError_t cudaMallocHostWrapper(void** ptr, size_t size) {
//   return cudaMallocHost(ptr, size);
// }

#endif