
#ifndef _LAZY_CUDA_OP_H_
#define _LAZY_CUDA_OP_H_

#include "Core.h"

static Runtime R;
static int id = 0;

extern bool is_fake_addr(void* ptr);

extern "C" cudaError_t cudaMallocWrapper(void** devPtr, size_t size) {
  R.registerMallocOp(devPtr, size);
#if DEBUG
  fprintf(stderr, "Delay a cudaMalloc (holder: %p, fake addr: %p)\n", devPtr,
          *devPtr);
#endif
  return cudaSuccess;
}

extern "C" cudaError_t cudaMemcpyWrapper(void* dst, void* src, size_t count,
                                         enum cudaMemcpyKind kind) {
  if (is_fake_addr(dst) && R.isAllocated(dst))
    dst = R.getValidAddrforFakeAddr(dst);

  if (is_fake_addr(src) && R.isAllocated(src))
    src = R.getValidAddrforFakeAddr(src);

  if ((kind == cudaMemcpyHostToDevice && !is_fake_addr(dst)) ||
      (kind == cudaMemcpyDeviceToDevice && !is_fake_addr(dst) &&
       !is_fake_addr(src)) ||
      kind == cudaMemcpyDeviceToHost) {
#if DEBUG
    fprintf(stderr,
            "perform a cudaMemcpy operation (dst: %p, src: %p, kind: %d)\n", dst,
            src, kind);
#endif
    return cudaMemcpy(dst, src, count, kind);
  } else {
    R.registerMemcpyOp(dst, src, count, kind);
#if DEBUG
    fprintf(stderr, "delayed a cudaMemcpy operation (dst: %p, src: %p, kind: %d)\n",
            dst, src, kind);
#endif
    return cudaSuccess;
  }
}

extern "C" cudaError_t cudaMemcpyToSymbolWrapper(char* sym, void* src,
                                                 size_t count, size_t offset,
                                                 enum cudaMemcpyKind kind) {
  R.registerMemcpyToSymbleOp(sym, src, count, offset, kind);
  return cudaSuccess;
}

extern "C" cudaError_t cudaKernelLaunchPrepare(uint64_t gxy, int gz,
                                               uint64_t bxy, int bz) {
  // TODO: here add code to call beacon
#define U32X(v) (int)((v & 0xFFFFFFFF00000000LL) >> 32)
#define U32Y(v) (int)(v & 0xFFFFFFFFLL)
  int gx = U32X(gxy);
  int gy = U32Y(gxy);
  int bx = U32X(bxy);
  int by = U32Y(bxy);
  int64_t membytes = R.getAggMemSize();

#if DEBUG
  printf(
      "A new kernel launch: \n\tgx: %d, gy: %d, gz: %d, bx: %d, by: %d, bz: "
      "%d, mem: %ld, toIssue: %d\n",
      gx, gy, gz, bx, by, bz, membytes, R.toIssue());
#endif

  if (R.toIssue()) {
    // bemps_begin(id, gx, gy, gz, bx, by, bz, membytes);
    R.disableIssue();
  }
  return R.prepare();
}

extern "C" cudaError_t cudaFreeWrapper(void* devPtr) {
  cudaError_t err = R.free(devPtr);
  if (R.toIssue()) {
    // bemps_free(id);
    id++;
  }
  return err;
}

// cudaError_t cudaMemcpyToSymbolWrapper(const char* symbol, const void* src,
//                                       size_t count, size_t offset,
//                                       enum cudaMemcpyKind kind) {
//   return cudaMemcpyToSymbol(symbol, src, count, offset, kind);
// }

// ​cudaError_t cudaMallocHostWrapper(void** ptr, size_t size) {
//   return cudaMallocHost(ptr, size);
// }

#endif