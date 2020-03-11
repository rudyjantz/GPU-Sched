/**
 * This is a toy program from CUDA tutorial, it is to compute the addition on
 * two square matrixes, Here I use it to study some basic built-in data
 * structures provided by CUDA
 */

#include <math.h>
#include <stdio.h>

#define N 512
#define BLK 32

__global__ void VecAdd(float *A, float *B, float *C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  C[i] = A[i] + B[i];
}

int main(int argc, char **argv) {
  // A and B are input data, C is to store output data
  float A[N], B[N], C[N];

  size_t size = N * sizeof(float);

  printf("size: %ld\n", size);

  // initialize the input data
  for (int i = 0; i < N; i++) {
    A[i] = sin(i) * sin(i);
    B[i] = cos(i) * cos(i);
  }

  // correspoinding memory for data in device
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  dim3 ThreadsPerBlock(BLK);
  dim3 BlocksPerGrid(N / BLK);
  VecAdd<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C);

  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < BLK; i++) {
    printf("%.2f ", C[i]);
  }
  printf("\n");

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  cudaMalloc(&d_A, N * sizeof(float));
  cudaMalloc(&d_B, N * sizeof(float));
  cudaMalloc(&d_C, N * sizeof(float));

  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}