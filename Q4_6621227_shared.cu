#include <stdio.h> // input-output header
#include <cuda.h> // CUDA header

// Create an array of successive integers that's accessible to both the kernel and the host
int* gen_array(int N) {
  // Allocate an array that's accessible to both the kernel and the host
  int* ary;
  size_t space = sizeof(int)*N;
  cudaMallocManaged(&ary, space);
  // Populate the array with its indices
  for (int i = 0; i < N; i++) {
    ary[i] = i;
  }
  // Return the pointer
  return ary;
}

// 1D CUDA Kernel to sum 2*blockDim.x subsections of an array into their first element
// NOTICE For correct operation, the block size must be a power of 2
__global__ void sumBlocks(int* array, int N)
{
  extern __shared__ int sdata[];
  int tid = threadIdx.x;
  // 1024 threads can sum 2048 numbers in the first step
  int idx = blockIdx.x * ( blockDim.x * 2 ) + tid;
  if (idx < N) {
    sdata[tid] = array[idx];
    if (idx + blockDim.x < N) {
      sdata[tid] += array[idx + blockDim.x];
    }
  }
  __syncthreads();
  // descend through powers of 2
  // We could find a better number to start from for very small arrays
  // but for very small arrays starting up a CUDA kernel is the performance bottleneck
  for (int s = blockDim.x/2; s > 0; s /= 2) { // /= 2 will compoile to shift 
    // Assert that this thread is in the working batch AND avoid overindexing
    if (tid < s && idx + s < N) {
      sdata[tid] += sdata[tid + s];
    } // TODO else thread is idle and should quit
    __syncthreads();
  }
  if (tid == 0) {
    array[idx] = sdata[tid];
  }
  __syncthreads();
}

// Sum an array using CUDA
int cuda_sum_array(int* array, int N) {
  int blockSize = 1024; // MUST be a power of 2
  // One block for every 2048 entries, rounded up
  int blockCnt = N/(2*blockSize);
  if (N % (2*blockSize) > 0) blockCnt++;
  // Run kernel
  sumBlocks<<<blockCnt, blockSize>>>(array, N);
  cudaDeviceSynchronize();
  for (int i = 0; i < N; i++) {
    printf("%d: %d", i, array[i]);
    if (i % 8 == 7) printf("\n");
    else printf("\t");
  }
  if (N % 8 != 7) printf("\n");
  // Sum subresults on host
  int acc = 0;
  for (int i = 0; i < N; i += (2*blockSize)) {
    acc += array[i];
  }
  return acc;
}

int main()
{
  // Obtain an integer from the user
  int count;
  printf("Enter the number of integers you wish to sum: ");
  scanf("%d", &count);
  // generate the array
  int* buf = gen_array(count);
  // Produce the sum
  int sum = cuda_sum_array(buf, count);
  // Print the final sum
  printf("The final sum is %d\n", sum);
}