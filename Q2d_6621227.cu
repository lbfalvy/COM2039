/************************************* // block comment
**Question 2c
*************************************/
#include <stdio.h> // input-output header
#include <cuda.h> // CUDA header

__global__ void loop() // CUDA kernel
{
  // Prints the current thread's index, offset by the group index, to standard output
  printf("This is iteration number %d\n", blockIdx.x * blockDim.x + threadIdx.x);
}

int main() // Entrypoint
{
  int N = 10; // Constant
  // Split the total workload across two blocks, while ensuring the right number of threads
  loop<<<2, N/2>>>(); // TODO: only works for N divisible by 2
  cudaDeviceSynchronize(); // Wait for threads to complete
}
