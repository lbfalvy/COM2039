/************************************* // block comment
**Question 2c
*************************************/
#include <stdio.h> // input-output header
#include <cuda.h> // CUDA header

__global__ void loop() // CUDA kernel
{
  printf("This is iteration number %d\n", threadIdx.x); // Print the current thread's index
}

int main() // Entry point
{
  int N = 10; // Constant
  loop<<<1, N>>>(); // Configure kernel for N threads
  cudaDeviceSynchronize(); // Wait for all threads to complete
}
