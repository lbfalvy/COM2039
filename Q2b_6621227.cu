/************************************* // block comment
**Question 2b
*************************************/
#include <stdio.h> // input-output header
#include <cuda.h> // CUDA header

__global__ void printSuccessForCorrectExecutionConfiguration() // kernel definition
{
  // The very last thread of the 256'th block, assuming maximum thread count
  if(threadIdx.x == 1023 && blockIdx.x == 255)
  {
    printf("Success!\n"); // Writes "Success!" to standard output, followed by a newline
  }
  else { // All threads other than the last of the 256'th block
    // Writes failure message to standard output
    printf("Failure. Update the execution configuration as necessary.\n");
  }
}

int main() // Entry point
{
  /** Kernel configuration that prints "Success!" in the last line.
   * Rather confusingly, it also prints over two hundred thousand instances
   * of the failure message, which lead me to spend more time researching
   * how to offset the thread and block indices than I did writing comments.
   */
  printSuccessForCorrectExecutionConfiguration<<<256, 1024>>>();
  cudaDeviceSynchronize(); // Wait for kernel
}
