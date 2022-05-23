/************************************* // block comment
**Question 2a
*************************************/
#include <stdio.h> // input-output header
#include <cuda.h> // CUDA header

// host function declaration
void // indicating absence of return value
helloCPU // function name
( // start of parameter list
// empty parameter list
) // end of parameter list
{ // start of function body
  // prints "Hello from the CPU." followed by a newline
  printf // formatted print function from the stdio header included above
  ( // start of invocation parameters
    "Hello from the CPU.\n" // template, notice no placeholders
    // no template values
  ) // end of invocation parameters
  ; // instruction terminator semicolon
} // end of function body

__global__ void helloGPU() // CUDA kernel
{ // start of function body
  printf("Hello from the GPU.\n"); // prints "Hello from the GPU." followed by a newline
} // end of function body

int main() // entry point
{ // start of function body
  helloGPU<<<1, 1>>>(); // kernel configured for a single thread
  cudaDeviceSynchronize(); // Wait for kernel

  helloCPU(); // host function called
} // end of function body