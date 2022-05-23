/************************************* // block comment
**Question 2e
*************************************/
#include <stdio.h> // input-output header
#include <cuda.h> // CUDA header

// Function to initialize each element of a with its index
void init(int *a, int N) // a is an array of ints, N is the length of a
{
  int i; // Iterator
  for (i = 0; i < N; ++i) // Iterate over indices of a
  {
    a[i] = i; // Initialize each 
  }
}

/** CUDA kernel which takes an array of ints and its length as arguments
 * and multiplies each element of the array by 2
 */
__global__ void doubleElements(int *a, int N)
{
  int i; // declare i at the top of the function like in CPL
  i = blockIdx.x * blockDim.x + threadIdx.x; // Assign sequential integers to threads
  if (i < N) // branch to avoid out-of-bounds indexing of a
  {
    a[i] *= 2; // Multiply element of a by 2
  }
}

/** Function which takes an array and its length
 * and decides whether each element's value is the double of its index
 */
bool checkElementsAreDoubled(int *a, int N)
{
  int i; // Forward-declare i
  for (i = 0; i < N; ++i) // Iterate over indices of a
  {
    if (a[i] != i*2) return false; // Early return if the assertion is void
  }
  // In absence of an incorrect element we can conclude that the assertion is true
  return true;
}

int main() // Entry point
{
  int N = 100; // Constant
  int *a; // Uninitialized pointer
  // The size of an int array of N elements
  size_t size = N * sizeof(int); // sizeof includes padding
  // Dynamically allocate virtual memory that is accessible to both the CUDA kernel
  // and the host. The function initializes a
  cudaMallocManaged(&a, size);

  init(a, N); // Initialize each element of a with its index

  // Configure kernel with 100 threads. Since the kernel does bounds checks, it's
  // possible to call it on arrays of inconvenient size
  size_t threads_per_block = 10;
  size_t number_of_blocks = 10;
  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
  cudaDeviceSynchronize(); // Wait for all threads
  // Assert that all elements of the array have been doubled
  bool areDoubled = checkElementsAreDoubled(a, N);
  // Print the conclusion
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");
  
  cudaFree(a); // Free the array with CUDA, since we allocated it with CUDA
}
