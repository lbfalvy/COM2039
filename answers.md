# Question 1

## (a)

```
a >> b = floor(a / (2^b))
12 >> 2 = 12 / (2^2) = 12 / 4 = 3
```

## (b)

```
p = .75
s = 1000
time = (1 - p) + p / s
    = .25 + .75 / 1000 = .25 + .00075 = .25075
speedup = 1 / time = 3.99
```

## (c)

100; 10 blocks with 10 threads each

## (d)

A: (M, K) matrix

B: (K, N) matrix

arithmetic intensity = operations / (memory reads + memory writes)
intensity of (A * B) = (M * N * K) / ((M*N + N*K) + (M*K))

## (e)

III.

`(blockIdx.x * blockDim + threadIdx.x)` iterates over integers

`*2` ensures that the next index is also allocated to each thread

## (f)

III.

Three blocks of 512 threads each can be allocated, completely utilizing the device

## (g)

1024 = 32x32, therefore this is the optimal block shape

```
400/32 = 12.5
900/32 = 28.125
```

therefore the corresponding grid size is 13x29, with 41 blocks hitting bounds in a
total of 4 combinations.

# Question 3

## (a)

Max degree of concurrency is 8 which is the number of leaves

Critical path length is 6, the height of the tree

Avg degree of concurrency is 4/3, the ratio of the above numbers

## (b)

Max degree of concurrency is 8, which is the number of leaves

Critical path length is 7, the height of the tree

Avg degree of concurrency is 8/7, the ratio of the above numbers

# Question 4

## (b)

The kernel will access global memory about N*log(N) times, which is the
total number of memory accesses in the algorithm. I originally
intended for it to work on shared memory, but I only realized far too
late that this requires additional modifications and I never managed to
fix some strange bugs in the shared memory based implementation. The
buggy implementation is attached to this document.

## (c)

I added a condition whereby only half of threads does any work on each
iteration, as the upper half would calculate discarded subresults.

## (d)

The shared memory based implementation could be fixed, the idle threads
could exit early. The loops could either be completely unrolled or Duff's
device could potentially be used unless such optimizations are already
performed by the compiler (as they often are in regular C).

With some experimentation we could measure whether it's worth it to
eliminate the bounds check by padding the array up to the nearest
multiple of the block size, this would introduce some redundant
calculations but they're very likely less wasteful on average than the
frequent comparisons and conditionals.