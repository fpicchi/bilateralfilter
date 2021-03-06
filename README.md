# Bilateral Filter

CUDA implementation of [Bilateral Filter](https://en.wikipedia.org/wiki/Bilateral_filter)

## Dependecies

- Cuda
- OpenCV
- OpenMP

## Settings

- Set CMAKE_CUDA_ARCHITECTURES in cmake/GPUConfig.cmake according to your device compute capability.
- Set SPEEDUP_PARALLEL to 1 in src/main.cpp to print the speedup of the CUDA parallel solution in relation with the others.
- Set USE_PARALLEL_NAIVE to 1 in src/main.cpp to use the naive (slow) CUDA version (not recommended).

## Run

```
mkdir build && cd build
cmake ..
make -j8
./bilateral_filter
```

## Implementation details

### Sequential

The sequential version has been provided by Mirco De Marchi, and it's the starting point of the other implementations.
In particular, both the gaussian space generation and the `bf_sequential_apply()` function have remained essentially the same. 

### OpenMP

There were several points were it could've made sense to apply OpenMP directives.

1. During the color weight initialization step
2. During the gaussian space generation step
3. During the filtering step

After various benchmarks it turned out that it wasn't worth using OpenMP in the first 2 cases, as the loops were small enough that the overhead outweighted the benefits compared to a purely sequential approach.  
So it was only applied to the third step, using a simple `#pragma omp parallel for` on the outer `for` loop.

### CUDA Naive

This was the first version made. Basically it parallelized the filtering step by creating a thread for each pixel (in blocks of 1024), each thread accessing the same global memory data.
Even if GPUs have memory coalescing, it turned out to be not that much efficient as it had many global memory accesses (which are slow).

The occupancy was 100%

### CUDA (Shared)

This second version uses shared memory and turned out to be quite fast, even if the occupancy is less than 100%.  

The general idea is dividing the image in `K-squares NxN` and then create `K-CUDA` blocks which are composed of `(N+2*radius)x(N+2*radius)` threads.  
Each thread in a block will load data (mainly pixels) in a shared memory chunk. Then only the most internal threads of the block will be used to actually calculate the new pixel values.

The following picture simulates the CUDA block division of a 25*25 image using 5x5 squares with a radius parameter of 1 (so using CUDA blocks of 7x7 threads).

<img src="sharedmem.png"></img>

The size of the CUDA blocks are calculated in an automatic way. In most GPUs they'll be composed of 1024 threads (32x32).  
Considering the previous formula, it means that to not waste too much resources one should avoid to have a big radius as parameter.

In the benchmarks it was chosen a radius of 4 pixels, which means that for every CUDA block we were only calculating ((32-8) x (32-8)) effective pixel values = 576 pixels.  
This means that **only `56.25%` of the threads in a CUDA block actually contributed to the final result**, while the other 43.75% were only used to load resources from the global memory into the shared memory.

## Benchmark Results (PC)

This filter has been tried on 2 different PCs.

| ID | CPU          | GPU             | OS            |
|----|--------------|-----------------|-------------- |
| A  | AMD FX-6350  | NVIDIA 1060 GTX | Windows 8.1   |
| B  | AMD R5-5600X | NVIDIA 3090 RTX | Ubuntu 20     |

On every computer the runtime of this implementation of the Bilateral Filter turned out to be comparable with OpenCV's (when both ran in parallel mode).  
Also it produced the same picture as OpenCV (checked pixel by pixel) so even from a qualitative point of view we can say that this implementation is correct.

### cones.png (2'435'860 pixels)

| ID | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|----|---------|--------------|-------------------|----------|-----------------|
| A  | 352 ms  | 84 ms        | 91 ms             |  23 ms   |  26 ms          |
| B  | 121 ms  | 15 ms        | 13 ms             |  8 ms    |  7 ms           |


### lenna.jpeg (361'200 pixels)

| ID | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|----|---------|--------------|-------------------|----------|-----------------|
| A  | 50 ms   | 14 ms        | 17 ms             |  5 ms    |  5 ms           |
| B  | 18 ms   | 9 ms         | 7 ms              |  1 ms    |  1 ms           |

### meadows.png (99'300 pixels)

| ID | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|----|---------|--------------|-------------------|----------|-----------------|
| A  | 16 ms   | 6 ms         | 7 ms              |  3 ms    |  3 ms           |
| B  | 5 ms    | 2 ms         | 3 ms              |  1 ms    |  1 ms           |

### mountain.jpg (114'072 pixels)

| ID | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|----|---------|--------------|-------------------|----------|-----------------|
| A  | 21 ms   | 4 ms         | 5 ms              |  3 ms    |  3 ms           |
| B  | 6 ms    | 2 ms         | 4 ms              |  1 ms    |  1 ms           |

### noir.png (111'000 pixels)

| ID | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|----|---------|--------------|-------------------|----------|-----------------|
| A  | 19 ms   | 4 ms         | 4 ms              |  3 ms    |  4 ms           |
| B  | 5 ms    | 2 ms         | 1 ms              |  1 ms    |  1 ms           |

### rosso.png (144'000 pixels)

| ID | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|----|---------|--------------|-------------------|----------|-----------------|
| A  | 25 ms   | 6 ms         | 7 ms              |  3 ms    |  3 ms           |
| B  | 7 ms    | 2 ms         | 3 ms              |  1 ms    |  2 ms           |

### rubik.png (230'337 pixels)

| ID | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|----|---------|--------------|-------------------|----------|-----------------|
| A  | 39 ms   | 10 ms        | 8 ms              |  4 ms    |  3 ms           |
| B  | 13 ms   | 2 ms         | 5 ms              |  2 ms    |  2 ms           |

### sky.png (132'300 pixels)

| ID | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|----|---------|--------------|-------------------|----------|-----------------|
| A  | 23 ms   | 5 ms         | 7 ms              |  3 ms    |  3 ms           |
| B  | 7 ms    | 2 ms         | 3 ms              |  2 ms    |  1 ms           |

As expected, the parallel versions performed better than the sequential version.  
And the GPU versions performed better than the CPU (parallel) ones. The speedup gets more noticeable with bigger images.

## Benchmark Results (Jetson)

### cones.png (2'435'860 pixels)

| Jetson     | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|------------|---------|--------------|-------------------|----------|-----------------|
| TX2        | 452 ms  | 120 ms       | 109 ms            |  87 ms   |  20 ms          |
| AGX XAVIER | 259 ms  | 39 ms        | 23 ms             |  39 ms   |  5 ms           |


### lenna.jpeg (361'200 pixels)

| Jetson     | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|------------|---------|--------------|-------------------|----------|-----------------|
| TX2        | 66 ms   | 17 ms        | 19 ms             |  13 ms   |  4 ms           |
| AGX XAVIER | 39 ms   | 6 ms         | 4 ms              |  6 ms    |  1 ms           |

### meadows.png (99'300 pixels)

| Jetson     | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|------------|---------|--------------|-------------------|----------|-----------------|
| TX2        | 18 ms   | 5 ms         | 8 ms              |  5 ms    |  2 ms           |
| AGX XAVIER | 11 ms   | 2 ms         | 2 ms              |  2 ms    |  1 ms           |

### mountain.jpg (114'072 pixels)

| Jetson     | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|------------|---------|--------------|-------------------|----------|-----------------|
| TX2        | 21 ms   | 6 ms         | 9 ms              |  5 ms    |  2 ms           |
| AGX XAVIER | 12 ms   | 2 ms         | 4 ms              |  3 ms    |  1 ms           |

### noir.png (111'000 pixels)

| Jetson     | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|------------|---------|--------------|-------------------|----------|-----------------|
| TX2        | 20 ms   | 5 ms         | 10 ms             |  5 ms    |  2 ms           |
| AGX XAVIER | 12 ms   | 2 ms         | 4 ms              |  2 ms    |  1 ms           |

### rosso.png (144'000 pixels)

| Jetson     | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|------------|---------|--------------|-------------------|----------|-----------------|
| TX2        | 27 ms   | 7 ms         | 12 ms             |  6 ms    |  2 ms           |
| AGX XAVIER | 15 ms   | 3 ms         | 5 ms              |  3 ms    |  1 ms           |

### rubik.png (230'337 pixels)

| Jetson     | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|------------|---------|--------------|-------------------|----------|-----------------|
| TX2        | 42 ms   | 11 ms        | 10 ms             |  10 ms   |  3 ms           |
| AGX XAVIER | 25 ms   | 4 ms         | 4 ms              |  4 ms    |  1 ms           |

### sky.png (132'300 pixels)

| Jetson     | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|------------|---------|--------------|-------------------|----------|-----------------|
| TX2        | 25 ms   | 7 ms         | 11 ms             |  6 ms    |  2 ms           |
| AGX XAVIER | 15 ms   | 3 ms         | 4 ms              |  4 ms    |  2 ms           |


## Notes

### Benchmark times

Beware that *sometimes* the first images processed, regardless of the image, has distorted timings due to, probably, some kind of hidden startup hiccup.  
To get valuable results (for both OpenCV and this implementation) it's advisable to run the benchmark multiple times while also changing the order in which the pictures get processed.

### OpenCV

Please note that OpenCV runs in parallel even when not using the GPU unless compiled with a special flag.  
This is the reason the CPU version of OpenCV is faster than the sequential implementation and about as fast as the OpenMP version.

### Naive CUDA version note

The Naive CUDA version performed quite bad and so it wasn't included in the benchmark. It was slower than all other parallel versions and about as fast as the Sequential version.