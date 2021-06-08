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

## Benchmark Results

This filter has been tried on 3 different devices.

| ID | CPU          | GPU             |
|----|--------------|-----------------|
| A  | AMD FX-6350  | NVIDIA 1060 GTX |
| B  | AMD R5-5600X | NVIDIA 3090 RTX |
| C  | ????         | ?????           |

On every computer this implementation of the Bilateral Filter turned out to be comparable with OpenCV's (when both ran in parallel mode).

### cones.png (2'435'860 pixels)

| ID | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|----|---------|--------------|-------------------|----------|-----------------|
| A  | 352 ms  | 84 ms        | 91 ms             |  23 ms   |  26 ms          |
| B  | 121 ms  | 15 ms        | 13 ms             |  8 ms    |  7 ms           |
| C  | ??????? | ????         | ?????             |          |                 |

### lenna.jpeg (361'200 pixels)

| ID | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|----|---------|--------------|-------------------|----------|-----------------|
| A  | 50 ms   | 14 ms        | 17 ms             |  5 ms    |  5 ms           |
| B  | 18 ms   | 9 ms         | 7 ms              |  1 ms    |  1 ms           |
| C  | ??????? | ????         | ?????             |          |                 |

### meadows.png (99'300 pixels)

| ID | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|----|---------|--------------|-------------------|----------|-----------------|
| A  | 16 ms   | 6 ms         | 7 ms              |  3 ms    |  3 ms           |
| B  | 5 ms    | 2 ms         | 3 ms              |  1 ms    |  1 ms           |
| C  | ??????? | ????         | ?????             |          |                 |

### mountain.jpg (114'072 pixels)

| ID | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|----|---------|--------------|-------------------|----------|-----------------|
| A  | 21 ms   | 4 ms         | 5 ms              |  3 ms    |  3 ms           |
| B  | 6 ms    | 2 ms         | 4 ms              |  1 ms    |  1 ms           |
| C  | ??????? | ????         | ?????             |          |                 |

### noir.png (111'000 pixels)

| ID | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|----|---------|--------------|-------------------|----------|-----------------|
| A  | 19 ms   | 4 ms         | 4 ms              |  3 ms    |  4 ms           |
| B  | 5 ms    | 2 ms         | 1 ms              |  1 ms    |  1 ms           |
| C  | ??????? | ????         | ?????             |          |                 |

### rosso.png (144'000 pixels)

| ID | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|----|---------|--------------|-------------------|----------|-----------------|
| A  | 25 ms   | 6 ms         | 7 ms              |  3 ms    |  3 ms           |
| B  | 7 ms    | 2 ms         | 3 ms              |  1 ms    |  2 ms           |
| C  | ??????? | ????         | ?????             |          |                 |

### rubik.png (230'337 pixels)

| ID | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|----|---------|--------------|-------------------|----------|-----------------|
| A  | 39 ms   | 10 ms        | 8 ms              |  4 ms    |  3 ms           |
| B  | 13 ms   | 2 ms         | 5 ms              |  2 ms    |  2 ms           |
| C  | ??????? | ????         | ?????             |          |                 |

### sky.png (132'300 pixels)

| ID | CPU Seq | CPU OMP      | CPU OpenCV        |  CUDA    | CUDA OpenCV     |
|----|---------|--------------|-------------------|----------|-----------------|
| A  | 23 ms   | 5 ms         | 7 ms              |  3 ms    |  3 ms           |
| B  | 7 ms    | 2 ms         | 3 ms              |  2 ms    |  1 ms           |
| C  | ??????? | ????         | ?????             |          |                 |

As expected, the parallel versions performed better than the sequential version.  
And the GPU versions performed better than the CPU (parallel) ones. The speedup gets more noticeable with bigger images.

## Naive CUDA version

The Naive CUDA version performs quite bad. It's slower than CPU OMP and about as fast as the Sequential version.


## Notes

In the benchmark the first images processed, regardless of the image, has distorted timings due to, probably, some kind of hidden CUDA startup.  
To get valuable results (for both OpenCV and this implementation) it's advisable to run the benchmark multiple times changing the order of the pictures.
