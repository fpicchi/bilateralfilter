# Bilateral Filter

## Dependecies

- Cuda
- OpenCV

## Settings

- Set CMAKE_CUDA_ARCHITECTURES in cmake/GPUConfig.cmake according to your device compute capability.
- Set CHECK_PARALLEL to 1 in src/main.cpp to check if your parallel solution is correct when it's completed.
- Set SPEEDUP_PARALLEL to 1 in src/main.cpp to print the speedup of your parallel solution in relation with the others.

## Run

```
mkdir build && cd build
cmake ..
make -j8
./bilateral_filter
```