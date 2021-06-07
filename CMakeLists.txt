cmake_minimum_required(VERSION 3.12.2)
message(STATUS "CMake version: ${CMAKE_VERSION}")

# ------------------------------------------------------------------------------
# Setup project.

# Project with not optional CUDA.
project(bilateral_filter LANGUAGES CUDA CXX C)
set(PROJECT_NAME bilateral_filter)
set(DEPENDENCIES "")

# ------------------------------------------------------------------------------
# Setup dependencies.
include(cmake/GPUConfig.cmake)

# Check if CUDA is available.
include(CheckLanguage)
check_language(CUDA)

set(DEPENDENCIES ${DEPENDENCIES} -lstdc++fs)

# ------------------------------------------------------------------------------
# Setup CUDA.

# Check if a compiler exists.
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "Cuda compiler has been found: ${CMAKE_CUDA_COMPILER}")
else(CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "Cuda compiler has not been found")
endif()

# Check host compiler.
if(DEFINED ${CMAKE_CUDA_HOST_COMPILER})
    message(STATUS "Cuda host compiler defined: ${CMAKE_CUDA_HOST_COMPILER}")
endif()

# Check correct compiler id.
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    message(STATUS "Cuda compiler id match NVIDIA")
    message(STATUS "Compute Capability: ${CMAKE_CUDA_ARCHITECTURES}")
else(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    message(FATAL_ERROR "Cuda compiler id doesn't match")
endif()

message(STATUS 
        "CMake setup project succeeds with NVIDIA "
        "v${CMAKE_CUDA_COMPILER_VERSION}")

# ------------------------------------------------------------------------------
# Setup OpenCV.
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(DEPENDENCIES ${DEPENDENCIES} ${OpenCV_LIBS})
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV linking library dependecies: ${OpenCV_LIBS}")
message(STATUS "OpenCV compilation compute capability: ${OpenCV_COMPUTE_CAPABILITIES}")
message(STATUS "OpenCV compilation with CUDA version: ${OpenCV_CUDA_VERSION}")
message(STATUS "OpenCV_USE_CUBLAS: ${OpenCV_USE_CUBLAS}")
message(STATUS "OpenCV_USE_CUFFT: ${OpenCV_USE_CUFFT}")
message(STATUS "OpenCV_USE_NVCUVID: ${OpenCV_USE_NVCUVID}")
if(OpenCV_CUDA_VERSION)
    # Have CUDA support
    list(APPEND CUDA_NVCC_FLAGS ${OpenCV_COMPUTE_CAPABILITIES})
    add_definitions(-DOPENCV_CUDA)
else(OpenCV_CUDA_VERSION)
    message(WARNING "OpenCV: no CUDA support")
endif()

# ------------------------------------------------------------------------------
# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CUDA_FLAGS}")
endif()
# ------------------------------------------------------------------------------
# Set the C++ compiler options.
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-stack-protector")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic-errors")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-align")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-qual")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdisabled-optimization")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-equal")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-conversion")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=2")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat-nonliteral")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat-security")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat-y2k")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wimport")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winit-self")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winline")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winvalid-pch")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-long-long")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-field-initializers")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-format-attribute")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-include-dirs")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpacked")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpointer-arith")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wredundant-decls")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wstack-protector")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wstrict-aliasing=2")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wswitch")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wswitch-default")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wswitch-enum")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunreachable-code")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-function")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-label")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-parameter")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-value")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-variable")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wvariadic-macros")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wwrite-strings")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wsign-compare")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wsign-conversion")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmessage-length=0")

# ------------------------------------------------------------------------------
# Set the C compiler options.
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -nostdlib")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -nostdinc")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O1")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g3")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fomit-frame-pointer")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-builtin")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-stack-protector")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
# set(CMAKE_C_FLAGS"${CMAKE_C_FLAGS} -Wall")
# set(CMAKE_C_FLAGS"${CMAKE_C_FLAGS} -Wpedantic")
# set(CMAKE_C_FLAGS"${CMAKE_C_FLAGS} -pedantic-errors")
# set(CMAKE_C_FLAGS"${CMAKE_C_FLAGS} -Wextra")
#set(CMAKE_C_FLAGS"${CMAKE_C_FLAGS} -Werror")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wconversion")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wcast-align")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wcast-qual")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wdisabled-optimization")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wfloat-equal")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wfloat-conversion")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wformat=2")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wformat-nonliteral")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wformat-security")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wformat-y2k")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wimport")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Winit-self")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Winline")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Winvalid-pch")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-long-long")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmissing-field-initializers")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmissing-format-attribute")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmissing-include-dirs")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpacked")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpointer-arith")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wredundant-decls")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wshadow")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wstack-protector")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wstrict-aliasing=2")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wswitch")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wswitch-default")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wswitch-enum")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunreachable-code")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused-function")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused-label")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused-parameter")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused-value")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused-variable")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wvariadic-macros")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wwrite-strings")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wsign-compare")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wsign-conversion")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wuninitialized")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fmessage-length=0")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse4")

# ------------------------------------------------------------------------------
# Set the CUDA compiler options.
set(CC ${CMAKE_CUDA_ARCHITECTURES}) # Compute Capability
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${CC},code=sm_${CC}")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${CC},code=compute_${CC}")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O1")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14")

# ------------------------------------------------------------------------------
# Add the includes.
include_directories(
    inc
)

# ------------------------------------------------------------------------------
# Add the source files.
set(CXX_SOURCE_FILES
    src/main.cpp
    src/bilateral_filter/bf_sequential.cpp
    src/bilateral_filter/bf_sequential_cv.cpp
    src/bilateral_filter/bf_parallel_cv.cpp
    src/bilateral_filter/bf_parallel_omp.cpp
)

set(CUDA_SOURCE_FILES
    src/bilateral_filter/bf_parallel.cu
    src/bilateral_filter/bf_parallel_naive.cu
)

# ------------------------------------------------------------------------------
# Add the executable.
set(SOURCES
    ${CXX_SOURCE_FILES}
    ${CUDA_SOURCE_FILES}
)

add_executable(${PROJECT_NAME}
    ${SOURCES}
)
target_link_libraries(${PROJECT_NAME} ${DEPENDENCIES})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA_ARCHITECTURES: ${CUDA_ARCHITECTURES}")