cmake_minimum_required(VERSION 3.15...3.21)

project(
  CUDAPlayground
  VERSION 0.1
  LANGUAGES C CXX CUDA
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT SKBUILD)
  message(FATAL_ERROR "This project should be built by scikit-build")
endif()

# scikit-build
find_package(PythonExtensions REQUIRED)

# pytorch
find_package(Torch REQUIRED)
set(CMAKE_CXXFLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
find_library(TORCH_PYTHON_LIBRARY torch_python REQUIRED PATHS "${TORCH_INSTALL_PREFIX}/lib")

# magma
find_path(MAGMA_INCLUDE_DIRS magma.h REQUIRED PATHS "/usr/local/magma/include" "$ENV{CONDA_PREFIX}/include")
find_library(MAGMA_LIBRARY magma REQUIRED PATHS "/usr/local/magma/lib" "$ENV{CONDA_PREFIX}/lib")

# cuda
find_package(CUDAToolkit REQUIRED)

# # thrust
# include(FetchContent)
# FetchContent_Declare(
#   Thrust
#   GIT_REPOSITORY  https://github.com/NVIDIA/thrust.git
#   GIT_TAG         1.14.0
# )
# FetchContent_MakeAvailable(Thrust)

add_subdirectory(src)
