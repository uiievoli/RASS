cmake_minimum_required(VERSION 3.21)
project(TriBase)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Standard)
elseif (NOT CMAKE_BUILD_TYPE MATCHES "Debug|Release|Standard")
    message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(OpenMP REQUIRED)
find_package(GTest REQUIRED)

# Conda installs Eigen's CMake package and headers below $CONDA_PREFIX. CMake
# does not consistently add the active environment to CMAKE_PREFIX_PATH, so
# make it an explicit search prefix before looking for Eigen3::Eigen.
if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    find_path(EIGEN3_INCLUDE_DIR
        NAMES Eigen/Dense
        HINTS
            "$ENV{CONDA_PREFIX}/include/eigen3"
            "$ENV{CONDA_PREFIX}/include"
        PATH_SUFFIXES eigen3)
    if(NOT EIGEN3_INCLUDE_DIR)
        message(FATAL_ERROR
            "Eigen3 headers were not found. Install Eigen in the active environment "
            "(conda install -c conda-forge eigen) or pass "
            "-DEIGEN3_INCLUDE_DIR=/path/to/include/eigen3")
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

set(MKL_ROOT "/opt/intel/oneapi/mkl/2024.2")
include_directories(${MKL_ROOT}/include)
link_directories(${MKL_ROOT}/lib/intel64)

option(ENABLE_PROFILING "Enable profiling" OFF)
option(ENABLE_STATS "Enable detailed pruning statistics" OFF)
option(ENABLE_LTO "Enable interprocedural/link-time optimization" OFF)

if (ENABLE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

# set(W_CXX_FLAGS " -Wall -Wextra")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "Standard")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Ofast -march=native -mtune=native -funroll-loops -fomit-frame-pointer -fno-semantic-interposition -falign-functions=64 -DNDEBUG")
endif()

if(ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT TRIBASE_IPO_SUPPORTED OUTPUT TRIBASE_IPO_ERROR)
    if(NOT TRIBASE_IPO_SUPPORTED)
        message(FATAL_ERROR "ENABLE_LTO=ON, but IPO/LTO is unavailable: ${TRIBASE_IPO_ERROR}")
    endif()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

if(ENABLE_STATS)
    add_compile_definitions(TRIBASE_ENABLE_STATS)
endif()

include_directories(${CMAKE_SOURCE_DIR}/third_party)
include_directories(${CMAKE_SOURCE_DIR}/third_party/faiss)
include_directories(${CMAKE_SOURCE_DIR}/hnswlib)
add_subdirectory(third_party/faiss)
add_subdirectory(src)
