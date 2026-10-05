cmake_minimum_required(VERSION 3.21)

# Prefer compilers from the active Conda environment.  Compiler selection must
# happen before project(); changing CC/CXX after a build directory has already
# been configured cannot replace the compiler cached by CMake.
set(TRIBASE_CONDA_PREFIX "$ENV{CONDA_PREFIX}" CACHE PATH
    "Active Conda prefix used to discover compilers and dependencies")
if(TRIBASE_CONDA_PREFIX)
    if(NOT CMAKE_C_COMPILER)
        find_program(TRIBASE_CONDA_C_COMPILER
            NAMES x86_64-conda-linux-gnu-gcc gcc
            PATHS "${TRIBASE_CONDA_PREFIX}/bin"
            NO_DEFAULT_PATH)
        if(TRIBASE_CONDA_C_COMPILER)
            set(CMAKE_C_COMPILER "${TRIBASE_CONDA_C_COMPILER}" CACHE FILEPATH
                "C compiler" FORCE)
        endif()
    endif()
    if(NOT CMAKE_CXX_COMPILER)
        find_program(TRIBASE_CONDA_CXX_COMPILER
            NAMES x86_64-conda-linux-gnu-g++ g++ c++
            PATHS "${TRIBASE_CONDA_PREFIX}/bin"
            NO_DEFAULT_PATH)
        if(TRIBASE_CONDA_CXX_COMPILER)
            set(CMAKE_CXX_COMPILER "${TRIBASE_CONDA_CXX_COMPILER}" CACHE FILEPATH
                "C++ compiler" FORCE)
        endif()
    endif()
endif()

project(TriBase LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Standard)
elseif (NOT CMAKE_BUILD_TYPE MATCHES "Debug|Release|Standard")
    message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Conda is not automatically part of CMake's package search path on every
# installation.  Add it explicitly without turning it into a sysroot (system
# headers and libraries remain available).
if(TRIBASE_CONDA_PREFIX)
    list(PREPEND CMAKE_PREFIX_PATH "${TRIBASE_CONDA_PREFIX}")
endif()

message(STATUS "TriBase C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "TriBase C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "TriBase C++ compiler version: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "TriBase Conda prefix: ${TRIBASE_CONDA_PREFIX}")

# std::format is supplied by the selected compiler's C++ standard library; it
# is not the external {fmt} package.  Fail during configure with a useful
# diagnostic instead of producing a later, ambiguous header error.
include(CheckCXXSourceCompiles)
check_cxx_source_compiles([=[
    #include <format>
    #include <string>
    int main() {
        const std::string value = std::format("{} {:.2f}", 7, 1.25);
        return value.empty();
    }
]=] TRIBASE_HAS_STD_FORMAT)
if(NOT TRIBASE_HAS_STD_FORMAT)
    message(FATAL_ERROR
        "The selected C++ compiler cannot compile C++20 std::format.\n"
        "  compiler: ${CMAKE_CXX_COMPILER}\n"
        "  compiler version: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\n"
        "  Conda prefix: ${TRIBASE_CONDA_PREFIX}\n"
        "Install a matching compiler and C++ development headers, for example:\n"
        "  conda install -c conda-forge 'gxx_linux-64=13.*' "
        "'gxx_impl_linux-64=13.*' 'libstdcxx-devel_linux-64=13.*'\n"
        "Then configure a NEW build directory or pass "
        "-DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++.")
endif()

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GTest REQUIRED)

find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    find_path(EIGEN3_INCLUDE_DIR
        NAMES Eigen/Dense
        HINTS
            "${TRIBASE_CONDA_PREFIX}/include/eigen3"
            "${TRIBASE_CONDA_PREFIX}/include"
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

# Locate MKL in this order: an explicit -DMKL_ROOT, MKLROOT/MKL_ROOT from the
# environment, the active Conda environment, then installed oneAPI versions.
set(MKL_ROOT "" CACHE PATH "Root of an Intel MKL or Conda MKL installation")
set(_TRIBASE_MKL_HINTS)
if(MKL_ROOT)
    list(APPEND _TRIBASE_MKL_HINTS "${MKL_ROOT}")
endif()
if(DEFINED ENV{MKLROOT} AND NOT "$ENV{MKLROOT}" STREQUAL "")
    list(APPEND _TRIBASE_MKL_HINTS "$ENV{MKLROOT}")
endif()
if(DEFINED ENV{MKL_ROOT} AND NOT "$ENV{MKL_ROOT}" STREQUAL "")
    list(APPEND _TRIBASE_MKL_HINTS "$ENV{MKL_ROOT}")
endif()
if(TRIBASE_CONDA_PREFIX)
    list(APPEND _TRIBASE_MKL_HINTS "${TRIBASE_CONDA_PREFIX}")
endif()
file(GLOB _TRIBASE_ONEAPI_MKL_ROOTS LIST_DIRECTORIES TRUE
    "/opt/intel/oneapi/mkl/*")
list(SORT _TRIBASE_ONEAPI_MKL_ROOTS COMPARE NATURAL ORDER DESCENDING)
list(APPEND _TRIBASE_MKL_HINTS ${_TRIBASE_ONEAPI_MKL_ROOTS})

find_path(MKL_INCLUDE_DIR
    NAMES mkl.h
    HINTS ${_TRIBASE_MKL_HINTS}
    PATH_SUFFIXES include)
find_library(MKL_INTEL_LP64_LIBRARY
    NAMES mkl_intel_lp64
    HINTS ${_TRIBASE_MKL_HINTS}
    PATH_SUFFIXES lib lib/intel64 lib/intel64_lin)
find_library(MKL_SEQUENTIAL_LIBRARY
    NAMES mkl_sequential
    HINTS ${_TRIBASE_MKL_HINTS}
    PATH_SUFFIXES lib lib/intel64 lib/intel64_lin)
find_library(MKL_CORE_LIBRARY
    NAMES mkl_core
    HINTS ${_TRIBASE_MKL_HINTS}
    PATH_SUFFIXES lib lib/intel64 lib/intel64_lin)

if(NOT MKL_INCLUDE_DIR OR
   NOT MKL_INTEL_LP64_LIBRARY OR
   NOT MKL_SEQUENTIAL_LIBRARY OR
   NOT MKL_CORE_LIBRARY)
    string(JOIN "\n    " _TRIBASE_MKL_HINTS_TEXT ${_TRIBASE_MKL_HINTS})
    message(FATAL_ERROR
        "Intel MKL was not found.\n"
        "Searched roots:\n    ${_TRIBASE_MKL_HINTS_TEXT}\n"
        "Required files: mkl.h, libmkl_intel_lp64, libmkl_sequential, libmkl_core.\n"
        "Activate the Conda environment before configuring, or pass:\n"
        "  -DTRIBASE_CONDA_PREFIX=$CONDA_PREFIX -DMKL_ROOT=$CONDA_PREFIX")
endif()

add_library(MKL::MKL INTERFACE IMPORTED)
target_include_directories(MKL::MKL INTERFACE "${MKL_INCLUDE_DIR}")
target_link_libraries(MKL::MKL INTERFACE
    "${MKL_INTEL_LP64_LIBRARY}"
    "${MKL_SEQUENTIAL_LIBRARY}"
    "${MKL_CORE_LIBRARY}"
    Threads::Threads
    ${CMAKE_DL_LIBS}
    m)

get_filename_component(MKL_LIBRARY_DIR "${MKL_CORE_LIBRARY}" DIRECTORY)
list(APPEND CMAKE_BUILD_RPATH "${MKL_LIBRARY_DIR}")
list(APPEND CMAKE_INSTALL_RPATH "${MKL_LIBRARY_DIR}")
message(STATUS "TriBase Eigen: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "TriBase MKL include: ${MKL_INCLUDE_DIR}")
message(STATUS "TriBase MKL libraries: ${MKL_INTEL_LP64_LIBRARY};${MKL_SEQUENTIAL_LIBRARY};${MKL_CORE_LIBRARY}")

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
