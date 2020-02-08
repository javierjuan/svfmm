cmake_minimum_required(VERSION 3.10)

# set the project name
project(svfmm VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# specify build type
set(CMAKE_BUILD_TYPE Release)

# Eigen support
find_package(Eigen3 3.3 REQUIRED)

# Boost support
# set(Boost_DEBUG ON)
find_package(Boost 1.65 REQUIRED)

# Python support
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

# OpenMP support
find_package(OpenMP REQUIRED)

# add the library
add_library(svfmm SHARED svfmm.cpp)
add_library(tstsvfmm SHARED tstsvfmm.cpp)
add_library(nlsvfmm SHARED nlsvfmm.cpp)

# set private library path
set(LIBRARY_DIR /home/javier/Library)

# Pybind11 support
set(PYBIND11_INCLUDE_DIR /home/javier/.local/include/python3.6m)

# set -fPIC
set_property(TARGET svfmm tstsvfmm nlsvfmm PROPERTY POSITION_INDEPENDENT_CODE ON)


# change default prefix for libs
set_property(TARGET svfmm tstsvfmm nlsvfmm PROPERTY PREFIX "")


# include library paths
#target_include_directories(svfmm PRIVATE
#                           "${LIBRARY_DIR}/cimg"
#                           "${LIBRARY_DIR}/classifiers"
#                           "${LIBRARY_DIR}/svfmm"
#                           "${LIBRARY_DIR}/tools"
#                           "${PYTHON_INCLUDE_DIR}"
#                           "${PYBIND11_INCLUDE_DIR}"
#                           )
include_directories(${LIBRARY_DIR}/cimg
                    ${LIBRARY_DIR}/classifiers
                    ${LIBRARY_DIR}/svfmm
                    ${LIBRARY_DIR}/tools
                    ${PYTHON_INCLUDE_DIR}
                    ${PYBIND11_INCLUDE_DIR}
                    )
# compile options
#target_compile_options(svfmm PRIVATE -Wall -Wextra -Wno-comment -Wno-unused-variable -Wno-unused-parameter)
add_compile_options(-Wall -Wextra -Wno-comment -Wno-unused-variable -Wno-unused-parameter)

# link libraries
target_link_libraries(svfmm Eigen3::Eigen)
target_link_libraries(svfmm Boost::boost)
target_link_libraries(svfmm OpenMP::OpenMP_CXX)

target_link_libraries(tstsvfmm Eigen3::Eigen)
target_link_libraries(tstsvfmm Boost::boost)
target_link_libraries(tstsvfmm OpenMP::OpenMP_CXX)

target_link_libraries(nlsvfmm Eigen3::Eigen)
target_link_libraries(nlsvfmm Boost::boost)
target_link_libraries(nlsvfmm OpenMP::OpenMP_CXX)
