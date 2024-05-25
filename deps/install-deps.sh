#!/bin/sh

## Modify the following lines to install dependencies for your project.
echo "Installing dependencies for VINS-Mono..."
apt update -q
apt install -y --no-install-recommends git libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libsuitesparse-dev

main_dir=$(pwd)
mkdir -p $main_dir/deps
git clone --depth 1 --branch 2.2.0 https://ceres-solver.googlesource.com/ceres-solver ./deps/ceres-solver
cd $main_dir/deps/ceres-solver

# Apply patch to fix glog issue
git apply $main_dir/FindGlog.cmake.patch
echo "Applied patch to fix glog issue"

# Proceed to build and install Ceres
mkdir build && cd build
echo "Building and installing Ceres..."
cmake .. && make -j10 && make install
cd ../../..
rm -rf $main_dir/deps