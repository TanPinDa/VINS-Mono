#!/bin/sh

## Modify the following lines to install dependencies for your project.
echo "Installing dependencies for non-ROS packages"
sudo apt update -q
sudo apt install -y --no-install-recommends git libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libsuitesparse-dev


# FOR RERUN
sudo apt-get -y install \
    libclang-dev \
    libatk-bridge2.0 \
    libfontconfig1-dev \
    libfreetype6-dev \
    libglib2.0-dev \
    libgtk-3-dev \
    libssl-dev \
    libxcb-render0-dev \
    libxcb-shape0-dev \
    libxcb-xfixes0-dev \
    libxkbcommon-dev \
    patchelf

sudo add-apt-repository ppa:kisak/kisak-mesa
sudo apt-get update
sudo apt-get install -y mesa-vulkan-drivers
sudo pip3 install rerun-sdk
ENV XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR
# END FOR RERUN

mkdir ./deps
cd ./deps

git clone --depth 1 --branch 2.2.0 https://ceres-solver.googlesource.com/ceres-solver ./ceres-solver
cd ./ceres-solver
mkdir build && cd build
cmake .. && make -j10 && sudo make install
cd ../..

git clone --depth 1 --branch v1.x https://github.com/gabime/spdlog.git ./spdlog
cd ./spdlog && mkdir build && cd build
cmake .. && make -j10 && sudo make install
cd ../..

cd ../
rm -rf ./deps