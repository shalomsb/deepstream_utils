#!/bin/bash

# Crash on errors
set -x

# Helper variables
TRITON_CHECKOUT_TAG=r24.11

# Install root directory
cd /opt

# Clone Triton Python Backend git
git clone https://github.com/triton-inference-server/python_backend.git triton_python_backend_gpu
cd triton_python_backend_gpu
git checkout $TRITON_CHECKOUT_TAG

# Build directory
mkdir build
cd build

# Configure, including the checkout tag to match
cmake \
    -DTRITON_ENABLE_GPU=ON \
    -DTRITON_BACKEND_REPO_TAG=${TRITON_CHECKOUT_TAG} \
    -DTRITON_COMMON_REPO_TAG=${TRITON_CHECKOUT_TAG} \
    -DTRITON_CORE_REPO_TAG=${TRITON_CHECKOUT_TAG} \
    -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..

# Build & Install the python backend
make -j$(nproc) install

# Backup original python backend build
cp -r /opt/tritonserver/backends/python /opt/tritonserver/backends/python_backend_backup

# Override the python backend binaries
cp libtriton_python.so /opt/tritonserver/backends/python/libtriton_python.so
cp triton_python_backend_stub /opt/tritonserver/backends/python/triton_python_backend_stub
