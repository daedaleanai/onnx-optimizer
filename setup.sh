#!/bin/sh
cd $(dirname "$0")
git submodule update --init --recursive
mkdir -p build && cd build
cmake -DONNX_ML=OFF ..
make -j8
