#!/bin/sh
cd $(dirname "$0")
git submodule update --init --recursive
mkdir -p build && cd build
PROTOBUF_INCL=$(cmake -DONNX_ML=OFF .. | grep 'Protobuf includes'| tr -s ' ' | cut -d' ' -f5)
PROTOBUF_LIBS=$(cmake -DONNX_ML=OFF .. | grep 'Protobuf libraries'| tr -s ' ' | cut -d' ' -f5)
ln -s $PROTOBUF_INCL protobuf
ln -s $PROTOBUF_LIBS .
make -j8
