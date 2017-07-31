#! /usr/bin/env bash

rm -rf build
mkdir build
cd build
cmake ../src
make -i -j 8

