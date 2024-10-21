#!/bin/zsh

python3 codegen/fdcoeffs.py -o include/eedft/core/stencil/data.h --min-order=2 --max-order=14 --namespace "eedft::core::fd"