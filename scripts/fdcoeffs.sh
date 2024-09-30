#!/bin/zsh

python3 codegen/fdcoeffs.py -o include/eedft/core/stencil/data.hpp --min-order=2 --max-order=14 --namespace "eedft::core::stencil"