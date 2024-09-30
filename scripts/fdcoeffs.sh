#!/bin/zsh

python3 codegen/fdcoeffs.py -o include/eedft/stencil/data.hpp --min-order=2 --max-order=14 --namespace "eedft::stencil"