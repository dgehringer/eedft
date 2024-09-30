//
// Created by Dominik Gehringer on 26.09.24.
//

#include <iostream>
#include <ostream>

#include "eedft/stencil/data.hpp"

int main() {
  auto val = eedft::stencil::fd<double, 1, 4>::coeffs;
  for (auto i : val) {
    std::cout << i << std::endl;
  }
}
