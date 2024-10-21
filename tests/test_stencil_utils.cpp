
#include "gtest/gtest.h"
#include <stdint.h>

namespace eedft::core::test {
  TEST(stencil, accept_correct_orders) {
    // we employ central finite differences. There we only allow even order
    // 2, 4, 6, 8, 10 .. etc.
    struct fd_stencil stencil;
    struct wf_grid grid = {.hi = 1.0, .hj = 1.0, .hk = 1.0};

    for (uint8_t order = 0; order < (FD_MAX_ORDER * 2); order++) {
      auto result = make_gradient(order, &grid, &stencil);
      if (order < FD_MIN_ORDER || order > FD_MAX_ORDER || order % 2 == 1)
        ASSERT_EQ(result, nullptr);
      result = make_laplacian(order, &grid, &stencil);
      if (order < FD_MIN_ORDER || order > FD_MAX_ORDER || order % 2 == 1)
        ASSERT_EQ(result, nullptr);
    }
  }

}