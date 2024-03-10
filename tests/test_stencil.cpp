
#include "gtest/gtest.h"
#include <stdint.h>

extern "C" {
#include "eedft/stencil.h"
}

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

TEST(stencil, correct_p) {
  struct fd_stencil stencil;
  struct wf_grid grid = {.hi = 1.0, .hj = 1.0, .hk = 1.0};

  for (uint8_t order = FD_MIN_ORDER; order < (FD_MAX_ORDER + 1); order += 2) {
    auto result = make_gradient(order, &grid, &stencil);
    ASSERT_EQ(result, &stencil);
    ASSERT_EQ(stencil.p, order / 2);
    result = make_laplacian(order, &grid, &stencil);
    ASSERT_EQ(result, &stencil);
    ASSERT_EQ(stencil.p, order / 2);
  }
}

TEST(stencil, correct_coeffs) {
  struct fd_stencil grad, lap;
  struct wf_grid grid = {.hi = 1.0, .hj = 1.0, .hk = 1.0};

  for (uint8_t order = FD_MIN_ORDER; order < (FD_MAX_ORDER + 1); order += 2) {
    make_gradient(order, &grid, &grad);
    make_laplacian(order, &grid, &lap);
    auto index = order / 2 - 1;
    auto exp_grad{_gradients[index]}, exp_lap{_laplacians[index]};
    ASSERT_EQ(lap.p, exp_lap.p);
    ASSERT_EQ(grad.p, exp_grad.p);
    auto ncoeffs = (2 * lap.p) + 1;
    for (int i = 0; i < ncoeffs; i++) {
      for (int ax = 0; ax < 3; ax++) {
        ASSERT_DOUBLE_EQ(lap.coeffs[ax][i], exp_lap.coeffs[i]);
        ASSERT_DOUBLE_EQ(grad.coeffs[ax][i], exp_grad.coeffs[i]);
      }
    }
    for (int i = ncoeffs; i < FD_COEFFS_MAX_COEFF; i++) {
      for (int ax = 0; ax < 3; ax++) {
        ASSERT_DOUBLE_EQ(lap.coeffs[ax][i], 0.0);
        ASSERT_DOUBLE_EQ(grad.coeffs[ax][i], 0.0);
      }
    }
  }
}
