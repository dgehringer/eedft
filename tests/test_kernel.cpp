
#include "gtest/gtest.h"
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <random>
#include <tuple>
#include <variant>

extern "C" {
#include "eedft/kernel.h"
#include "eedft/stencil.h"
#include "eedft/wf_grid.h"
}

#define MAX_GRID_DIM FD_MAX_ORDER * 2

typedef std::tuple<int, int> Bounds;
typedef std::tuple<int, int, int> GridSize;
typedef std::tuple<SCALAR, SCALAR, SCALAR> Spacings;
typedef std::function<SCALAR(SCALAR, SCALAR, SCALAR)> GeneratingFunc;
typedef std::function<SCALAR(int, int, int)> GeneratingFuncIndex;
typedef std::variant<GeneratingFunc, GeneratingFuncIndex> AnyGeneratingFunc;
typedef std::function<void(SCALAR, SCALAR, SCALAR)> ActionFunc;
typedef std::function<void(int, int, int)> ActionFuncIndex;
typedef std::variant<ActionFunc, ActionFuncIndex> AnyActionFunc;
typedef std::function<void(SCALAR, SCALAR, SCALAR, SCALAR)> TestingFunc;
typedef std::function<void(int, int, int, SCALAR)> TestingFuncIndex;
typedef std::variant<TestingFunc, TestingFuncIndex> AnyTestingFunc;

GridSize random_shape(uint8_t p) {
  int min{p * 2 + 1}, max{MAX_GRID_DIM};
  std::random_device device;
  std::mt19937 rng(device());
  std::uniform_int_distribution<std::mt19937::result_type> rand(min, max);
  return std::make_tuple(rand(rng), rand(rng), rand(rng));
}

Spacings random_spacings() {
  std::default_random_engine rng;
  std::uniform_real_distribution<SCALAR> rand(0.0, 1.0);
  return std::make_tuple(rand(rng), rand(rng), rand(rng));
}

struct wf_grid grid_from_index_func(GeneratingFuncIndex func, uint8_t p,
                                    std::optional<GridSize> grid_size,
                                    std::optional<Spacings> spacings) {
  auto [hi, hj, hk] = spacings.value_or(random_spacings());
  auto [ni, nj, nk] = grid_size.value_or(random_shape(p));
  auto jstride = nk;
  auto istride = nj * nk;
  SCALAR *grid_array = (SCALAR *)std::malloc(sizeof(SCALAR) * ni * nj * nk);
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) {
      for (int k = 0; k < nk; k++) {
        grid_array[i * istride + j * jstride + k] = func(i, j, k);
      }
    }
  }
  // we allocate only one array and set all pointer rho12, si12, sj12 and sk12
  // to the same array
  struct wf_grid result {
    grid_array, grid_array, grid_array, grid_array, ni, nj, nk, hi, hj, hk
  };
  return result;
}

struct wf_grid grid_from_func(GeneratingFunc func, uint8_t p,
                              std::optional<GridSize> grid_size,
                              std::optional<Spacings> spacings) {
  SCALAR hi, hj, hk;
  std::tie(hi, hj, hk) = spacings.value_or(random_spacings());
  GeneratingFuncIndex index_func = [=](int i, int j, int k) {
    return func(i * hi, j * hj, k * hk);
  };
  return grid_from_index_func(index_func, p, grid_size,
                              std::make_optional(std::make_tuple(hi, hj, hk)));
}

std::optional<Bounds> make_bounds(int min, int max) {
  return std::make_optional(std::make_tuple(min, max));
}

void grid_forall_block(struct wf_grid *grid, AnyActionFunc func,
                       std::optional<Bounds> xbounds = std::nullopt,
                       std::optional<Bounds> ybounds = std::nullopt,
                       std::optional<Bounds> zbounds = std::nullopt) {
  // again we assume that grid->rho12 == grid->si12 == grid->sj12 == grid->sk12
  // for each element
  int ni{grid->ni}, nj{grid->nj}, nk{grid->nk};
  auto jstride = nk;
  auto istride = nj * nk;
  auto [imin, imax] = xbounds.value_or(std::make_tuple(0, ni));
  auto [jmin, jmax] = ybounds.value_or(std::make_tuple(0, nj));
  auto [kmin, kmax] = zbounds.value_or(std::make_tuple(0, nk));
  for (int i = imin; i < imax; i++) {
    SCALAR vi = i * grid->hi;
    for (int j = jmin; j < jmax; j++) {
      SCALAR vj = j * grid->hj;
      for (int k = kmin; k < kmax; k++) {
        SCALAR vk = k * grid->hk;
        if (std::holds_alternative<ActionFunc>(func)) {
          std::get<ActionFunc>(func)(vi, vj, vk);
        } else if (std::holds_alternative<ActionFuncIndex>(func)) {
          std::get<ActionFuncIndex>(func)(i, j, k);
        }
      }
    }
  }
}

void call_testing_func(struct wf_grid *grid, int i, int j, int k, SCALAR value,
                       std::optional<AnyTestingFunc> func) {
  if (func.has_value()) {
    auto testing_func = func.value();
    if (std::holds_alternative<TestingFuncIndex>(testing_func)) {
      std::get<TestingFuncIndex>(testing_func)(i, j, k, value);
    } else if (std::holds_alternative<TestingFunc>(testing_func)) {
      std::get<TestingFunc>(testing_func)(i * grid->hi, j * grid->hj,
                                          k * grid->hk, value);
    }
  }
}

void test_laplacian_and_gradient(
    AnyGeneratingFunc generator,
    std::optional<AnyTestingFunc> test_lap = std::nullopt,
    std::optional<AnyTestingFunc> test_gx = std::nullopt,
    std::optional<AnyTestingFunc> test_gy = std::nullopt,
    std::optional<AnyTestingFunc> test_gz = std::nullopt,
    bool handle_boundary = false) {
  for (uint8_t order = FD_MIN_ORDER; order < FD_MAX_ORDER; order += 2) {
    uint8_t p = order / 2;
    // initialize wf_grid from a given function which either takes the index
    // values of the x, y and z coordinates
    struct wf_grid grid;
    if (std::holds_alternative<GeneratingFuncIndex>(generator)) {
      grid = grid_from_index_func(std::get<GeneratingFuncIndex>(generator), p,
                                  std::nullopt, std::nullopt);
    } else if (std::holds_alternative<GeneratingFunc>(generator)) {
      grid = grid_from_func(std::get<GeneratingFunc>(generator), p,
                            std::nullopt, std::nullopt);
    }

    // initialize the corresponding stencils
    struct fd_stencil lap, grad;
    make_laplacian(order, &grid, &lap);
    make_gradient(order, &grid, &grad);
    ASSERT_EQ(lap.p, p);
    ASSERT_EQ(grad.p, p);
    // create output array for laplacians and gradients

    int ni{grid.ni}, nj{grid.nj}, nk{grid.nk};
    auto jstride = nk;
    auto istride = nj * nk;
    // create a function that wraps laplacian_and_gradient function from
    // kernel.h
    bool consider_boundary = false;
    ActionFuncIndex apply_and_test = [&grid, &lap, &grad, istride, jstride,
                                      &test_lap, &test_gx, &test_gy, &test_gz,
                                      &consider_boundary](int i, int j, int k) {
      int index = i * istride + j * jstride + k;
      // grid_from_func only initializes one array it does not matter which one
      // we use for wf_grid out all fields( rho12, si12, sj12 and sk12) are
      // separate blocks of mem because we want to hold 4 values, whereas
      // laplacian = rho12, gradient_x = si12, gradient_y = sj12 and finally
      // gradient_z = sk12
      SCALAR is_lap, is_gx, is_gy, is_gz;
      laplacian_and_gradient(grid.rho12, &grid, i, j, k, consider_boundary,
                             &lap, &grad, &is_lap, &is_gx, &is_gy, &is_gz);
      call_testing_func(&grid, i, j, k, is_lap, test_lap);
      call_testing_func(&grid, i, j, k, is_gx, test_gx);
      call_testing_func(&grid, i, j, k, is_gy, test_gy);
      call_testing_func(&grid, i, j, k, is_gz, test_gz);
    };
    // apply laplacian and gradient
    grid_forall_block(&grid, apply_and_test, make_bounds(p, ni - p),
                      make_bounds(p, nj - p), make_bounds(p, nk - p));
    if (handle_boundary) {
      consider_boundary = handle_boundary;
      grid_forall_block(&grid, apply_and_test, make_bounds(0, p),
                        make_bounds(p, nj - p), make_bounds(p, nk - p));
      grid_forall_block(&grid, apply_and_test, make_bounds(ni-p, nj),
                        make_bounds(p, nj - p), make_bounds(p, nk - p));
    }
    std::free(grid.rho12);
  }
}

TestingFuncIndex assert_near(SCALAR should, SCALAR abs_tol = 1.0e-3) {
  return [should, abs_tol](int x, int y, int z, SCALAR actual) {
    ASSERT_NEAR(actual, should, abs_tol);
  };
}

TEST(kernel, laplacian_and_gradient_linear) {
  auto is_zero = std::make_optional(assert_near(0.0));
  auto is_one = std::make_optional(assert_near(1.0));
  GeneratingFunc linear_x = [](SCALAR x, SCALAR, SCALAR) -> SCALAR {
    return x;
  };
  GeneratingFunc linear_y = [](SCALAR, SCALAR y, SCALAR) -> SCALAR {
    return y;
  };
  GeneratingFunc linear_z = [](SCALAR, SCALAR, SCALAR z) -> SCALAR {
    return z;
  };

  test_laplacian_and_gradient(linear_x, is_zero, is_one, is_zero, is_zero);
  test_laplacian_and_gradient(linear_y, is_zero, is_zero, is_one, is_zero);
  test_laplacian_and_gradient(linear_z, is_zero, is_zero, is_zero, is_one);
}

TEST(kernel, laplacian_and_gradient_quadratic) {
  GeneratingFunc linear_xyz = [](SCALAR x, SCALAR y, SCALAR z) {
    return x * x + x;
  };
  auto abs_tol = 1e-3;
  auto is_two = std::make_optional(assert_near(2.0));
  auto is_zero = std::make_optional(assert_near(0.0));
  std::optional<TestingFunc> gx =
      std::make_optional([abs_tol](SCALAR x, SCALAR, SCALAR, SCALAR actual) {
        ASSERT_NEAR(2.0 * x + 1.0, actual, abs_tol);
      });
  test_laplacian_and_gradient(linear_xyz, is_two, gx, is_zero, is_zero);
}

TEST(kernel, laplacian_and_gradient_triquadratic) {
  GeneratingFunc linear_xyz = [](SCALAR x, SCALAR y, SCALAR z) {
    return x * x + y * y + z * z + x * y * z;
  };
  auto abs_tol = 1e-3;
  auto is_six = std::make_optional(assert_near(6.0));
  std::optional<TestingFunc> gx = std::make_optional(
      [abs_tol](SCALAR x, SCALAR y, SCALAR z, SCALAR actual) {
        ASSERT_NEAR(2.0 * x + y * z, actual, abs_tol);
      });
  std::optional<TestingFunc> gy = std::make_optional(
      [abs_tol](SCALAR x, SCALAR y, SCALAR z, SCALAR actual) {
        ASSERT_NEAR(2.0 * y + x * z, actual, abs_tol);
      });
  std::optional<TestingFunc> gz = std::make_optional(
      [abs_tol](SCALAR x, SCALAR y, SCALAR z, SCALAR actual) {
        ASSERT_NEAR(2.0 * z + x * y, actual, abs_tol);
      });
  test_laplacian_and_gradient(linear_xyz, is_six, gx, gy, gz);
}

TEST(kernel, laplacian_and_gradient_handle_boundary) {
  GeneratingFunc linear_xyz = [](SCALAR x, SCALAR y, SCALAR z) {
    return x * x + y * y + z * z + x * y * z;
  };
  auto abs_tol = 1e-3;
  auto is_six = std::make_optional(assert_near(6.0));
  // std::optional<TestingFunc> gx = std::make_optional(
  //     [abs_tol](SCALAR x, SCALAR y, SCALAR z, SCALAR actual) {
  //       ASSERT_NEAR(2.0 * x + y * z, actual, abs_tol);
  //     });
  test_laplacian_and_gradient(linear_xyz, std::nullopt, std::nullopt,
                              std::nullopt, std::nullopt, true);
}
