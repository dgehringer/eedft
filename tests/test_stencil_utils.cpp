
#include <stdint.h>

#include "eedft/core/fd/helpers.h"
#include "gtest/gtest.h"

namespace eedft::core::test {

  template <typename T, std::size_t N>
  constexpr std::array<T, N> make_const_array(T value) {
    std::array<T, N> arr;
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      ((arr[I] = value), ...);
    }(std::make_index_sequence<N>{});
    return arr;
  }

  template <typename T, std::size_t N> constexpr std::array<T, N*N*N> make_grid_data() {
    std::array<T, N * N * N> arr;
    grid_t<T> data(arr.data(), N, N, N);
    constexpr_for<0, N, 1>([&](auto i) {
      constexpr_for<0, N, 1>([&](auto j) {
        constexpr_for<0, N, 1>([&](auto k) {
          data[i, j, k] = i * (N * N) + (N * j) + k;
        });
      });
    });
    return arr;
  }

  template <std::size_t N>
  struct Size {
    static constexpr std::size_t value = N;
  };


  template <class Size>
  class TestStencilSumFixture : public ::testing::Test {
  public:
    static constexpr  std::size_t N = Size::value;
    static constexpr std::size_t length = fd::stencil_size<std::size_t>(1, N);
    static constexpr auto values_ = make_const_array<double, length>(1);
    static constexpr auto grid_data_ = make_grid_data<double, length>();
    static constexpr auto grid_ = grid_t<double>(grid_data_.size(), N, N, N);
  };


  using TestStencilSumFixtureTypes = ::testing::Types<Size<2>, Size<4>>;
  TYPED_TEST_SUITE(TestStencilSumFixture, TestStencilSumFixtureTypes);

  TYPED_TEST(TestStencilSumFixture, sum_axis_one) {

  }

}  // namespace eedft::core::test