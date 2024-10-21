//
// Created by Dominik Gehringer on 30.09.24.
//

#ifndef EEDFT_CORE_FD_HELPERS_H
#define EEDFT_CORE_FD_HELPERS_H
#include <mdspan>

#include "eedft/core/helpers.h"
#include "eedft/core/types.h"

namespace eedft::core::fd {
  template <class Out>
    requires std::is_integral_v<Out>
  constexpr Out compute_p(Out m, Out n) {
    return (m + 1) / 2 - 1 + n / 2;
  }

  template <class Out>
    requires std::is_integral_v<Out>
  constexpr Out stencil_size(Out m, Out n) {
    return 2 * compute_p(m, n) + 1;
  }

  template <class Out>
    requires std::is_integral_v<Out>
  constexpr Out p_from_size(Out size) {
    // static_assert(size % 2 == 0, "Size must be odd");
    return (size - 1) / 2;
  }

  template <Axis Axis, auto Offset, Boundary Boundary, class T, class Layout = std::layout_right>
  constexpr index_t wrap_index(grid_t<T, Layout> &&grid, index_t &&value) {
    if constexpr (Boundary == Lower) {
      return value + grid.extent(Axis) + Offset;
    } else if constexpr (Boundary == Upper) {
      return value - grid.extent(Axis) + Offset;
    } else {
      return value;
    }
  }

  template <Axis Axis, auto Offset, Boundary Boundary, class T, class Layout = std::layout_right>
  constexpr auto offset_on_axis(grid_t<T, Layout> &&grid, index_t &&i, index_t &&j, index_t &&k) {
    if constexpr (Axis == I) {
      return grid[wrap_index<Axis, Offset, Boundary, T, Layout>(
                      std::forward<grid_t<T, Layout> >(grid), std::forward<index_t>(i)),
                  j, k];
    } else if constexpr (Axis == J) {
      return grid[i,
                  wrap_index<Axis, Offset, Boundary, T, Layout>(
                      std::forward<grid_t<T, Layout> >(grid), std::forward<index_t>(j)),
                  k];
    } else if constexpr (Axis == K) {
      return grid[i, j,
                  wrap_index<Axis, Offset, Boundary, T, Layout>(
                      std::forward<grid_t<T, Layout> >(grid), std::forward<index_t>(k))];
    }
  }

  template <Axis Axis, Boundary Boundary, class T, std::size_t N, class Layout = std::layout_right>
  constexpr T sum_stencil(std::array<T, N> &&coeffs, grid_t<T> &&grid, index_t &&i, index_t &&j,
                          index_t &&k) {
    constexpr auto p = p_from_size(N);
    return [&]<auto... Indices>(std::integer_sequence<index_t, Indices...>) constexpr {
      return ((coeffs[Indices]
               * offset_on_axis<Axis, Indices - p, Boundary, T, Layout>(
                   std::forward<grid_t<T> >(grid), std::forward<index_t>(i),
                   std::forward<index_t>(j), std::forward<index_t>(k)))
              + ...);
    }(std::make_index_sequence<N>{});
  }
}  // namespace eedft::core::stencil

#endif  // EEDFT_CORE_FD_HELPERS_H
