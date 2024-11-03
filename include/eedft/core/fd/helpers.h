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

  template <index_t Offset, Boundary Boundary>
  constexpr auto wrap(index_t &&index, index_t &&length) {
    if constexpr (Boundary == Lower) {
      return index + length + Offset;
    } else if constexpr (Boundary == Upper) {
      return index - length + Offset;
    } else {
      return index;
    }
  }

  template <Axis Axis, index_t Offset, Boundary Boundary, class T, class Layout = std::layout_right>
  constexpr T grid_at(grid_t<T, Layout> &&grid, index_t &&i, index_t &&j, index_t &&k) {
    constexpr auto length = grid.extent(Axis);
    if constexpr (Axis == I) {
      return grid[wrap<Offset, Boundary>(std::forward<index_t>(i), std::forward<index_t>(length)),
                  std::forward<index_t>(j), std::forward<index_t>(k)];
    } else if constexpr (Axis == J) {
      return grid[std::forward<index_t>(i),
                  wrap<Offset, Boundary>(std::forward<index_t>(j), std::forward<index_t>(length)),
                  std::forward<index_t>(k)];
    } else if constexpr (Axis == K) {
      return grid[std::forward<index_t>(i), std::forward<index_t>(j),
                  wrap<Offset, Boundary>(std::forward<index_t>(k), std::forward<index_t>(length))];
    } else {
      static_assert(false);
    }
  }

  template <Axis Axis, Boundary Boundary, class T, std::size_t N, class Layout = std::layout_right>
  constexpr T sum_stencil(std::array<T, N> &&coeffs, grid_t<T, Layout> &&grid, index_t &&i,
                          index_t &&j, index_t &&k) {
    constexpr auto P = p_from_size<index_t>(N);
    constexpr auto sum_range
        = [&]<auto Offset, core::Boundary Bound, auto CoeffOffset = 0, auto... Indices>(
              std::integer_sequence<index_t, Indices...>)
              ->T constexpr {
      return ((coeffs[Indices + CoeffOffset]
               * grid_at<Axis, Indices + Offset, Bound, T, Layout>(
                   std::forward<grid_t<T, Layout>>(grid), std::forward<index_t>(i),
                   std::forward<index_t>(j), std::forward<index_t>(k)))
              + ...);
    };
    if constexpr (Boundary == None) {
      /* In case we are not in a boundary region we sum over the whole stencil length.
       * We do not have to care about wrapping the indices 0, N
       */
      return sum_range<-P, None, 0>(std::make_index_sequence<N>{});
    } else if constexpr (Boundary == Lower) {
      /* In case of a lower bound we only have to wrap the indices for the left (-p, 0) coords
       * on the right site (0, p+1) we can safely assume to be in the bulk region
       */
      T sum{coeffs[P] * grid[i, j, k]};
      sum += sum_range<-P, Lower, 0>(std::make_index_sequence<P>{});
      sum += sum_range<1, None, P + 1>(std::make_index_sequence<P>{});
      return sum;
    } else if constexpr (Boundary == Upper) {
      T sum{coeffs[P] * grid[i, j, k]};
      sum += sum_range<-P, None, 0>(std::make_index_sequence<P>{});
      sum += sum_range<1, Upper, P + 1>(std::make_index_sequence<P>{});
      return sum;
    }
  }

}  // namespace eedft::core::fd

#endif  // EEDFT_CORE_FD_HELPERS_H
