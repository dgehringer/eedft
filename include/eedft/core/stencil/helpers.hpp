//
// Created by Dominik Gehringer on 30.09.24.
//

#ifndef EEDFT_CORE_STENCIL_HELPERS_HPP
#define EEDFT_CORE_STENCIL_HELPERS_HPP
#include <mdspan>

#include "eedft/core/types.hpp"

namespace eedft::core::stencil {
    template<class Out> requires std::is_integral_v<Out>
    constexpr Out compute_p(Out m, Out n) {
        return (m + 1) / 2 - 1 + n / 2;
    }

    template<class Out> requires std::is_integral_v<Out>
    constexpr Out stencil_size(Out m, Out n) {
        return 2 * compute_p(m, n) + 1;
    }

    template<class Out> requires std::is_integral_v<Out>
    constexpr Out p_from_size(Out size) {
        static_assert(size % 2 == 0, "Size must be odd");
        return (size - 1) / 2;
    }

    template<class T, std::size_t N>
    constexpr T sum_stencil(const std::array<T, N>& coeffs, grid_t<T> const& grid) {
        constexpr auto coeff_indices = std::make_index_sequence<N>{};
    }
}

#endif //EEDFT_CORE_STENCIL_HELPERS_HPP
