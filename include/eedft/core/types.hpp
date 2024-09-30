//
// Created by Dominik Gehringer on 30.09.24.
//

#ifndef EEDFT_TYPES_HPP
#define EEDFT_TYPES_HPP
#include <cstddef>
#include <mdspan>

namespace eedft::core {
    enum Axis: std::size_t {
        I = 0,
        J = 1,
        K = 2
    };

    using index_t = int;

    template<class T, class Layout = std::layout_right>
    using grid_t = std::mdspan<T, std::extents<index_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>,
        Layout>;
}

#endif //EEDFT_TYPES_HPP
