//
// Created by Dominik Gehringer on 30.09.24.
//

#ifndef EEDFT_CORE_HELPERS_HPP
#define EEDFT_CORE_HELPERS_HPP
#include <type_traits>

namespace eedft::core {
    template<auto Start, auto End, auto Inc, class F>
    constexpr void constexpr_for(F &&f) {
        if constexpr (Start < End) {
            f(std::integral_constant<decltype(Start), Start>());
            constexpr_for<Start + Inc, End, Inc>(f);
        }
    }
}

#endif //EEDFT_CORE_HELPERS_HPP
