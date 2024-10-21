//
// Created by Dominik Gehringer on 30.09.24.
//

#ifndef EEDFT_CORE_HELPERS_H
#define EEDFT_CORE_HELPERS_H
#include <type_traits>

#include "eedft/core/types.h"

namespace eedft::core {
    template<auto Start, auto End, auto Inc, class Fn>
    constexpr void constexpr_for(Fn &&f) {
        if constexpr (Start < End) {
            f(std::integral_constant<decltype(Start), Start>());
            constexpr_for<Start + Inc, End, Inc>(f);
        }
    }

}

#endif //EEDFT_CORE_HELPERS_H
