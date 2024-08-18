//
// Created by dominik on 8/18/24.
//

#ifndef FUNCUTILS_UTILS_HPP
#define FUNCUTILS_UTILS_HPP

#include <utility>

namespace funcutils::core {

template <class AA, class B, class Function, class A = std::remove_reference_t<AA>>
static constexpr auto call(Function &&f, A &&arg) {
  if constexpr (std::is_rvalue_reference_v<AA> &&
                std::is_move_constructible_v<B>) {
    return std::move(f(std::move(std::forward<A>(arg))));
  } else if constexpr (std::is_move_constructible_v<B>) {
    return std::move(f(std::forward<A>(arg)));
  } else {
    return f(std::forward<A>(arg));
  }
}
} // namespace funcutils::core
#endif // FUNCUTILS_UTILS_HPP
