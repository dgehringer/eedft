//
// Created by dominik on 8/17/24.
//

#ifndef FUNCUTILS_FUNCTOR_VECTOR_IMPL_HPP
#define FUNCUTILS_FUNCTOR_VECTOR_IMPL_HPP

#include "funcutils/functor/functor.hpp"

namespace funcutils::core {

template <> struct _functor_impl<std::vector> {
  
  template <class AA, class B, class Function, class A = remove_reference_t<AA>>
  static constexpr decltype(auto) fmap(Function &&f) {
    return [&f](std::vector<A> &&arg) -> std::vector<B> {
      std::vector<B> result(arg.size());
      for (size_t i = 0; i < arg.size(); ++i) {
        if constexpr (std::is_move_assignable_v<B> ||
                      std::is_trivially_move_assignable_v<B>) {
          result[i] = std::move(f(std::forward<A>(arg[i])));
        } else if constexpr (std::is_copy_assignable_v<B> ||
                             std::is_trivially_copy_assignable_v<B>) {
          result[i] = f(arg[i]);
        }
      }
      return result;
    };
  }
};
  
}

#endif // FUNCUTILS_FUNCTOR_VECTOR_IMPL_HPP
