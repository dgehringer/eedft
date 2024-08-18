//
// Created by dominik on 8/17/24.
//

#ifndef FUNCUTILS_FUNCTOR_VECTOR_IMPL_HPP
#define FUNCUTILS_FUNCTOR_VECTOR_IMPL_HPP

#include "funcutils/functor/functor.hpp"
#include "funcutils/utils.hpp"
#include <vector>
#include <utility>

namespace funcutils::core {

template <template <class...> class F> struct _functor_impl;

template <> struct _functor_impl<std::vector> {

  template <class AA, class B, class Function,
            class A = std::remove_reference_t<AA>>
  static constexpr decltype(auto) fmap(Function &&f) {
    return [&f](std::vector<A> &&arg) -> std::vector<B> {
      std::vector<B> result(arg.size());
      for (size_t i = 0; i < arg.size(); ++i) {
          result[i] = call<AA, B, Function>(std::forward<Function>(f), std::forward<A>(arg[i]));
      }
      return result;
    };
  }
};

} // namespace funcutils::core

#endif // FUNCUTILS_FUNCTOR_VECTOR_IMPL_HPP
