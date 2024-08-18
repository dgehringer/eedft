//
// Created by dominik on 8/17/24.
//

#ifndef FUNCUTILS_FUNCTOR_OPTIONAL_IMPL_HPP
#define FUNCUTILS_FUNCTOR_OPTIONAL_IMPL_HPP

#include <optional>
#include "funcutils/functor/functor.hpp"


namespace funcutils::core {

template <template <class...> class F> struct _functor_impl;

template <> struct _functor_impl<std::optional> {

  template <class AA, class B, class Function, class A = std::remove_reference_t<AA>>
  static constexpr decltype(auto) fmap(Function &&f) {
      return [&f](std::optional<A> &&arg) -> std::optional<B> {
        if (!arg.has_value()) return std::nullopt;
        else{
          auto mapped_value= call<AA, B, Function>(std::forward<Function>(f), std::forward<A>(arg.value()));
          return std::make_optional(mapped_value);
        }
    };
  }
};

} // namespace funcutils::core

#endif // FUNCUTILS_FUNCTOR_OPTIONAL_IMPL_HPP
