//
// Created by dominik on 8/17/24.
//

#ifndef FUNCUTILS_FUNCTOR_HPP
#define FUNCUTILS_FUNCTOR_HPP

#include <functional>
#include "funcutils/functor/impl/vector.hpp"
#include "funcutils/functor/impl/optional.hpp"

namespace funcutils::core {

template <template <class...> class F> struct _functor_impl;

template <template <class> class F, class Function, class R, class A>
constexpr auto _functor_helper(Function &&f, const std::function<R(A)> &) {
  return _functor_impl<F>::template fmap<A, R, Function>(
      std::forward<Function>(f));
}


template <template <class> class... F> struct functor;

template <template <class> class F> struct functor<F> {

  template <class Function> static constexpr decltype(auto) fmap(Function &&f) {
    using std_function_type =
        decltype(std::function{std::forward<Function>(f)});
    return _functor_helper<F>(std::forward<Function>(f), std_function_type{});
  }
};


template <template <class> class G, template <class> class F>
struct functor<G, F> {
  template <class Function> static constexpr decltype(auto) fmap(Function &&f) {
    return functor<G>::template fmap(functor<F>::template fmap(std::forward<Function>(f)));
  }
};


template <template <class> class H, template <class> class ...G>
struct functor<H, G...> {

  template <class Function> static constexpr decltype(auto) fmap(Function &&f) {
    return functor<H>::template fmap(functor<G...>::template fmap(std::forward<Function>(f)));
  }
};


}

#endif // FUNCUTILS_FUNCTOR_HPP
