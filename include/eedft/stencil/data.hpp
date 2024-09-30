
#ifndef EEDFT_STENCIL_DATA_HPP
#define EEDFT_STENCIL_DATA_HPP

#include <array>
#include <type_traits>

namespace eedft::stencil {
   template<class Out> requires std::is_integral_v<Out>
   constexpr Out compute_p(Out m, Out n) {
      return (m + 1) / 2 - 1 + n / 2;
   }

   template<class Out> requires std::is_integral_v<Out>
   constexpr Out stencil_size(Out m, Out n) {
      return 2 * compute_p(m, n) + 1;
   }


   template<class T, std::size_t, std::size_t>
   struct fd {
   };


   template<class T>
   struct fd<T, 1, 2> {
      static constexpr std::array<T, stencil_size<std::size_t>(1, 2)> coeffs = {-1.0 / 2.0, 0.0, 1.0 / 2.0};
   };


   template<class T>
   struct fd<T, 2, 2> {
      static constexpr std::array<T, stencil_size<std::size_t>(2, 2)> coeffs = {1.0 / 1.0, -2.0 / 1.0, 1.0 / 1.0};
   };


   template<class T>
   struct fd<T, 1, 4> {
      static constexpr std::array<T, stencil_size<std::size_t>(1, 4)> coeffs = {
         1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0
      };
   };


   template<class T>
   struct fd<T, 2, 4> {
      static constexpr std::array<T, stencil_size<std::size_t>(2, 4)> coeffs = {
         -1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0
      };
   };


   template<class T>
   struct fd<T, 1, 6> {
      static constexpr std::array<T, stencil_size<std::size_t>(1, 6)> coeffs = {
         -1.0 / 60.0, 3.0 / 20.0, -3.0 / 4.0, 0.0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0
      };
   };


   template<class T>
   struct fd<T, 2, 6> {
      static constexpr std::array<T, stencil_size<std::size_t>(2, 6)> coeffs = {
         1.0 / 90.0, -3.0 / 20.0, 3.0 / 2.0, -49.0 / 18.0, 3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0
      };
   };


   template<class T>
   struct fd<T, 1, 8> {
      static constexpr std::array<T, stencil_size<std::size_t>(1, 8)> coeffs = {
         1.0 / 280.0, -4.0 / 105.0, 1.0 / 5.0, -4.0 / 5.0, 0.0, 4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0
      };
   };


   template<class T>
   struct fd<T, 2, 8> {
      static constexpr std::array<T, stencil_size<std::size_t>(2, 8)> coeffs = {
         -1.0 / 560.0, 8.0 / 315.0, -1.0 / 5.0, 8.0 / 5.0, -205.0 / 72.0, 8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0,
         -1.0 / 560.0
      };
   };


   template<class T>
   struct fd<T, 1, 10> {
      static constexpr std::array<T, stencil_size<std::size_t>(1, 10)> coeffs = {
         -1.0 / 1260.0, 5.0 / 504.0, -5.0 / 84.0, 5.0 / 21.0, -5.0 / 6.0, 0.0, 5.0 / 6.0, -5.0 / 21.0, 5.0 / 84.0,
         -5.0 / 504.0, 1.0 / 1260.0
      };
   };


   template<class T>
   struct fd<T, 2, 10> {
      static constexpr std::array<T, stencil_size<std::size_t>(2, 10)> coeffs = {
         1.0 / 3150.0, -5.0 / 1008.0, 5.0 / 126.0, -5.0 / 21.0, 5.0 / 3.0, -5269.0 / 1800.0, 5.0 / 3.0, -5.0 / 21.0,
         5.0 / 126.0, -5.0 / 1008.0, 1.0 / 3150.0
      };
   };


   template<class T>
   struct fd<T, 1, 12> {
      static constexpr std::array<T, stencil_size<std::size_t>(1, 12)> coeffs = {
         1.0 / 5544.0, -1.0 / 385.0, 1.0 / 56.0, -5.0 / 63.0, 15.0 / 56.0, -6.0 / 7.0, 0.0, 6.0 / 7.0, -15.0 / 56.0,
         5.0 / 63.0, -1.0 / 56.0, 1.0 / 385.0, -1.0 / 5544.0
      };
   };


   template<class T>
   struct fd<T, 2, 12> {
      static constexpr std::array<T, stencil_size<std::size_t>(2, 12)> coeffs = {
         -1.0 / 16632.0, 2.0 / 1925.0, -1.0 / 112.0, 10.0 / 189.0, -15.0 / 56.0, 12.0 / 7.0, -5369.0 / 1800.0,
         12.0 / 7.0, -15.0 / 56.0, 10.0 / 189.0, -1.0 / 112.0, 2.0 / 1925.0, -1.0 / 16632.0
      };
   };


   template<class T>
   struct fd<T, 1, 14> {
      static constexpr std::array<T, stencil_size<std::size_t>(1, 14)> coeffs = {
         -1.0 / 24024.0, 7.0 / 10296.0, -7.0 / 1320.0, 7.0 / 264.0, -7.0 / 72.0, 7.0 / 24.0, -7.0 / 8.0, 0.0, 7.0 / 8.0,
         -7.0 / 24.0, 7.0 / 72.0, -7.0 / 264.0, 7.0 / 1320.0, -7.0 / 10296.0, 1.0 / 24024.0
      };
   };


   template<class T>
   struct fd<T, 2, 14> {
      static constexpr std::array<T, stencil_size<std::size_t>(2, 14)> coeffs = {
         1.0 / 84084.0, -7.0 / 30888.0, 7.0 / 3300.0, -7.0 / 528.0, 7.0 / 108.0, -7.0 / 24.0, 7.0 / 4.0,
         -266681.0 / 88200.0, 7.0 / 4.0, -7.0 / 24.0, 7.0 / 108.0, -7.0 / 528.0, 7.0 / 3300.0, -7.0 / 30888.0,
         1.0 / 84084.0
      };
   };
}

#endif //EEDFT_STENCIL_DATA_HPP

