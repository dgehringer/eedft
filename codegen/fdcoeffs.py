import os
import math
import argparse
from fractions import Fraction

Number = Fraction | float | int
Matrix = list[list[Number]]


def nonzero_row(matrix: Matrix, row: int = 0, col: int = 0) -> int | None:
    for row_index in range(row, len(matrix)):
        value = matrix[row][col]
        if value != 0 or not math.isclose(value, 0.0):
            return row_index
    else:
        return None


def compute_p(m: int, n: int) -> int:
    return ((m + 1) // 2) - 1 + n // 2


def gaussian_elimination(m: Matrix):
    nrows = len(m)
    for i in range(nrows):
        piv = i
        for j in range(i + 1, nrows):
            if abs(m[j][i]) > abs(m[piv][i]):
                piv = j
        if piv != i:
            m[i], m[piv] = m[piv], m[i]
        for j in range(i + 1, nrows):
            factor = m[j][i] / m[i][i]
            m[j] = [jj - factor * ii for ii, jj in zip(m[i], m[j])]
    x = [0] * nrows
    for i in range(nrows - 1, -1, -1):
        x[i] = (m[i][nrows] - sum(m[i][j] * x[j] for j in range(i + 1, nrows))) / m[i][
            i
        ]
    return x


def fdcoeffs(m: int, n: int):
    p = compute_p(m, n)
    matrix = [
        [Fraction(j**i, 1) for j in range(-p, p + 1)]
        + [Fraction(0 if i != m else math.factorial(m), 1)]
        for i in range(2 * p + 1)
    ]
    return gaussian_elimination(matrix)


HPP_PARTIAL_SPECIALISATION = """
template <class T> struct fd<T, {m}, {n}> {{
   static constexpr std::array<T, stencil_size<std::size_t>({m}, {n})> coeffs = {{{coeffs}}};
}};
"""

HPP_TEXT = """
#ifndef {guard_name}
#define {guard_name}

#include <array>
#include "eedft/core/stencil/helpers.hpp"

namespace {namespace} {{

template<class T, std::size_t, std::size_t>
struct fd {{}};

{specs}

}}

#endif //{guard_name}

"""


def format_coeffs(m: int, n: int) -> str:
    coeffs = ", ".join(
        "0.0" if c == 0 else f"{c.numerator:.1f} / {c.denominator:.1f}"
        for c in fdcoeffs(m, n)
    )
    return HPP_PARTIAL_SPECIALISATION.format(coeffs=coeffs, m=m, n=n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate central finite difference coefficients"
    )
    parser.add_argument(
        "--min-order", type=int, default=2, help="min. FD order to generate"
    )
    parser.add_argument(
        "--max-order", type=int, default=12, help="max. FD order to generate"
    )
    parser.add_argument("--min", type=int, default=1, help="min. derivative")
    parser.add_argument("--max", type=int, default=2, help="min. derivative")
    parser.add_argument("--namespace", type=str, default="")
    parser.add_argument("-o", type=argparse.FileType("w"), default="data.hpp")
    args = parser.parse_args()

    guard_name = (
        f"{args.namespace.replace('::', '_')}_{os.path.basename(args.o.name).replace('.', '_')}".upper()
    )
    specs = "\n".join(
        format_coeffs(m, n)
        for n in range(args.min_order, args.max_order + 1, 2)
        for m in range(args.min, args.max + 1)
    )
    args.o.write(
        HPP_TEXT.format(
            guard_name=guard_name,
            namespace=args.namespace,
            specs=specs
        )
    )
