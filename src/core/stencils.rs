use crate::core::math::{factorial, gaussian_elimination};
use num_rational::Ratio;

type FDCoefficients = Vec<Ratio<i64>>;
pub fn compute_p(m: usize, n: usize) -> usize {
    return (2 * ((m + 1) / 2) - 2 + n) / 2;
}

pub fn coefficients_orthogonal(m: usize, n: usize) -> FDCoefficients {
    let p = compute_p(m, n) as i64;
    let length = (2 * p) + 1;
    // first row of the coefficient matrix
    let first_row: Vec<Ratio<i64>> = (0..length).map(|_| Ratio::from_integer(1)).collect();
    let mut coefficients: Vec<Vec<Ratio<i64>>> = (1..length)
        .map(|j| {
            (-p..(p + 1))
                .map(move |i| Ratio::from_integer(i.pow(j as u32)))
                .collect()
        })
        .collect();
    coefficients.insert(0, first_row);
    (0..(length as usize)).for_each(|index| {
        let value = if m == index { factorial(m as i64) } else { 0 };
        coefficients[index].push(Ratio::from_integer(value));
    });
    gaussian_elimination(&mut coefficients)
}

#[cfg(test)]
mod tests {
    use num_traits::NumCast;

    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_coefficients_orthogonal() {
        let compare_coefficients: HashMap<i32, HashMap<i32, Vec<Ratio<i64>>>> = HashMap::from([
            (
                1,
                HashMap::from([
                    (
                        2,
                        vec![Ratio::new(-1, 2), Ratio::from_integer(0), Ratio::new(1, 2)],
                    ),
                    (
                        4,
                        vec![
                            Ratio::new(1, 12),
                            Ratio::new(-2, 3),
                            Ratio::from(0),
                            Ratio::new(2, 3),
                            Ratio::new(-1, 12),
                        ],
                    ),
                    (
                        6,
                        vec![
                            Ratio::new(-1, 60),
                            Ratio::new(3, 20),
                            Ratio::new(-3, 4),
                            Ratio::from(0),
                            Ratio::new(3, 4),
                            Ratio::new(-3, 20),
                            Ratio::new(1, 60),
                        ],
                    ),
                    (
                        8,
                        vec![
                            Ratio::new(1, 280),
                            Ratio::new(-4, 105),
                            Ratio::new(1, 5),
                            Ratio::new(-4, 5),
                            Ratio::from(0),
                            Ratio::new(4, 5),
                            Ratio::new(-1, 5),
                            Ratio::new(4, 105),
                            Ratio::new(-1, 280),
                        ],
                    ),
                ]),
            ),
            (
                2,
                HashMap::from([
                    (2, vec![Ratio::from(1), Ratio::from(-2), Ratio::from(1)]),
                    (
                        4,
                        vec![
                            Ratio::new(-1, 12),
                            Ratio::new(4, 3),
                            Ratio::new(-5, 2),
                            Ratio::new(4, 3),
                            Ratio::new(-1, 12),
                        ],
                    ),
                    (
                        6,
                        vec![
                            Ratio::new(1, 90),
                            Ratio::new(-3, 20),
                            Ratio::new(3, 2),
                            Ratio::new(-49, 18),
                            Ratio::new(3, 2),
                            Ratio::new(-3, 20),
                            Ratio::new(1, 90),
                        ],
                    ),
                    (
                        8,
                        vec![
                            Ratio::new(-1, 560),
                            Ratio::new(8, 315),
                            Ratio::new(-1, 5),
                            Ratio::new(8, 5),
                            Ratio::new(-205, 72),
                            Ratio::new(8, 5),
                            Ratio::new(-1, 5),
                            Ratio::new(8, 315),
                            Ratio::new(-1, 560),
                        ],
                    ),
                ]),
            ),
        ]);
        for (m, orders) in compare_coefficients.into_iter() {
            for (n, coefficients) in orders.into_iter() {
                coefficients_orthogonal(m as usize, n as usize)
                    .into_iter()
                    .zip(coefficients)
                    .for_each(|(is, should)| assert_eq!(is, should));
            }
        }
    }
}
