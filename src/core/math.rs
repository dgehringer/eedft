use num_traits::{one, zero, Num};
use std::ops::{Mul, Sub};

pub fn factorial<T: Num + Copy>(num: T) -> T {
    let o: T = one();
    let z: T = zero();
    if num == z || num == o {
        o
    } else {
        factorial(num - o) * num
    }
}

pub fn gaussian_elimination<T: Num + Copy>(matrix: &mut [Vec<T>]) -> Vec<T> {
    let size = matrix.len();
    assert_eq!(size, matrix[0].len() - 1);
    let z: T = zero();

    for i in 0..size - 1 {
        for j in i..size - 1 {
            echelon(matrix, i, j);
        }
    }

    for i in (1..size).rev() {
        eliminate(matrix, i);
    }

    // Disable cargo clippy warnings about needless range loops.
    // Checking the diagonal like this is simpler than any alternative.
    #[allow(clippy::needless_range_loop)]
    for i in 0..size {
        if matrix[i][i] == z {
            println!("Infinitely many solutions");
        }
    }

    let mut result: Vec<T> = vec![z; size];
    for i in 0..size {
        result[i] = matrix[i][size] / matrix[i][i];
    }
    result
}

fn echelon<T: Num + Copy>(matrix: &mut [Vec<T>], i: usize, j: usize) {
    let size = matrix.len();
    let z: T = zero();
    if matrix[i][i] == z {
    } else {
        let factor = matrix[j + 1][i] / matrix[i][i];
        (i..size + 1).for_each(|k| {
            matrix[j + 1][k] = matrix[j + 1][k] - (factor * matrix[i][k]);
        });
    }
}

fn eliminate<T: Num + Copy>(matrix: &mut [Vec<T>], i: usize) {
    let size = matrix.len();
    let z: T = zero();
    if matrix[i][i] == z {
    } else {
        for j in (1..i + 1).rev() {
            let factor = matrix[j - 1][i] / matrix[i][i];
            for k in (0..size + 1).rev() {
                matrix[j - 1][k] = matrix[j - 1][k] - (factor * matrix[i][k]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::gaussian_elimination;

    #[test]
    fn test_gaussian_elimination() {
        let mut matrix: Vec<Vec<f32>> = vec![
            vec![1.5, 2.0, 1.0, -1.0, -2.0, 1.0, 1.0],
            vec![3.0, 3.0, -1.0, 16.0, 18.0, 1.0, 1.0],
            vec![1.0, 1.0, 3.0, -2.0, -6.0, 1.0, 1.0],
            vec![1.0, 1.0, 99.0, 19.0, 2.0, 1.0, 1.0],
            vec![1.0, -2.0, 16.0, 1.0, 9.0, 10.0, 1.0],
            vec![1.0, 3.0, 1.0, -5.0, 1.0, 1.0, 95.0],
        ];
        let result = vec![
            -264.05893, 159.63196, -6.156921, 35.310387, -18.806696, 81.67839,
        ];
        assert_eq!(gaussian_elimination(&mut matrix), result);
    }
}
