use ndarray::prelude::*;
use ndarray_linalg::{Scalar, Solve};
use num_rational::Ratio;

mod core {
    pub mod math;
    pub mod stencils;
}

use crate::core::stencils::coefficients_orthogonal;

fn main() {
    // test
    println!("Hello, world! {:?}", coefficients_orthogonal(1, 8));
}
