#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate fraction;
extern crate num_traits;
extern crate petgraph;

use fraction::{Fraction, GenericFraction};
use num_traits::{Num, One, Signed, Zero};
use std::fmt;
use std::ops::{Div, Mul, Rem};

pub mod mock;
pub mod network;
pub mod walk;

/// The `Osrank` score, modeled as a fraction. It has a default value of `Zero`,
/// in case no `Osrank` is provided/calculated yet.
pub type Osrank = Fraction;

/// The number of random walks the algorithm has to perform for each node.
pub type R = u32;

/// The "pruning threshold" for the initial phase of the Osrank computation.
/// The objective of the initial phase is to prune any node from the graph
/// falling below this threshold, to avoid sybil attacks and mitigate other
/// hostile behaviours.
pub type Tau = f64;

#[derive(Clone, Copy, PartialEq, Add, Sub, Neg, PartialOrd)]
pub struct Weight {
    get_weight: GenericFraction<u32>,
}

impl fmt::Debug for Weight {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match (self.get_weight.numer(), self.get_weight.denom()) {
            (Some(n), Some(d)) => write!(f, "{}/{}", n, d),
            _ => write!(f, "NaN"),
        }
    }
}

impl fmt::Display for Weight {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.get_weight)
    }
}

impl Weight {
    pub fn new(numerator: u32, denominator: u32) -> Self {
        Weight {
            get_weight: GenericFraction::new(numerator, denominator),
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match (self.get_weight.numer(), self.get_weight.denom()) {
            (Some(n), Some(d)) => Some(f64::from(*n) / f64::from(*d)),
            _ => None,
        }
    }
}

impl Default for Weight {
    fn default() -> Self {
        One::one()
    }
}

impl std::convert::From<Weight> for f64 {
    fn from(w: Weight) -> Self {
        w.as_f64().unwrap()
    }
}

impl Mul for Weight {
    type Output = Weight;

    fn mul(self, rhs: Self) -> Self::Output {
        Weight {
            get_weight: self.get_weight * rhs.get_weight,
        }
    }
}

impl Signed for Weight {
    fn abs(self: &Self) -> Self {
        Weight {
            get_weight: self.get_weight.abs(),
        }
    }

    fn abs_sub(self: &Self, other: &Self) -> Self {
        Weight {
            get_weight: self.get_weight.abs_sub(&other.get_weight),
        }
    }

    fn signum(self: &Self) -> Self {
        Weight {
            get_weight: self.get_weight.signum(),
        }
    }

    fn is_positive(self: &Self) -> bool {
        self.get_weight.is_positive()
    }

    fn is_negative(self: &Self) -> bool {
        self.get_weight.is_negative()
    }
}

impl Div for Weight {
    type Output = Weight;

    fn div(self, rhs: Self) -> Self::Output {
        Weight {
            get_weight: self.get_weight / rhs.get_weight,
        }
    }
}

impl Rem for Weight {
    type Output = Weight;

    fn rem(self, rhs: Self) -> Self::Output {
        Weight {
            get_weight: self.get_weight.rem(rhs.get_weight),
        }
    }
}

impl Num for Weight {
    type FromStrRadixErr = fraction::ParseRatioError;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let inner = Num::from_str_radix(str, radix)?;
        Ok(Weight { get_weight: inner })
    }
}

impl One for Weight {
    fn one() -> Self {
        Weight::new(1, 1)
    }
}

impl Zero for Weight {
    fn zero() -> Self {
        Weight::new(0, 1)
    }

    fn is_zero(&self) -> bool {
        self.get_weight.numer() == Some(&0)
    }
}

/// The hyperparams from the paper, which are used to weight the edges.
pub struct HyperParams {
    pub contrib_factor: Weight,
    pub contrib_prime_factor: Weight,
    pub depend_factor: Weight,
    pub maintain_factor: Weight,
    pub maintain_prime_factor: Weight,
}

/// A default implementation based on the values from the paper.
impl Default for HyperParams {
    fn default() -> Self {
        HyperParams {
            contrib_factor: Weight::new(1, 7),
            contrib_prime_factor: Weight::new(2, 5),
            depend_factor: Weight::new(4, 7),
            maintain_factor: Weight::new(2, 7),
            maintain_prime_factor: Weight::new(3, 5),
        }
    }
}

/// The damping factors for project and accounts
pub struct DampingFactors {
    pub project: f64,
    pub account: f64,
}

/// The default for the damping factors in other ranks.
/// The whitepaper did not suggest values for the damping factors.
impl Default for DampingFactors {
    fn default() -> Self {
        DampingFactors {
            project: 0.85,
            account: 0.85,
        }
    }
}
