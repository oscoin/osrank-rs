#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate fraction;
extern crate num_traits;
extern crate petgraph;
extern crate quickcheck;

use fraction::{Fraction, GenericFraction};
use num_traits::{Num, One, Signed, Zero};
use quickcheck::{Arbitrary, Gen};
use std::fmt;
use std::iter::Sum;
use std::ops::{Deref, Div, Mul, Rem};

/// Trait to calculate the edge's weight on the fly.
pub mod dynamic_weight;
pub mod mock;
pub mod network;
pub mod walk;

/// The `Osrank` score, modeled as a fraction. It has a default value of `Zero`,
/// in case no `Osrank` is provided/calculated yet.
#[derive(Copy, Clone, Display, Add, PartialEq, PartialOrd)]
pub struct Osrank(pub Fraction);

impl fmt::Debug for Osrank {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Osrank(frac) = self;
        write!(f, "{}/{}", frac.numer().unwrap(), frac.denom().unwrap())
    }
}

impl Deref for Osrank {
    type Target = Fraction;

    #[must_use]
    fn deref(&self) -> &Self::Target {
        match self {
            Osrank(f) => f,
        }
    }
}

impl Zero for Osrank {
    fn zero() -> Self {
        Osrank(Zero::zero())
    }

    fn is_zero(&self) -> bool {
        self.deref().is_zero()
    }
}

impl One for Osrank {
    fn one() -> Self {
        Osrank(One::one())
    }
}

impl Mul for Osrank {
    type Output = Osrank;

    fn mul(self, rhs: Self) -> Self::Output {
        let Osrank(r1) = self;
        let Osrank(r2) = rhs;
        Osrank(r1 * r2)
    }
}

impl Sum for Osrank {
    fn sum<I: Iterator<Item = Osrank>>(iter: I) -> Osrank {
        Osrank(iter.fold(GenericFraction::zero(), |acc, Osrank(x)| acc + x))
    }
}

impl Arbitrary for Osrank {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let denom = g.next_u32().min(u32::max_value() - 1); // avoids 0.
        Osrank(Fraction::new(g.next_u32(), denom + 1))
    }
}

impl std::convert::From<Weight> for Osrank {
    fn from(w: Weight) -> Osrank {
        Osrank(GenericFraction::new(
            *w.get_weight.numer().unwrap() as u64,
            *w.get_weight.denom().unwrap() as u64,
        ))
    }
}

impl std::convert::From<f64> for Osrank {
    fn from(w: f64) -> Osrank {
        Osrank(GenericFraction::from(w))
    }
}

/// The number of random walks the algorithm has to perform for each node.
pub type R = u32;

#[derive(Clone, Copy, PartialEq, Add, Sub, Neg, PartialOrd)]
pub struct Weight {
    get_weight: GenericFraction<u32>,
}

impl std::convert::From<f64> for Weight {
    fn from(t: f64) -> Weight {
        Weight {
            get_weight: GenericFraction::from(t),
        }
    }
}

impl Arbitrary for Weight {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let denom = g.next_u32().min(u32::max_value() - 1); // avoids 0.
        Weight {
            get_weight: GenericFraction::new(g.next_u32(), denom + 1),
        }
    }
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

impl From<u32> for Weight {
    fn from(t: u32) -> Self {
        Weight {
            get_weight: GenericFraction::new(t, 1u32),
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
