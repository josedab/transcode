//! Rational number type for precise time and rate representation.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

/// A rational number represented as a numerator and denominator.
///
/// Used for precise representation of frame rates, sample rates, and time bases.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rational {
    /// Numerator
    pub num: i64,
    /// Denominator (must be positive)
    pub den: i64,
}

impl Rational {
    /// Create a new rational number.
    ///
    /// # Panics
    ///
    /// Panics if denominator is zero.
    pub fn new(num: i64, den: i64) -> Self {
        assert!(den != 0, "Denominator cannot be zero");
        let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
        Self { num, den }
    }

    /// Create a rational from an integer.
    pub fn from_int(n: i64) -> Self {
        Self { num: n, den: 1 }
    }

    /// Create a zero rational.
    pub const fn zero() -> Self {
        Self { num: 0, den: 1 }
    }

    /// Create a rational representing one.
    pub const fn one() -> Self {
        Self { num: 1, den: 1 }
    }

    /// Check if this rational is zero.
    pub fn is_zero(&self) -> bool {
        self.num == 0
    }

    /// Check if this rational is positive.
    pub fn is_positive(&self) -> bool {
        self.num > 0
    }

    /// Check if this rational is negative.
    pub fn is_negative(&self) -> bool {
        self.num < 0
    }

    /// Reduce the rational to its simplest form.
    pub fn reduce(&self) -> Self {
        if self.num == 0 {
            return Self { num: 0, den: 1 };
        }
        let g = gcd(self.num.unsigned_abs(), self.den.unsigned_abs());
        Self {
            num: self.num / g as i64,
            den: self.den / g as i64,
        }
    }

    /// Convert to f64.
    pub fn to_f64(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Convert to f32.
    pub fn to_f32(&self) -> f32 {
        self.num as f32 / self.den as f32
    }

    /// Get the reciprocal of this rational.
    ///
    /// # Panics
    ///
    /// Panics if the numerator is zero.
    pub fn recip(&self) -> Self {
        assert!(self.num != 0, "Cannot take reciprocal of zero");
        if self.num < 0 {
            Self::new(-self.den, -self.num)
        } else {
            Self::new(self.den, self.num)
        }
    }

    /// Multiply by an integer.
    pub fn mul_int(&self, n: i64) -> Self {
        Self::new(self.num * n, self.den)
    }

    /// Divide by an integer.
    pub fn div_int(&self, n: i64) -> Self {
        Self::new(self.num, self.den * n)
    }

    /// Rescale a value from this time base to another.
    pub fn rescale(&self, value: i64, target: Rational) -> i64 {
        // value * self / target
        let num = value as i128 * self.num as i128 * target.den as i128;
        let den = self.den as i128 * target.num as i128;
        (num / den) as i64
    }
}

impl Default for Rational {
    fn default() -> Self {
        Self::zero()
    }
}

impl fmt::Debug for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rational({}/{})", self.num, self.den)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        let lhs = self.num as i128 * other.den as i128;
        let rhs = other.num as i128 * self.den as i128;
        lhs.cmp(&rhs)
    }
}

impl Add for Rational {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let num = self.num * rhs.den + rhs.num * self.den;
        let den = self.den * rhs.den;
        Self::new(num, den).reduce()
    }
}

impl Sub for Rational {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let num = self.num * rhs.den - rhs.num * self.den;
        let den = self.den * rhs.den;
        Self::new(num, den).reduce()
    }
}

impl Mul for Rational {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.num * rhs.num, self.den * rhs.den).reduce()
    }
}

impl Div for Rational {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        // Division by multiplying by reciprocal: a/b ÷ c/d = a/b × d/c = (a*d)/(b*c)
        Self::new(self.num * rhs.den, self.den * rhs.num).reduce()
    }
}

impl From<i32> for Rational {
    fn from(n: i32) -> Self {
        Self::from_int(n as i64)
    }
}

impl From<i64> for Rational {
    fn from(n: i64) -> Self {
        Self::from_int(n)
    }
}

impl From<(i32, i32)> for Rational {
    fn from((num, den): (i32, i32)) -> Self {
        Self::new(num as i64, den as i64)
    }
}

impl From<(i64, i64)> for Rational {
    fn from((num, den): (i64, i64)) -> Self {
        Self::new(num, den)
    }
}

/// Calculate the greatest common divisor using Euclidean algorithm.
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational_new() {
        let r = Rational::new(1, 2);
        assert_eq!(r.num, 1);
        assert_eq!(r.den, 2);
    }

    #[test]
    fn test_rational_negative_den() {
        let r = Rational::new(1, -2);
        assert_eq!(r.num, -1);
        assert_eq!(r.den, 2);
    }

    #[test]
    fn test_rational_reduce() {
        let r = Rational::new(4, 8).reduce();
        assert_eq!(r.num, 1);
        assert_eq!(r.den, 2);
    }

    #[test]
    fn test_rational_add() {
        let a = Rational::new(1, 2);
        let b = Rational::new(1, 3);
        let c = a + b;
        assert_eq!(c, Rational::new(5, 6));
    }

    #[test]
    fn test_rational_mul() {
        let a = Rational::new(2, 3);
        let b = Rational::new(3, 4);
        let c = a * b;
        assert_eq!(c, Rational::new(1, 2));
    }

    #[test]
    fn test_rational_to_f64() {
        let r = Rational::new(1, 4);
        assert!((r.to_f64() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_rational_ord() {
        let a = Rational::new(1, 2);
        let b = Rational::new(1, 3);
        assert!(a > b);
    }
}
