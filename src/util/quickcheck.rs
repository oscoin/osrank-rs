use num_traits::Num;
use quickcheck::{Arbitrary, Gen};
use rand::distributions;
use rand::distributions::Distribution;
use rand::Rng;

#[derive(Clone)]
pub struct DebugDisplay<A> {
    pub get_internal: A,
}

impl<A: Arbitrary> Arbitrary for DebugDisplay<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        DebugDisplay {
            get_internal: Arbitrary::arbitrary(g),
        }
    }
}

pub type Frequency = u32;

pub fn frequency<G: Rng, A>(g: &mut G, xs: Vec<(Frequency, A)>) -> A {
    let mut tot: u32 = 0;

    for (f, _) in &xs {
        tot += f
    }

    let choice = g.gen_range(1, tot);
    pick(choice, xs)
}

fn pick<A>(n: u32, xs: Vec<(Frequency, A)>) -> A {
    let mut acc = n;

    for (k, x) in xs {
        if acc <= k {
            return x;
        } else {
            acc -= k;
        }
    }

    panic!("QuickCheck.pick used with an empty vector");
}

#[derive(Debug, Clone)]
pub struct Positive<N> {
    pub get_positive: N,
}

// An alphanumeric string.
#[derive(Debug, Clone)]
pub struct Alphanumeric {
    pub get_alphanumeric: String,
}

#[derive(Debug, Clone)]
pub struct NonEmpty<E> {
    pub get_nonempty: Vec<E>,
}

#[derive(Debug, Clone)]
pub struct Vec32<E> {
    pub get_vec32: Vec<E>,
}

impl<E> Arbitrary for Vec32<E>
where
    E: Arbitrary,
{
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let mut n = 0;
        let mut xs = Vec::with_capacity(32);

        while n < 32 {
            xs.push(Arbitrary::arbitrary(g));
            n += 1;
        }

        Vec32 { get_vec32: xs }
    }
}

impl<N> Arbitrary for Positive<N>
where
    N: Num + Clone + Arbitrary,
{
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let mut x: N = Arbitrary::arbitrary(g);

        while x == N::zero() {
            x = Arbitrary::arbitrary(g);
        }

        Positive { get_positive: x }
    }
}

impl<E> Arbitrary for NonEmpty<E>
where
    E: Arbitrary,
{
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let mut x: Vec<E> = Arbitrary::arbitrary(g);

        while x.len() == 0 {
            x = Arbitrary::arbitrary(g);
        }

        NonEmpty { get_nonempty: x }
    }
}

impl Arbitrary for Alphanumeric {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let n = g.gen_range(1, g.size() + 1);
        Alphanumeric {
            get_alphanumeric: distributions::Alphanumeric.sample_iter(g).take(n).collect(),
        }
    }
}
