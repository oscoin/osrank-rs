use quickcheck::Arbitrary;
use rand::Rng;

pub type Frequency = u32;

pub fn frequency<G: Rng, A: Arbitrary>(g: &mut G, xs: Vec<(Frequency, A)>) -> A {
    let tot: u32 = xs.iter().cloned().map(|(f, _)| f).sum();
    let choice = g.gen_range(1, tot);
    pick(choice, xs)
}

fn pick<A: Arbitrary>(n: u32, xs: Vec<(Frequency, A)>) -> A {
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
