extern crate fraction;
extern crate num_traits;
extern crate petgraph;

use std::fmt;
use std::ops::{Div, Mul, Rem};

use fraction::{Fraction, GenericFraction};
use num_traits::{Num, One, Signed, Zero};
use petgraph::graph::NodeIndex;
use petgraph::{Directed, Graph};

pub type Osrank = Fraction;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct RandomWalks {
    random_walks_internal: Vec<RandomWalk>,
}

impl RandomWalks {
    pub fn new() -> Self {
        RandomWalks {
            random_walks_internal: Vec::new(),
        }
    }

    pub fn add_walk(&mut self, walk: RandomWalk) {
        self.random_walks_internal.push(walk);
    }

    pub fn len(&self) -> usize {
        self.random_walks_internal.len()
    }

    pub fn count_visits(&self, idx: NodeIndex) -> usize {
        self.random_walks_internal.iter().map(|rw| rw.count_visits(&idx)).sum()
    }
}

#[derive(Debug, Default, PartialEq, Eq, Hash)]
pub struct RandomWalk {
    random_walk_internal: Vec<NodeIndex>,
}

impl RandomWalk {
    pub fn new() -> Self {
        RandomWalk {
            random_walk_internal: Vec::new(),
        }
    }

    pub fn add_next(&mut self, idx: NodeIndex) {
        self.random_walk_internal.push(idx);
    }

    pub fn count_visits(&self, idx: &NodeIndex) -> usize {
        self.random_walk_internal.iter().filter(|i| i == &idx).count()
    }
}

// Just an alias for now.
pub type SeedSet = ();

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

#[derive(Debug)]
pub enum Dependency {
    Contrib(Weight),
    ContribPrime(Weight),
    Maintain(Weight),
    MaintainPrime(Weight),
    Depend(Weight),
}

impl fmt::Display for Dependency {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            Dependency::Contrib(ref w) => write!(f, "{:.5}", w.get_weight),
            Dependency::ContribPrime(ref w) => write!(f, "{:.5}", w.get_weight),
            Dependency::Maintain(ref w) => write!(f, "{:.5}", w.get_weight),
            Dependency::MaintainPrime(ref w) => write!(f, "{:.5}", w.get_weight),
            Dependency::Depend(ref w) => write!(f, "{:.5}", w.get_weight),
        }
    }
}

impl  Dependency {
    pub fn get_weight(&self) -> &Weight {
        match self {
            Dependency::Contrib(ref w) => w,
            Dependency::ContribPrime(ref w) => w,
            Dependency::Maintain(ref w) => w,
            Dependency::MaintainPrime(ref w) => w,
            Dependency::Depend(ref w) => w,
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd)]
pub struct ProjectAttributes {
    pub id: String,
    pub osrank: Option<Osrank>,
}

#[derive(Debug, PartialOrd, Eq, PartialEq)]
pub struct AccountAttributes {
    pub id: String,
    pub osrank: Option<Osrank>,
}

#[derive(Debug, PartialOrd, PartialEq, Eq)]
pub enum Artifact {
    Project(ProjectAttributes),
    Account(AccountAttributes),
}

impl Artifact {
    // Set the osrank attribute of the Artifact
    pub fn set_osrank(&mut self, rank: Option<Osrank>) {
        match self {
            Artifact::Project(attrs) => attrs.osrank = rank,
            Artifact::Account(attrs) => attrs.osrank = rank,
        }
    }
}

impl fmt::Display for Artifact {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            Artifact::Project(ref attrs) => write!(f, "id: {} osrank: {:.5}", attrs.id, attrs.osrank.unwrap()),
            Artifact::Account(ref attrs) => write!(f, "id: {} osrank: {:.5}", attrs.id, attrs.osrank.unwrap()),
        }
    }
}

/// The network graph from the paper, comprising of both accounts and projects.
#[derive(Debug, Default)]
pub struct Network {
    pub from_graph: Graph<Artifact, Dependency, Directed>,
}

impl Network {
    /// Adds an Artifact to the Network.
    pub fn add_artifact(&mut self, artifact: Artifact) {
        let _ = self.from_graph.add_node(artifact);
    }

    /// Adds a Dependency to the Network. It's unsafe in the sense it's
    /// callers' responsibility to ensure that the source and target exist
    /// in the input Network.
    pub fn unsafe_add_dependency(&mut self, source: u32, target: u32, dependency: Dependency) {
        let _ =
            self.from_graph
                .add_edge(NodeIndex::from(source), NodeIndex::from(target), dependency);
    }

    // So far only for debugging. Prints all artifacts with their ids and osranks
    pub fn print_artifacts(&self) {
        for arti in self.from_graph.raw_nodes().iter().map(|node| &node.weight) {
            println!("{}", arti);
        }
    }
}
