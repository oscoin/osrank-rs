extern crate fraction;
extern crate petgraph;

use std::fmt;

use fraction::{Fraction, GenericFraction};
use petgraph::graph::NodeIndex;
use petgraph::{Directed, Graph};

type Osrank = Fraction;

#[derive(Debug)]
pub struct Weight {
    get_weight: GenericFraction<u32>,
}

impl Weight {
    pub fn new(numerator: u32, denominator: u32) -> Self {
        Weight {
            get_weight: GenericFraction::new(numerator, denominator),
        }
    }
}

/// The hyperparams from the paper, which are used to weight the edges.
pub struct HyperParams {
    contrib_factor: Weight,
    contrib_prime_factor: Weight,
    depend_factor: Weight,
    maintain_factor: Weight,
    maintain_prime_factor: Weight,
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

#[derive(Debug, PartialEq, Eq, PartialOrd)]
pub struct ProjectAttributes {
    pub id: String,
    pub osrank: Option<Osrank>,
}

#[derive(Debug, PartialOrd, Eq, PartialEq)]
pub struct AccountAttributes {
    id: String,
    osrank: Option<Osrank>,
}

#[derive(Debug, PartialOrd, PartialEq, Eq)]
pub enum Artifact {
    Project(ProjectAttributes),
    Account(AccountAttributes),
}

impl fmt::Display for Artifact {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            Artifact::Project(ref attrs) => write!(f, "{}", attrs.id),
            Artifact::Account(ref attrs) => write!(f, "{}", attrs.id),
        }
    }
}

/// The network graph from the paper, comprising of both accounts and projects.
#[derive(Debug)]
pub struct Network {
    pub from_graph: Graph<Artifact, Dependency, Directed>,
}

impl Network {
    pub fn new() -> Self {
        Network {
            from_graph: Graph::new(),
        }
    }

    /// Adds an Artifact to the Network.
    pub fn add_artifact(&mut self, artifact: Artifact) {
        let _ = self.from_graph.add_node(artifact);
        ()
    }

    /// Adds a Dependency to the Network. It's unsafe in the sense it's
    /// callers' responsibility to ensure that the source and target exist
    /// in the input Network.
    pub fn unsafe_add_dependency(&mut self, source: u32, target: u32, dependency: Dependency) {
        let _ =
            self.from_graph
                .add_edge(NodeIndex::from(source), NodeIndex::from(target), dependency);
        ()
    }
}
