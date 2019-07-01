extern crate fraction;
extern crate petgraph;

use fraction::{Fraction, GenericFraction};
use petgraph::graph::NodeIndex;
use petgraph::{Directed, Graph};

type Osrank = Fraction;
type HyperParam = GenericFraction<u8>;

#[derive(Debug)]
pub struct Weight(f32);

/// The hyperparams from the paper, which are used to weight the edges.
struct HyperParams {
    contrib_factor: HyperParam,
    contrib_prime_factor: HyperParam,
    depend_factor: HyperParam,
    maintain_factor: HyperParam,
    maintain_prime_factor: HyperParam,
}

/// A default implementation based on the values from the paper.
impl Default for HyperParams {
    fn default() -> Self {
        HyperParams {
            contrib_factor: HyperParam::new(1, 7),
            contrib_prime_factor: HyperParam::new(2, 5),
            depend_factor: HyperParam::new(4, 7),
            maintain_factor: HyperParam::new(2, 7),
            maintain_prime_factor: HyperParam::new(3, 5),
        }
    }
}

#[derive(Debug)]
enum Dependency {
    Contrib(Weight),
    ContribPrime(Weight),
    Maintain(Weight),
    MaintainPrime(Weight),
    Depend(Weight),
}

#[derive(Debug, PartialEq, Eq, PartialOrd)]
struct ProjectAttributes {
    id: String,
    osrank: Option<Osrank>,
}

#[derive(Debug, PartialOrd, Eq, PartialEq)]
struct AccountAttributes {
    id: String,
    osrank: Option<Osrank>,
}

#[derive(Debug, PartialOrd, PartialEq, Eq)]
enum Artifact {
    Project(ProjectAttributes),
    Account(AccountAttributes),
}

/// The network graph from the paper, comprising of both accounts and projects.
#[derive(Debug)]
pub struct Network {
    from_graph: Graph<Artifact, Dependency, Directed>,
}

impl Network {
    fn new() -> Self {
        Network {
            from_graph: Graph::new(),
        }
    }

    /// Adds an Artifact to the Network.
    fn add_artifact(&mut self, artifact: Artifact) {
        let _ = self.from_graph.add_node(artifact);
        ()
    }

    /// Adds a Dependency to the Network. It's unsafe in the sense it's
    /// callers' responsibility to ensure that the source and target exist
    /// in the input Network.
    fn unsafe_add_dependency(
        &mut self,
        source: NodeIndex,
        target: NodeIndex,
        dependency: Dependency,
    ) {
        let _ = self.from_graph.add_edge(source, target, dependency);
        ()
    }
}
