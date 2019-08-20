#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate fraction;
extern crate num_traits;
extern crate petgraph;

use std::collections::HashMap;
use std::fmt;
use std::fs::OpenOptions;
use std::io::Write;
use std::ops::{Div, Mul, Rem};
use std::path::Path;

use crate::protocol_traits::graph::{
    EdgeReference, EdgeReferences, Graph, GraphObject, Id, Metadata, Nodes, NodesMut,
};
use fraction::{Fraction, GenericFraction};
use num_traits::{Num, One, Signed, Zero};
use petgraph::dot::{Config, Dot};
use petgraph::graph::{node_index, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Directed;

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

#[derive(Debug, Default, PartialEq, Eq)]
pub struct RandomWalks<Id> {
    random_walks_internal: Vec<RandomWalk<Id>>,
}

impl<Id> RandomWalks<Id>
where
    Id: PartialEq,
{
    pub fn new() -> Self {
        RandomWalks {
            random_walks_internal: Vec::new(),
        }
    }

    pub fn add_walk(&mut self, walk: RandomWalk<Id>) {
        self.random_walks_internal.push(walk);
    }

    pub fn len(&self) -> usize {
        self.random_walks_internal.len()
    }

    pub fn count_visits(&self, idx: Id) -> usize {
        self.random_walks_internal
            .iter()
            .map(|rw| rw.count_visits(&idx))
            .sum()
    }
}

#[derive(Debug, Default, PartialEq, Eq, Hash)]
pub struct RandomWalk<Id> {
    random_walk_internal: Vec<Id>,
}

impl<Id> RandomWalk<Id>
where
    Id: PartialEq,
{
    pub fn new() -> Self {
        RandomWalk {
            random_walk_internal: Vec::new(),
        }
    }

    pub fn add_next(&mut self, idx: Id) {
        self.random_walk_internal.push(idx);
    }

    pub fn count_visits(&self, idx: &Id) -> usize {
        self.random_walk_internal
            .iter()
            .filter(|i| i == &idx)
            .count()
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

#[derive(Debug)]
pub struct Dependency<Id, W> {
    id: Id,
    dependency_type: DependencyType<W>,
}

impl<Id, W> fmt::Display for Dependency<Id, W>
where
    W: fmt::Display,
    Id: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            f,
            "Dependency [ id = {}, type = {} ]",
            self.id, self.dependency_type
        )
    }
}

#[derive(Debug, PartialEq)]
pub enum DependencyType<W> {
    Contrib(W),
    ContribPrime(W),
    Maintain(W),
    MaintainPrime(W),
    Depend(W),
    /// This is not in the original whitepaper, but it's used as an edge
    /// metadata once we have normalised the edges and we have now only a
    /// single directed edge from A -> B
    Influence(W),
}

impl<W> fmt::Display for DependencyType<W>
where
    W: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            DependencyType::Contrib(ref w) => write!(f, "{:.5}", w),
            DependencyType::ContribPrime(ref w) => write!(f, "{:.5}", w),
            DependencyType::Maintain(ref w) => write!(f, "{:.5}", w),
            DependencyType::MaintainPrime(ref w) => write!(f, "{:.5}", w),
            DependencyType::Depend(ref w) => write!(f, "{:.5}", w),
            DependencyType::Influence(ref w) => write!(f, "{:.5}", w),
        }
    }
}

impl<W> DependencyType<W> {
    pub fn get_weight(&self) -> &W {
        match self {
            DependencyType::Contrib(ref w) => w,
            DependencyType::ContribPrime(ref w) => w,
            DependencyType::Maintain(ref w) => w,
            DependencyType::MaintainPrime(ref w) => w,
            DependencyType::Depend(ref w) => w,
            DependencyType::Influence(ref w) => w,
        }
    }
}

impl<DependencyId, W> GraphObject for Dependency<DependencyId, W> {
    type Id = DependencyId;
    type Metadata = DependencyType<W>;

    fn id(&self) -> &Self::Id {
        &self.id
    }

    fn get_metadata(&self) -> &Self::Metadata {
        &self.dependency_type
    }

    fn set_metadata(&mut self, v: Self::Metadata) {
        self.dependency_type = v;
    }
}

#[derive(Debug, PartialOrd, PartialEq, Eq)]
pub struct Artifact<Id> {
    id: Id,
    artifact_type: ArtifactType,
}

#[derive(Debug, PartialOrd, PartialEq, Eq)]
pub enum ArtifactType {
    Project { osrank: Osrank },
    Account { osrank: Osrank },
}

impl ArtifactType {
    pub fn set_osrank(&mut self, new: Osrank) {
        match self {
            ArtifactType::Project { ref mut osrank } => *osrank = new,
            ArtifactType::Account { ref mut osrank } => *osrank = new,
        }
    }
}

impl<ArtifactId> GraphObject for Artifact<ArtifactId> {
    type Id = ArtifactId;
    type Metadata = ArtifactType;

    fn id(&self) -> &Self::Id {
        &self.id
    }

    fn get_metadata(&self) -> &Self::Metadata {
        &self.artifact_type
    }

    fn set_metadata(&mut self, v: Self::Metadata) {
        self.artifact_type = v
    }
}

impl<Id> fmt::Display for Artifact<Id>
where
    Id: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self.artifact_type {
            ArtifactType::Project { osrank } => write!(f, "id: {} osrank: {:.5}", self.id, osrank),
            ArtifactType::Account { osrank } => write!(f, "id: {} osrank: {:.5}", self.id, osrank),
        }
    }
}

/// The network graph from the paper, comprising of both accounts and projects.
#[derive(Debug, Default)]
pub struct Network<W> {
    from_graph: petgraph::Graph<Artifact<String>, Dependency<usize, W>, Directed>,
    node_ids: HashMap<String, NodeIndex>,
    edge_ids: HashMap<usize, EdgeIndex>,
}

impl<W> Network<W>
where
    W: fmt::Display,
{
    /// Adds an Artifact to the Network.
    fn add_artifact(&mut self, id: String, artifact_type: ArtifactType) {
        let id_cloned = id.clone();
        let a = Artifact { id, artifact_type };
        let nid = self.from_graph.add_node(a);

        self.node_ids.insert(id_cloned, nid);
    }

    /// Adds a Dependency to the Network. It's unsafe in the sense it's
    /// callers' responsibility to ensure that the source and target exist
    /// in the input Network.
    fn unsafe_add_dependency(
        &mut self,
        source: usize,
        target: usize,
        id: usize,
        dependency_type: DependencyType<W>,
    ) {
        let d = Dependency {
            id,
            dependency_type,
        };
        let eid = self
            .from_graph
            .add_edge(node_index(source), node_index(target), d);
        self.edge_ids.insert(id, eid);
    }

    /// Debug-only function to render a Network into a Graphiz dot file.
    pub fn to_graphviz_dot(&self, output_path: &Path) -> Result<(), Box<std::io::Error>> {
        let mut dot_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output_path)?;
        dot_file.write_fmt(format_args!(
            "{}",
            Dot::with_config(&self.from_graph, &[Config::EdgeNoLabel])
        ))?;
        Ok(())
    }
}

impl<W> Graph for Network<W>
where
    W: Default + fmt::Display,
{
    type Node = Artifact<String>;
    type Edge = Dependency<usize, W>;

    fn add_node(&mut self, node_id: Id<Self::Node>, node_metadata: Metadata<Self::Node>) {
        self.add_artifact(node_id, node_metadata);
    }

    fn add_edge(
        &mut self,
        source: &Id<Self::Node>,
        target: &Id<Self::Node>,
        edge_id: Id<Self::Edge>,
        edge_metadata: Metadata<Self::Edge>,
    ) {
        //NOTE(adn) Should we return a `Result` rather than blowing everything up?
        match self
            .node_ids
            .get(source)
            .iter()
            .zip(self.node_ids.get(target).iter())
            .next()
        {
            Some((s, t)) => {
                let src = (*s).index();
                let tgt = (*t).index();
                self.unsafe_add_dependency(src, tgt, edge_id, edge_metadata);
            }
            None => panic!(
                "add_adge: invalid link, source {} or target {} are missing.",
                source, target
            ),
        }
    }

    fn neighbours(
        &self,
        node_id: &Id<Self::Node>,
    ) -> EdgeReferences<Id<Self::Node>, Id<Self::Edge>> {
        let mut neighbours = Vec::default();

        if let Some(nid) = self.node_ids.get(node_id) {
            for eref in self.from_graph.edges(*nid) {
                neighbours.push(EdgeReference {
                    source: self.from_graph[eref.source()].id(),
                    target: self.from_graph[eref.target()].id(),
                    id: self.from_graph[eref.id()].id(),
                })
            }
        }

        neighbours
    }

    fn nodes(&self) -> Nodes<Self::Node> {
        Nodes {
            range: 0..self.from_graph.node_count(),
            to_node_id: Box::new(move |i| &self.from_graph[node_index(i)]),
        }
    }

    fn nodes_mut(&mut self) -> NodesMut<Self::Node> {
        NodesMut {
            // NOTE(adn) Not the most efficient iterator, but due to my
            // limited Rust-fu, I couldn't find anything better.
            range: {
                let mut v = Vec::default();

                for n in self.from_graph.node_weights_mut() {
                    v.push(n);
                }

                v.into_iter()
            },
        }
    }

    fn edge_count(&self) -> usize {
        self.from_graph.edge_count()
    }

    fn lookup_node_metadata(&self, node_id: &Id<Self::Node>) -> Option<&Metadata<Self::Node>> {
        self.node_ids
            .get(node_id)
            .and_then(|n| Some(self.from_graph[*n].get_metadata()))
    }
    fn lookup_edge_metadata(&self, edge_id: &Id<Self::Edge>) -> Option<&Metadata<Self::Edge>> {
        self.edge_ids
            .get(edge_id)
            .and_then(|e| Some(self.from_graph[*e].get_metadata()))
    }

    fn set_node_metadata(&mut self, node_id: &Id<Self::Node>, new: Metadata<Self::Node>) {
        if let Some(nid) = self.node_ids.get(node_id) {
            self.from_graph[*nid].set_metadata(new)
        }
    }

    fn set_edge_metadata(&mut self, edge_id: &Id<Self::Edge>, new: Metadata<Self::Edge>) {
        if let Some(eid) = self.edge_ids.get(edge_id) {
            self.from_graph[*eid].set_metadata(new)
        }
    }

    fn subgraph_by_nodes(
        &self,
        sub_nodes: Vec<&Artifact<String>>
    ) -> Self
    {
        // TODO filter_map might not keep the same index if nodes are removed. An alternative to subgraph could
        // be to add a metadata to the graph `trusted` and only use `trusted_neighbours` in phase 2
        let sub_graph = self.from_graph.filter_map(
            |id, node| if sub_nodes.iter().any(|sn| sn.id() == node.id()) {Some(node.clone())} else {None},
            |id, edge| Some(edge.clone())
        );
        let mut sub_net = Network::default();
        sub_net.from_graph = sub_graph;
        sub_net
    }
}

/// Trait to print (parts of) the for debugging purposes
pub trait PrintableGraph<'a>: Graph {
    /// Prints all nodes and their attributes
    fn print_nodes(&self);
}

impl<'a, W> PrintableGraph<'a> for Network<W>
where
    W: Default + fmt::Display,
{
    fn print_nodes(&self) {
        for arti in self.from_graph.raw_nodes().iter().map(|node| &node.weight) {
            println!("{}", arti);
        }
    }
}
