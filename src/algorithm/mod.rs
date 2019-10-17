/// Incremental Osrank algorithm.
pub mod incremental;
/// Naive Osrank algorithm.
pub mod naive;

use crate::protocol_traits::graph::GraphExtras;
use oscoin_graph_api::{
    Data, Direction, EdgeRefs, Edges, Graph, GraphDataReader, GraphDataWriter, GraphWriter, Id,
    Nodes, NodesMut,
};
use quickcheck::{Arbitrary, Gen};

/// Shared types between algorithms.

/// Newtype-wapper that forces API producers and consumers to work with a
/// normalised Graph, i.e. a `Graph` which edges have been collapsed into a
/// single one.
///
/// You cannot "deconstruct" a `Normalised` graph. You can only create a new
/// one.
#[derive(Debug, Clone)]
pub struct Normalised<G> {
    normalised_graph: G,
}

/// NOTE: Do *not* implement a setter here (or make normalised_graph `pub`).
/// The whole point of this newtype writter is to prevent using the incremental
/// algorithm on a normalised graph, and viceversa use the naive one on an
/// "un-normalised" one.
impl<G> Normalised<G>
where
    G: Graph,
{
    pub fn new(normalised_graph: G) -> Self {
        Normalised { normalised_graph }
    }
}

/// Conceptually the companion for the `Normalised` newtype, this trait ensure
/// that certain `GraphAlgorithm` implementation can be given only in terms
/// of normalised graphs. This is the case for the "naive" algorithm, which
/// requires a normalised input to be fed as input. However, nothing in the
/// type signature of `osrank_naive` would prevent upstream code to call it
/// with *any* type, provided it implement the `Graph` trait. This is why we
/// need the `NormalisedGraph` trait: This embeds evidence in the type signature
/// that a graph must be normalised, first.
pub trait NormalisedGraph: private::NormalisedGraphSealed {}

impl<G> NormalisedGraph for Normalised<G> where G: Graph {}

impl<G> Default for Normalised<G>
where
    G: Default,
{
    fn default() -> Self {
        Normalised {
            normalised_graph: G::default(),
        }
    }
}

impl<A> Arbitrary for Normalised<A>
where
    A: Arbitrary,
{
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Normalised {
            normalised_graph: A::arbitrary(g),
        }
    }
}

impl<G> GraphDataWriter for Normalised<G>
where
    G: GraphDataWriter,
{
    fn edge_data_mut(&mut self, id: &Id<Self::Edge>) -> Option<&mut Data<Self::Edge>> {
        self.normalised_graph.edge_data_mut(id)
    }

    fn node_data_mut(&mut self, id: &Id<Self::Node>) -> Option<&mut Data<Self::Node>> {
        self.normalised_graph.node_data_mut(id)
    }
}

impl<G> GraphWriter for Normalised<G>
where
    G: GraphWriter,
{
    fn add_node(&mut self, node_id: Id<Self::Node>, node_metadata: Data<Self::Node>) {
        self.normalised_graph.add_node(node_id, node_metadata)
    }

    fn remove_node(&mut self, node_id: Id<Self::Node>) {
        self.normalised_graph.remove_node(node_id)
    }

    fn add_edge(
        &mut self,
        edge_id: Id<Self::Edge>,
        source: &Id<Self::Node>,
        to: &Id<Self::Node>,
        edge_metadata: Data<Self::Edge>,
    ) {
        self.normalised_graph
            .add_edge(edge_id, source, to, edge_metadata)
    }

    fn remove_edge(&mut self, edge_id: Id<Self::Edge>) {
        self.normalised_graph.remove_edge(edge_id)
    }

    fn nodes_mut(&mut self) -> NodesMut<Self::Node> {
        self.normalised_graph.nodes_mut()
    }
}

impl<G> Graph for Normalised<G>
where
    G: Graph,
{
    type Node = G::Node;
    type Edge = G::Edge;
    type Weight = G::Weight;
    type NodeData = G::NodeData;
    type EdgeData = G::EdgeData;

    fn neighbors(&self, node_id: &Id<Self::Node>) -> Nodes<Self::Node> {
        self.normalised_graph.neighbors(node_id)
    }

    fn get_node(&self, id: &Id<Self::Node>) -> Option<&Self::Node> {
        self.normalised_graph.get_node(id)
    }

    fn get_edge(&self, id: &Id<Self::Edge>) -> Option<&Self::Edge> {
        self.normalised_graph.get_edge(id)
    }

    fn nodes(&self) -> Nodes<Self::Node> {
        self.normalised_graph.nodes()
    }

    fn edges(&self, node: &Id<Self::Node>) -> Edges<Self::Edge> {
        self.normalised_graph.edges(node)
    }

    fn edges_directed(
        &self,
        node_id: &Id<Self::Node>,
        dir: Direction,
    ) -> EdgeRefs<Id<Self::Node>, Id<Self::Edge>> {
        self.normalised_graph.edges_directed(node_id, dir)
    }
}

impl<G> GraphDataReader for Normalised<G>
where
    G: GraphDataReader,
{
    fn node_data(&self, node_id: &Id<Self::Node>) -> Option<&Data<Self::Node>> {
        self.normalised_graph.node_data(node_id)
    }

    fn edge_data(&self, edge_id: &Id<Self::Edge>) -> Option<&Data<Self::Edge>> {
        self.normalised_graph.edge_data(edge_id)
    }
}

impl<G> GraphExtras for Normalised<G>
where
    G: GraphExtras,
{
    fn edge_count(&self) -> usize {
        self.normalised_graph.edge_count()
    }

    fn node_count(&self) -> usize {
        self.normalised_graph.node_count()
    }

    fn subgraph_by_nodes(&self, sub_nodes: Vec<&Id<Self::Node>>) -> Self {
        Normalised {
            normalised_graph: self.normalised_graph.subgraph_by_nodes(sub_nodes),
        }
    }
}

#[derive(Debug, Display, PartialEq, Eq)]
/// Errors that the `osrank` algorithm might throw.
pub enum OsrankError {
    /// Generic, catch-all error for things which can go wrong during the
    /// algorithm.
    UnknownError,
    RngFailedToSplit(String),
}

impl From<rand::Error> for OsrankError {
    fn from(err: rand::Error) -> OsrankError {
        OsrankError::RngFailedToSplit(format!("{}", err))
    }
}

// See: https://rust-lang-nursery.github.io/api-guidelines/future-proofing.html
// We want to avoid crate users to deliberately define this trait, which should
// be fully controlled by the `osrank` crate.
mod private {
    use super::Normalised;
    use oscoin_graph_api::Graph;
    pub trait NormalisedGraphSealed {}
    impl<G> NormalisedGraphSealed for Normalised<G> where G: Graph {}
}
