#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate oscoin_graph_api as oscoin;
extern crate rand;

/// This is a compatibility-shim trait for things that the "official"
/// GraphAPI trait(s) don't give us for free.
pub trait GraphExtras: oscoin::Graph + oscoin::GraphDataWriter + oscoin::GraphWriter {
    fn lookup_node_metadata(
        &self,
        node_id: &oscoin::Id<Self::Node>,
    ) -> Option<&oscoin::Data<Self::Node>>;

    /// Lookups the _metadata_ for an edge in a layer, if any.
    fn lookup_edge_metadata(
        &self,
        edge_id: &oscoin::Id<Self::Edge>,
    ) -> Option<&oscoin::Data<Self::Edge>>;

    /// Returns the number of edges for this `Graph`.
    fn edge_count(&self) -> usize;

    /// Returns the number of nodes for this `Graph`.
    fn node_count(&self) -> usize;

    /// Creates a subgraph of on the nodes of `sub_nodes` of `self`.
    fn subgraph_by_nodes(&self, sub_nodes: Vec<&oscoin::Id<Self::Node>>) -> Self;
}
