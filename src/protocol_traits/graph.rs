#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate rand;

use std::ops::Range;

use crate::protocol_traits::storage::KeyValueStorage;
use rand::SeedableRng;

/// An trait abstracting over a `GraphAPI`.
///
/// This trait is developed here as a quick proof-of-concept to break any
/// dependency from concrete implementations in the Osrank algorithm, but
/// it's likely it will be re-developed from the _Graph API Working Group_
/// potentially based (or not) from this initial sketch.
///
/// The `GraphAPI` offers functions to view and manipulate _graphs_.
pub trait GraphAPI<G, R>
where
    R: SeedableRng,
    G: Graph<NodeMetadata = Self::NodeMetadata, EdgeMetadata = Self::EdgeMetadata>,
{
    /// We will likely need some state to generate the seed from, but
    /// this is not spec-ed out yet.
    type SomeState;

    /// Metadata a node might have.
    type NodeMetadata;

    /// Metadata an edge might have.
    type EdgeMetadata;

    /// A type representing unique identifier within the system.
    type Id;

    /// The storage this `GraphAPI` will query.
    type Storage: KeyValueStorage<Key = Self::Id, Value = G>;

    /// Adds a new node to a particular layer, given its metadata.
    fn add_node(&mut self, layer_id: &Self::Id, node_metadata: Self::NodeMetadata);

    /// Adds a new edge to a particular layer, given its metadata, the source
    /// and the target.
    fn add_edge(
        &mut self,
        layer_id: &Self::Id,
        source: &Self::Id,
        target: &Self::Id,
        edge_metadata: Self::EdgeMetadata,
    );

    /// Returns the neighbours for a node, given its `NodeId` and the
    /// _layer_ its located.
    fn neighbours(
        &self,
        layer_id: &Self::Id,
        node: &G::NodeId,
    ) -> Vec<EdgeReference<Self::Id, Self::Id>>;

    /// Given an initial `SomeState`' returns a random-yet-predicable seed
    /// to be used by the _Osrank_ algorithm.
    fn seed(&self, some_state: Self::SomeState) -> R::Seed;

    /// Lookups a `Graph` given its `Id`.
    fn lookup_graph(&self, layer_id: &Self::Id) -> Option<&G>;

    /// Lookups the _metadata_ for a node in a layer, if any.
    fn lookup_node_metadata(
        &self,
        layer_id: &Self::Id,
        node_id: &Self::Id,
    ) -> Option<Self::NodeMetadata>;

    /// Lookups the _metadata_ for an edge in a layer, if any.
    fn lookup_edge_metadata(
        &self,
        layer_id: &Self::Id,
        edge_id: &Self::Id,
    ) -> Option<Self::EdgeMetadata>;

    /// Replaces the _metadata_ associated to the input `NodeId`, given the
    /// layer where its located.
    fn set_node_metadata(&mut self, layer_id: &Self::Id, node_id: &Self::Id);

    /// Replaces the _metadata_ associated to the input `EdgeId`, given the
    /// layer where its located.
    fn set_edge_metadata(&mut self, layer_id: &Self::Id, edge_id: &Self::Id);
}

pub trait Graph: Default {
    type NodeId: Copy;
    type EdgeId: Copy;
    type NodeMetadata;
    type EdgeMetadata;

    /// Adds a new node to the graph, given its metadata.
    fn add_node(&mut self, node_metadata: Self::NodeMetadata);

    /// Adds a new edge to the graph, given its metadata, the source
    /// and the target.
    fn add_edge(
        &mut self,
        source: Self::NodeId,
        target: Self::NodeId,
        edge_metadata: Self::EdgeMetadata,
    );

    /// Returns the neighbours for a node, given its `NodeId`.
    fn neighbours(&self, node_id: &Self::NodeId) -> Vec<EdgeReference<Self::NodeId, Self::EdgeId>>;

    /// Lookups the _metadata_ for a node, if any.
    fn lookup_node_metadata(&self, node_id: &Self::NodeId) -> Option<&Self::NodeMetadata>;

    /// Lookups the _metadata_ for an edge, if any.
    fn lookup_edge_metadata(&self, edge_id: &Self::EdgeId) -> Option<&Self::EdgeMetadata>;

    /// Returns an iterator over the node ids in the `Graph`.
    fn node_ids(&self) -> NodeIds<Self::NodeId>;

    /// Returns the number of edges for this `Graph`.
    fn edge_count(&self) -> usize;

    /// Replaces the _metadata_ associated to the input `NodeId`.
    fn set_node_metadata(&mut self, node_id: &Self::NodeId, new: Self::NodeMetadata);

    /// Replaces the _metadata_ associated to the input `EdgeId`.
    fn update_node_metadata(
        &mut self,
        node_id: &Self::NodeId,
        update_fn: Box<FnMut(&mut Self::NodeMetadata) -> ()>,
    );

    /// Replaces the _metadata_ associated to the input `EdgeId`.
    fn set_edge_metadata(&mut self, edge_id: &Self::EdgeId, new: Self::EdgeMetadata);
}

pub struct EdgeReference<NodeId, EdgeId> {
    pub source: NodeId,
    pub target: NodeId,
    pub id: EdgeId,
}

pub struct NodeIds<I> {
    pub range: Range<usize>,
    pub to_node_id: Box<Fn(usize) -> I>,
}

impl<I> Iterator for NodeIds<I> {
    type Item = I;

    fn next(&mut self) -> Option<I> {
        let f = &self.to_node_id;

        match self.range.next() {
            None => None,
            Some(v) => Some(f(v)),
        }
    }
}
