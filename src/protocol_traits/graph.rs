#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate rand;

use std::ops::Range;

use crate::protocol_traits::storage::KeyValueStorage;
use rand::SeedableRng;

/// The notion of a generic `GraphObject` in a `Graph`, which can be
/// either a node or an edge.
pub trait GraphObject {
    /// The unique identifier for this object.
    type Id;
    /// The metadata attached to this object.
    type Metadata;

    fn id(&self) -> &Self::Id;

    fn get_metadata(&self) -> &Self::Metadata;
    fn set_metadata(&mut self, v: Self::Metadata);
}

/// Useful type alias for a `GraphObject`'s `Id`.
pub type Id<N> = <N as GraphObject>::Id;

/// Useful type alias for a `GraphObject`'s `Metadata`.
pub type Metadata<N> = <N as GraphObject>::Metadata;

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
    G: Graph,
{
    /// We will likely need some state to generate the seed from, but
    /// this is not spec-ed out yet.
    type SomeState;

    /// A type representing unique identifier within the system.
    type Id;

    /// The storage this `GraphAPI` will query.
    type Storage: KeyValueStorage<Key = Self::Id, Value = G>;

    /// Adds a new node to a particular layer, given its metadata.
    fn add_node(&mut self, layer_id: &Self::Id, node_metadata: Metadata<G::Node>);

    /// Adds a new edge to a particular layer, given its metadata, the source
    /// and the target.
    fn add_edge(
        &mut self,
        layer_id: &Self::Id,
        source: &Self::Id,
        target: &Self::Id,
        edge_metadata: Metadata<G::Edge>,
    );

    /// Returns the neighbours for a node, given its `NodeId` and the
    /// _layer_ its located.
    fn neighbours(
        &self,
        layer_id: &Self::Id,
        node: &Metadata<G::Node>,
    ) -> EdgeReferences<Self::Id, Self::Id>;

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
    ) -> Option<&Metadata<G::Node>>;

    /// Lookups the _metadata_ for an edge in a layer, if any.
    fn lookup_edge_metadata(
        &self,
        layer_id: &Self::Id,
        edge_id: &Self::Id,
    ) -> Option<&Metadata<G::Edge>>;

    /// Replaces the _metadata_ associated to the input `NodeId`, given the
    /// layer where its located.
    fn set_node_metadata(&mut self, layer_id: &Self::Id, node_id: &Self::Id);

    /// Replaces the _metadata_ associated to the input `EdgeId`, given the
    /// layer where its located.
    fn set_edge_metadata(&mut self, layer_id: &Self::Id, edge_id: &Self::Id);
}

pub trait Graph: Default {
    /// The "userland" `Id` associated to a `Node`. It allows users and third
    /// party apps to assign a `Node` any arbitrary Id while being able to query
    /// the `Graph` with such `Id`.
    type Node: GraphObject;
    type Edge: GraphObject;

    /// Adds a new node to the graph, given its metadata.
    fn add_node(&mut self, node_id: Id<Self::Node>, node_metadata: Metadata<Self::Node>);

    /// Adds a new edge to the graph, given its metadata, the source
    /// and the target.
    fn add_edge(
        &mut self,
        source: &Id<Self::Node>,
        target: &Id<Self::Node>,
        edge_id: Id<Self::Edge>,
        edge_metadata: Metadata<Self::Edge>,
    );

    /// Returns the neighbours for a node, given its `NodeId`.
    fn neighbours(
        &self,
        node_id: &Id<Self::Node>,
    ) -> EdgeReferences<Id<Self::Node>, Id<Self::Edge>>;

    /// Lookups the _metadata_ for a node, if any.
    fn lookup_node_metadata(&self, node_id: &Id<Self::Node>) -> Option<&Metadata<Self::Node>>;

    /// Lookups the _metadata_ for an edge, if any.
    fn lookup_edge_metadata(&self, edge_id: &Id<Self::Edge>) -> Option<&Metadata<Self::Edge>>;

    /// Returns an iterator over the node ids in the `Graph`.
    fn nodes(&self) -> Nodes<Self::Node>;

    fn nodes_mut(&mut self) -> NodesMut<Self::Node>;

    /// Returns the number of edges for this `Graph`.
    fn edge_count(&self) -> usize;

    /// Replaces the _metadata_ associated to the input `NodeId`.
    fn set_node_metadata(&mut self, node_id: &Id<Self::Node>, new: Metadata<Self::Node>);

    /// Replaces the _metadata_ associated to the input `EdgeId`.
    fn set_edge_metadata(&mut self, edge_id: &Id<Self::Edge>, new: Metadata<Self::Edge>);

    /// Creates a subgraph of on the nodes of `sub_nodes` of `self`.
    fn subgraph_by_nodes(&self, sub_nodes: Vec<&Self::Node>) -> Self;
}

pub struct EdgeReference<'a, NodeId, EdgeId> {
    pub source: &'a NodeId,
    pub target: &'a NodeId,
    pub id: &'a EdgeId,
}

pub type EdgeReferences<'a, N, E> = Vec<EdgeReference<'a, N, E>>;

pub struct Nodes<'a, N: 'a> {
    pub range: Range<usize>,
    pub to_node_id: Box<(Fn(usize) -> &'a N) + 'a>,
}

pub struct NodesMut<'a, N: 'a> {
    pub range: std::vec::IntoIter<&'a mut N>,
    // pub to_node_id: Box<(FnMut(usize) -> &'a mut N) + 'a>,
}

impl<'a, N: 'a> Iterator for Nodes<'a, N> {
    type Item = &'a N;

    fn next(&mut self) -> Option<Self::Item> {
        let f = &self.to_node_id;

        match self.range.next() {
            None => None,
            Some(v) => Some(f(v)),
        }
    }
}

impl<'a, N: 'a> Iterator for NodesMut<'a, N> {
    type Item = &'a mut N;

    fn next(&mut self) -> Option<Self::Item> {
        self.range.next()
    }
}
