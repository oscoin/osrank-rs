#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate fraction;
extern crate num_traits;
extern crate petgraph;
extern crate quickcheck;

use num_traits::Zero;
use std::collections::HashMap;
use std::fmt;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use super::Osrank;
use crate::protocol_traits::graph::GraphExtras;
use oscoin_graph_api::{
    Data, Edge, EdgeReference, EdgeReferences, Edges, Graph, GraphDataWriter, GraphObject,
    GraphWriter, Id, Node, Nodes, NodesMut,
};
use petgraph::dot::{Config, Dot};
use petgraph::graph::{node_index, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Directed;

use crate::util::quickcheck::frequency;
use quickcheck::{Arbitrary, Gen};

#[derive(Debug, Clone)]
pub struct Dependency<Id: Clone, W: Clone> {
    id: Id,
    dependency_type: DependencyType<W>,
}

impl<Id, W> fmt::Display for Dependency<Id, W>
where
    W: fmt::Display + Clone,
    Id: fmt::Display + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            f,
            "Dependency [ id = {}, type = {} ]",
            self.id, self.dependency_type
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DependencyType<W: Clone> {
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
    W: fmt::Display + Clone,
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

impl<W> DependencyType<W>
where
    W: Clone,
{
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

impl<DependencyId, W> GraphObject for Dependency<DependencyId, W>
where
    DependencyId: Clone,
    W: Clone,
{
    type Id = DependencyId;
    type Data = DependencyType<W>;

    fn id(&self) -> &Self::Id {
        &self.id
    }

    fn data(&self) -> &Self::Data {
        &self.dependency_type
    }

    fn data_mut(&mut self) -> &mut Self::Data {
        &mut self.dependency_type
    }
}

impl<DependencyId, W> Edge<W, DependencyType<W>> for Dependency<DependencyId, W>
where
    DependencyId: Clone,
    W: Clone,
{
    fn weight(&self) -> W {
        self.dependency_type.get_weight().clone()
    }
}

#[derive(Clone, Debug, PartialOrd, PartialEq, Eq)]
pub struct Artifact<Id: Clone> {
    id: Id,
    artifact_type: ArtifactType,
}

impl<Id> Artifact<Id>
where
    Id: Clone,
{
    pub fn new_account(id: Id) -> Self {
        Artifact {
            id,
            artifact_type: ArtifactType::Account {
                osrank: Zero::zero(),
            },
        }
    }

    pub fn new_project(id: Id) -> Self {
        Artifact {
            id,
            artifact_type: ArtifactType::Project {
                osrank: Zero::zero(),
            },
        }
    }
}

#[derive(Clone, Debug, PartialOrd, PartialEq, Eq)]
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

    pub fn get_osrank(&self) -> Osrank {
        match self {
            ArtifactType::Project { osrank } => *osrank,
            ArtifactType::Account { osrank } => *osrank,
        }
    }
}

impl Arbitrary for ArtifactType {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let choices = vec![
            (
                50,
                ArtifactType::Project {
                    osrank: Osrank::new(g.next_u64(), 1u8),
                },
            ),
            (
                50,
                ArtifactType::Account {
                    osrank: Osrank::new(g.next_u64(), 1u8),
                },
            ),
        ];
        frequency(g, choices)
    }
}

impl<ArtifactId> GraphObject for Artifact<ArtifactId>
where
    ArtifactId: Clone,
{
    type Id = ArtifactId;
    type Data = ArtifactType;

    fn id(&self) -> &Self::Id {
        &self.id
    }

    fn data(&self) -> &Self::Data {
        &self.artifact_type
    }

    fn data_mut(&mut self) -> &mut Self::Data {
        &mut self.artifact_type
    }
}

impl<ArtifactId> Node<ArtifactType> for Artifact<ArtifactId> where ArtifactId: Clone {}

impl<Id> fmt::Display for Artifact<Id>
where
    Id: fmt::Display + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self.artifact_type {
            ArtifactType::Project { osrank } => write!(f, "id: {} osrank: {:.5}", self.id, osrank),
            ArtifactType::Account { osrank } => write!(f, "id: {} osrank: {:.5}", self.id, osrank),
        }
    }
}

impl<ArtifactId> Arbitrary for Artifact<ArtifactId>
where
    ArtifactId: Arbitrary,
{
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let id: ArtifactId = Arbitrary::arbitrary(g);
        let choices = vec![
            (50, Artifact::new_account(id.clone())),
            (50, Artifact::new_project(id)),
        ];
        frequency(g, choices)
    }
}

/// The network graph from the paper, comprising of both accounts and projects.
#[derive(Clone, Debug, Default)]
pub struct Network<W: Clone> {
    from_graph: petgraph::Graph<Artifact<String>, Dependency<usize, W>, Directed>,
    node_ids: HashMap<String, NodeIndex>,
    edge_ids: HashMap<usize, EdgeIndex>,
}

impl<W> Network<W>
where
    W: fmt::Display + Clone,
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

    pub fn is_empty(&self) -> bool {
        self.from_graph.node_count() == 0
            && self.from_graph.edge_count() == 0
            && self.node_ids.is_empty()
            && self.edge_ids.is_empty()
    }
}

impl<W> GraphWriter for Network<W>
where
    W: Default + fmt::Display + Clone,
{
    fn add_node(&mut self, node_id: Id<Self::Node>, node_metadata: Data<Self::Node>) {
        self.add_artifact(node_id, node_metadata);
    }

    fn remove_node(&mut self, node_id: Id<Self::Node>) {
        panic!("Remove node not yet implemented.")
    }

    fn add_edge(
        &mut self,
        edge_id: Id<Self::Edge>,
        source: &Id<Self::Node>,
        target: &Id<Self::Node>,
        weight: Self::Weight,
        edge_metadata: Data<Self::Edge>,
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

    fn remove_edge(&mut self, edge_id: Id<Self::Edge>) {
        panic!("remove_edge not yet implemented")
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
}

impl<W> Graph for Network<W>
where
    W: Default + fmt::Display + Clone,
{
    type Node = Artifact<String>;
    type Edge = Dependency<usize, W>;
    type Weight = W;
    type NodeData = ArtifactType;
    type EdgeData = DependencyType<W>;

    fn neighbors(
        &self,
        node_id: &Id<Self::Node>,
    ) -> EdgeReferences<Id<Self::Node>, Id<Self::Edge>> {
        let mut neighbors = Vec::default();

        if let Some(nid) = self.node_ids.get(node_id) {
            for eref in self.from_graph.edges(*nid) {
                neighbors.push(EdgeReference {
                    source: self.from_graph[eref.source()].id(),
                    target: self.from_graph[eref.target()].id(),
                    id: self.from_graph[eref.id()].id(),
                })
            }
        }

        neighbors
    }

    fn get_node(&self, id: &Id<Self::Node>) -> Option<&Self::Node> {
        self.node_ids
            .get(id)
            .and_then(|nid| Some(&self.from_graph[*nid]))
    }

    fn get_edge(&self, id: &Id<Self::Edge>) -> Option<&Self::Edge> {
        self.edge_ids
            .get(id)
            .and_then(|eid| Some(&self.from_graph[*eid]))
    }

    fn nodes(&self) -> Nodes<Self::Node> {
        Nodes {
            range: self
                .from_graph
                .raw_nodes()
                .into_iter()
                .map(|n| &n.weight)
                .collect::<Vec<&Self::Node>>()
                .into_iter(),
        }
    }

    fn edges(&self, node: Id<Self::Node>) -> Edges<Self::Edge> {
        panic!("edges not yet implemented.")
    }
}

impl<W> GraphDataWriter for Network<W>
where
    W: Default + fmt::Display + Clone,
{
    fn edge_data_mut(&mut self, id: &Id<Self::Edge>) -> Option<&mut Data<Self::Edge>> {
        let mb_id = self.edge_ids.get(id);
        match mb_id {
            None => None,
            Some(eid) => Some(self.from_graph[*eid].data_mut()),
        }
    }

    fn node_data_mut(&mut self, id: &Id<Self::Node>) -> Option<&mut Data<Self::Node>> {
        let mb_id = self.node_ids.get(id);
        match mb_id {
            None => None,
            Some(nid) => Some(self.from_graph[*nid].data_mut()),
        }
    }
}

impl<W> GraphExtras for Network<W>
where
    W: Default + fmt::Display + Clone,
{
    fn edge_count(&self) -> usize {
        self.from_graph.edge_count()
    }

    fn node_count(&self) -> usize {
        self.from_graph.node_count()
    }

    fn lookup_node_metadata(&self, node_id: &Id<Self::Node>) -> Option<&Data<Self::Node>> {
        self.node_ids
            .get(node_id)
            .and_then(|n| Some(self.from_graph[*n].data()))
    }
    fn lookup_edge_metadata(&self, edge_id: &Id<Self::Edge>) -> Option<&Data<Self::Edge>> {
        self.edge_ids
            .get(edge_id)
            .and_then(|e| Some(self.from_graph[*e].data()))
    }

    fn subgraph_by_nodes(&self, sub_nodes: Vec<&String>) -> Self {
        let mut sub_network = Network::default();

        for graph_node_id in &sub_nodes {
            // Add the node only if `graph_node_id` exists.
            if let Some(petgraph_node_id) = &self.node_ids.get(*graph_node_id) {
                let node = &self.from_graph[**petgraph_node_id].clone();
                sub_network.add_node(node.id().to_string(), node.data().clone());
            }
        }

        // Once we have added all the nodes, we can now add all the edges.
        for graph_node_id in sub_nodes {
            for graph_edge_ref in self.neighbors(graph_node_id) {
                let graph_edge_target = graph_edge_ref.target.clone();
                let graph_edge_id = graph_edge_ref.id;
                let petgraph_edge_id = &self.edge_ids[&graph_edge_id];
                let edge_object = &self.from_graph[*petgraph_edge_id];

                // Due to the fact not all edges might start or end in a node which
                // exist, we have to explictly check.
                if sub_network.node_ids.get(&graph_edge_target).is_some() {
                    sub_network.add_edge(
                        graph_edge_id.clone(),
                        graph_node_id,
                        &graph_edge_target,
                        edge_object.weight().clone(),
                        edge_object.data().clone(),
                    );
                }
            }
        }

        sub_network
    }
}

/// Trait to print (parts of) the for debugging purposes
pub trait PrintableGraph<'a>: Graph {
    /// Prints all nodes and their attributes
    fn print_nodes(&self);
}

impl<'a, W> PrintableGraph<'a> for Network<W>
where
    W: Default + fmt::Display + Clone,
{
    fn print_nodes(&self) {
        for arti in self.from_graph.raw_nodes().iter().map(|node| &node.weight) {
            println!("{}", arti);
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::types::Weight;

    fn network_fixture() -> Network<f64> {
        let mut network = Network::default();

        for node in &["p1", "p2", "p3"] {
            let a = Artifact::new_project(node.to_string());
            network.add_node(a.id().clone(), a.data().clone());
        }

        let edges = [
            ("p1", "p2", Weight::new(1, 1)),
            ("p1", "p2", Weight::new(4, 7)),
            ("p3", "p1", Weight::new(2, 7)),
            ("p3", "p2", Weight::new(2, 7)),
        ];

        for edge in &edges {
            network.add_edge(
                2,
                &edge.0.to_string(),
                &edge.1.to_string(),
                edge.2.as_f64().unwrap(),
                DependencyType::Influence(edge.2.as_f64().unwrap()),
            )
        }

        network
    }

    #[test]
    fn subgraph_by_nodes_empty_subnodes() {
        let graph = network_fixture();
        let subgraph = graph.subgraph_by_nodes(vec![]);
        assert_eq!(subgraph.is_empty(), true);
    }

    #[test]
    fn subgraph_by_nodes_single_node() {
        let graph = network_fixture();
        let subgraph = graph.subgraph_by_nodes(vec![&"p1".to_string()]);
        assert_eq!(subgraph.is_empty(), false);
    }

    #[test]
    fn subgraph_by_nodes_not_existent_node() {
        let graph = network_fixture();
        let subgraph = graph.subgraph_by_nodes(vec![&"bar".to_string()]);
        assert_eq!(subgraph.is_empty(), true);
    }

    // Tests that setting & getting an `Artifact`'s metadata roundtrips.

    #[quickcheck]
    fn artifact_get_set_metadata_roundtrip(meta: ArtifactType) {
        let mut a = Artifact::new_account("foo");
        *a.data_mut() = meta.clone();
        assert_eq!(*a.data(), meta);
    }

}
