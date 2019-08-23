#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate fraction;
extern crate num_traits;
extern crate petgraph;

use num_traits::Zero;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use super::Osrank;
use crate::protocol_traits::graph::{
    EdgeReference, EdgeReferences, Graph, GraphObject, Id, Metadata, Nodes, NodesMut,
};
use petgraph::dot::{Config, Dot};
use petgraph::graph::{node_index, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Directed;

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
}

impl<ArtifactId> GraphObject for Artifact<ArtifactId>
where
    ArtifactId: Clone,
{
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
    Id: fmt::Display + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self.artifact_type {
            ArtifactType::Project { osrank } => write!(f, "id: {} osrank: {:.5}", self.id, osrank),
            ArtifactType::Account { osrank } => write!(f, "id: {} osrank: {:.5}", self.id, osrank),
        }
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
}

impl<W> Graph for Network<W>
where
    W: Default + fmt::Display + Clone,
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

    fn node_count(&self) -> usize {
        self.from_graph.node_count()
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

    fn subgraph_by_nodes(&self, sub_nodes: Vec<&String>) -> Self {
        let mut sub_network = Network::default();

        for graph_node_id in &sub_nodes {
            let petgraph_node_id = &self.node_ids[*graph_node_id];
            let node = &self.from_graph[*petgraph_node_id].clone();

            sub_network.add_node(node.id().to_string(), node.get_metadata().clone());
        }

        // Once we have added all the nodes, we can now add all the edges
        for graph_node_id in sub_nodes {
            for graph_edge_ref in self.neighbours(graph_node_id) {
                let graph_edge_target = graph_edge_ref.target.clone();
                let graph_edge_id = graph_edge_ref.id;
                let petgraph_edge_id = &self.edge_ids[&graph_edge_id];
                let edge_object = &self.from_graph[*petgraph_edge_id];

                sub_network.add_edge(
                    graph_node_id,
                    &graph_edge_target,
                    *graph_edge_id,
                    edge_object.get_metadata().clone(),
                );
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
