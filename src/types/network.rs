#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate fraction;
extern crate num_traits;
extern crate petgraph;

#[cfg(test)]
extern crate quickcheck;

use num_traits::Zero;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::ops::{Div, Mul};

use crate::protocol_traits::graph::{GraphExtras, PetgraphEdgeAdaptor, PetgraphNodeAdaptor};
use oscoin_graph_api::{
    types, Data, Direction, Edge, EdgeRefs, Edges, Graph, GraphDataReader, GraphDataWriter,
    GraphObject, GraphWriter, Id, Node, Nodes, NodesMut,
};
use petgraph::graph::{node_index, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Directed;

use crate::types::dynamic_weight::DynamicWeights;
use crate::types::Weight;

#[cfg(test)]
use crate::util::quickcheck::{frequency, Alphanumeric, Positive};

#[cfg(test)]
use quickcheck::{Arbitrary, Gen};

#[derive(Debug, Clone)]
pub struct Dependency<Id, NodeId, W> {
    id: Id,
    source: NodeId,
    target: NodeId,
    dependency_data: types::EdgeData<W>,
}

impl<Id, NodeId, W> fmt::Display for Dependency<Id, NodeId, W>
where
    W: fmt::Display,
    Id: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            f,
            "Dependency [ id = {}, type = {:#?} ]",
            self.id, self.dependency_data.edge_type
        )
    }
}

impl<DependencyId, NodeId, W> GraphObject for Dependency<DependencyId, NodeId, W> {
    type Id = DependencyId;
    type Data = types::EdgeData<W>;

    fn id(&self) -> &Self::Id {
        &self.id
    }

    fn data(&self) -> &Self::Data {
        &self.dependency_data
    }

    fn data_mut(&mut self) -> &mut Self::Data {
        &mut self.dependency_data
    }
}

impl<EdgeId, NodeId, W> Edge<W, NodeId, types::EdgeData<W>> for Dependency<EdgeId, NodeId, W> {
    fn source(&self) -> &NodeId {
        &self.source
    }

    fn target(&self) -> &NodeId {
        &self.target
    }

    fn weight(&self) -> W {
        panic!("Network doesn't support static weight calculation. Please use dynamic_weight() instead.")
    }

    fn edge_type(&self) -> &types::EdgeType {
        &self.dependency_data.edge_type
    }
}

impl<W, R> DynamicWeights for Network<W, R>
where
    W: Clone + Mul<Output = W> + Div<Output = W> + From<Weight>,
    R: Clone + Zero,
{
    fn dynamic_weight(
        &self,
        edge: &impl Edge<Self::Weight, <Self::Node as GraphObject>::Id, Self::EdgeData>,
        hyperparams: &types::HyperParameters<Self::Weight>,
    ) -> Self::Weight {
        let e_type = edge.edge_type();

        // Let's start by assigning this edge the stock default value, by
        // reading it from the hyperparams.
        let mut weight: Self::Weight = (hyperparams.get_param(&e_type.to_tag())).clone();

        // others can't be zero as there is at least one edge, i.e. the
        // input one.
        let others = edges_of_same_type(self, edge, Direction::Outgoing, e_type);

        let source_node = self
            .get_node(edge.source())
            .expect("dynamic_weight: source node not found.");

        // Then we need to do something different based on the type of edge.
        match e_type.to_tag() {
            types::EdgeTypeTag::ProjectToUserContribution => {
                // contrib is multiplied by the number of contributions of
                // the account to the project, divided by the total number
                // of contributions in the project.
                let total_project_contrib = source_node.node_type().total_contributions();
                let user_contribs = edge.edge_type().total_contributions();

                weight = weight * Weight::new(user_contribs, total_project_contrib).into()
            }
            types::EdgeTypeTag::UserToProjectContribution => {
                // contrib* and maintain* are multiplied by the number of
                // contributions of the account to the project, divided by
                // the total number of contributions of the account.
                let total_account_contrib = source_node.node_type().total_contributions();
                let user_contribs = edge.edge_type().total_contributions();

                weight = weight * Weight::new(user_contribs, total_account_contrib).into()
            }
            types::EdgeTypeTag::UserToProjectMembership => {
                // The weight is divided by the corresponding count of
                // outgoing edges of the same type on the node.
                weight = weight / others.into()
            }
            types::EdgeTypeTag::ProjectToUserMembership => {
                // contrib* and maintain* are multiplied by the number of
                // contributions of the account to the project, divided by
                // the total number of contributions of the account.
                let total_account_contrib = source_node.node_type().total_contributions();
                let user_contribs = edge.edge_type().total_contributions();

                weight = weight * Weight::new(user_contribs, total_account_contrib).into()
            }
            types::EdgeTypeTag::Dependency => {
                // The weight is divided by the corresponding count of
                // outgoing edges of the same type on the node.
                weight = weight / others.into()
            }
        }

        weight
    }
}

fn edges_of_same_type<G>(
    graph: &G,
    edge: &impl Edge<G::Weight, <G::Node as GraphObject>::Id, G::EdgeData>,
    direction: Direction,
    edge_type: &types::EdgeType,
) -> Weight
where
    G: Graph,
{
    Weight::new(
        graph
            .edges_directed(edge.source(), direction)
            .iter()
            .filter(|eref| eref.edge_type == edge_type)
            .count() as u32,
        1u32,
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Artifact<Id, W> {
    id: Id,
    artifact_data: types::NodeData<W>,
}

impl<Id, W> Artifact<Id, W>
where
    W: Zero,
{
    pub fn new_account(id: Id, contributions_to_all_projects: u32) -> Self {
        Artifact {
            id,
            artifact_data: types::NodeData {
                node_type: types::NodeType::User {
                    contributions_to_all_projects,
                },
                rank: types::NodeRank { rank: W::zero() },
            },
        }
    }

    pub fn new_project(id: Id, contributions_from_all_users: u32) -> Self {
        Artifact {
            id,
            artifact_data: types::NodeData {
                node_type: types::NodeType::Project {
                    contributions_from_all_users,
                },
                rank: types::NodeRank { rank: W::zero() },
            },
        }
    }
}

impl<ArtifactId, W> GraphObject for Artifact<ArtifactId, W>
where
    ArtifactId: Clone,
{
    type Id = ArtifactId;
    type Data = types::NodeData<W>;

    fn id(&self) -> &Self::Id {
        &self.id
    }

    fn data(&self) -> &Self::Data {
        &self.artifact_data
    }

    fn data_mut(&mut self) -> &mut Self::Data {
        &mut self.artifact_data
    }
}

impl<ArtifactId, W> Node<types::NodeData<W>> for Artifact<ArtifactId, W>
where
    ArtifactId: Clone,
{
    fn node_type(&self) -> &types::NodeType {
        &self.artifact_data.node_type
    }
}

impl<Id, W> fmt::Display for Artifact<Id, W>
where
    Id: fmt::Display,
    W: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self.artifact_data.node_type {
            types::NodeType::Project { .. } => write!(
                f,
                "Project [ id: {}, osrank: {:.5} ]",
                self.id, self.artifact_data.rank.rank
            ),
            types::NodeType::User { .. } => write!(
                f,
                "User [ id: {}, osrank: {:.5} ]",
                self.id, self.artifact_data.rank.rank
            ),
        }
    }
}

#[cfg(test)]
impl<W> Arbitrary for Artifact<String, W>
where
    W: Zero + Arbitrary,
{
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let id: String = Alphanumeric::arbitrary(g).get_alphanumeric;
        let acc_contribs: Positive<u32> = Arbitrary::arbitrary(g);
        let prj_contribs: Positive<u32> = Arbitrary::arbitrary(g);
        let choices = vec![
            (
                50,
                Artifact::new_account(id.clone(), acc_contribs.get_positive),
            ),
            (50, Artifact::new_project(id, prj_contribs.get_positive)),
        ];
        frequency(g, choices)
    }
}

/// The network graph from the paper, comprising of both accounts and projects.
#[derive(Clone, Debug)]
pub struct Network<W, R> {
    from_graph: petgraph::Graph<Artifact<String, R>, Dependency<usize, String, W>, Directed>,
    node_ids: HashMap<String, NodeIndex>,
    edge_ids: HashMap<usize, EdgeIndex>,
}

impl<W, R> Default for Network<W, R> {
    fn default() -> Self {
        Network {
            from_graph: petgraph::Graph::new(),
            node_ids: HashMap::default(),
            edge_ids: HashMap::default(),
        }
    }
}

impl<W, R> Network<W, R>
where
    W: Clone,
    R: Clone + Zero,
{
    /// Adds an Artifact to the Network.
    fn add_artifact(&mut self, id: String, artifact_data: types::NodeData<R>) {
        let id_cloned = id.clone();
        let mut a = match artifact_data.node_type {
            types::NodeType::Project {
                contributions_from_all_users,
            } => Artifact::new_project(id, contributions_from_all_users),
            types::NodeType::User {
                contributions_to_all_projects,
            } => Artifact::new_account(id, contributions_to_all_projects),
        };

        a.artifact_data = artifact_data;

        let nid = self.from_graph.add_node(a);

        self.node_ids.insert(id_cloned, nid);
    }

    /// Adds a Dependency to the Network. It's unsafe in the sense it's
    /// callers' responsibility to ensure that the source and target exist
    /// in the input Network.
    fn unsafe_add_dependency(
        &mut self,
        source_id: &String,
        target_id: &String,
        source: usize,
        target: usize,
        id: usize,
        dependency_data: types::EdgeData<W>,
    ) {
        let d = Dependency {
            id,
            source: source_id.to_string(),
            target: target_id.to_string(),
            dependency_data,
        };
        let eid = self
            .from_graph
            .add_edge(node_index(source), node_index(target), d);
        self.edge_ids.insert(id, eid);
    }

    pub fn is_empty(&self) -> bool {
        self.from_graph.node_count() == 0
            && self.from_graph.edge_count() == 0
            && self.node_ids.is_empty()
            && self.edge_ids.is_empty()
    }
}

impl<W, R> GraphWriter for Network<W, R>
where
    W: Clone,
    R: Clone + Zero,
{
    fn add_node(&mut self, node_id: Id<Self::Node>, node_data: types::NodeData<R>) {
        self.add_artifact(node_id, node_data);
    }

    fn remove_node(&mut self, node_id: Id<Self::Node>) {
        // Removes the node from petgraph as well as from the internal map
        if let Some(nid) = self.node_ids.get(&node_id) {
            self.from_graph.remove_node(*nid);
            self.node_ids.remove(&node_id);
        }
    }

    fn add_edge(
        &mut self,
        edge_id: Id<Self::Edge>,
        source: &Id<Self::Node>,
        to: &Id<Self::Node>,
        edge_data: types::EdgeData<W>,
    ) {
        //NOTE(adn) Should we return a `Result` rather than blowing everything up?
        match self
            .node_ids
            .get(source)
            .iter()
            .zip(self.node_ids.get(to).iter())
            .next()
        {
            Some((s, t)) => {
                let src = (*s).index();
                let tgt = (*t).index();
                self.unsafe_add_dependency(source, to, src, tgt, edge_id, edge_data);
            }
            None => panic!(
                "add_adge: invalid link, source {:#?} or target {:#?} are missing.",
                source, to
            ),
        }
    }

    fn remove_edge(&mut self, edge_id: Id<Self::Edge>) {
        // Removes the edge from petgraph as well as from the internal map
        if let Some(eid) = self.edge_ids.get(&edge_id) {
            self.from_graph.remove_edge(*eid);
            self.edge_ids.remove(&edge_id);
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
}

impl<W, R> Graph for Network<W, R>
where
    R: Zero,
{
    type Node = Artifact<String, R>;
    type Edge = Dependency<usize, String, W>;
    type Weight = W;
    type NodeData = types::NodeData<R>;
    type EdgeData = types::EdgeData<W>;

    fn neighbors(&self, node_id: &Id<Self::Node>) -> Nodes<Self::Node> {
        let mut nodes = Vec::new();
        let mut unique_node_ids = BTreeSet::new();

        unique_node_ids.insert(node_id); // avoids having the input as neighbor.

        if let Some(nid) = self.node_ids.get(node_id) {
            for raw_edge in self.from_graph.raw_edges() {
                if raw_edge.target() == *nid {
                    match self.from_graph.node_weight(raw_edge.source()) {
                        None => continue,
                        Some(node_from) => {
                            if !unique_node_ids.contains(node_from.id()) {
                                nodes.push(node_from);
                                unique_node_ids.insert(node_from.id());
                            }
                        }
                    }
                }

                if raw_edge.source() == *nid {
                    match self.from_graph.node_weight(raw_edge.target()) {
                        None => continue,
                        Some(node_to) => {
                            if !unique_node_ids.contains(node_to.id()) {
                                nodes.push(node_to);
                                unique_node_ids.insert(node_to.id());
                            }
                        }
                    }
                }
            }
        }

        Nodes {
            range: nodes.into_iter(),
        }
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
                .iter()
                .map(|n| n.petgraph_node_data())
                .collect::<Vec<&Self::Node>>()
                .into_iter(),
        }
    }

    fn edges(&self, node: &Id<Self::Node>) -> Edges<Self::Edge> {
        let mb_id = self.node_ids.get(node);
        match mb_id {
            None => Edges {
                range: Vec::new().into_iter(),
            },
            Some(nid) => Edges {
                range: self
                    .from_graph
                    .edges(*nid)
                    .map(|e| e.petgraph_edge_data())
                    .collect::<Vec<&Self::Edge>>()
                    .into_iter(),
            },
        }
    }

    fn edges_directed(
        &self,
        node_id: &Id<Self::Node>,
        dir: Direction,
    ) -> EdgeRefs<Id<Self::Node>, Id<Self::Edge>> {
        let mut neighbors = Vec::default();

        let petgraph_dir = match dir {
            Direction::Outgoing => petgraph::Direction::Outgoing,
            Direction::Incoming => petgraph::Direction::Incoming,
        };

        if let Some(nid) = self.node_ids.get(node_id) {
            for eref in self.from_graph.edges_directed(*nid, petgraph_dir) {
                neighbors.push(oscoin_graph_api::EdgeRef {
                    from: self.from_graph[eref.source()].id(),
                    to: self.from_graph[eref.target()].id(),
                    id: self.from_graph[eref.id()].id(),
                    edge_type: &self.from_graph[eref.id()].dependency_data.edge_type,
                })
            }
        }

        neighbors
    }
}

impl<W, R> GraphDataWriter for Network<W, R>
where
    W: Clone,
    R: Clone + Zero,
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

impl<W, R> GraphDataReader for Network<W, R>
where
    R: Zero,
{
    fn node_data(&self, node_id: &Id<Self::Node>) -> Option<&Data<Self::Node>> {
        self.node_ids
            .get(node_id)
            .and_then(|n| Some(self.from_graph[*n].data()))
    }

    fn edge_data(&self, edge_id: &Id<Self::Edge>) -> Option<&Data<Self::Edge>> {
        self.edge_ids
            .get(edge_id)
            .and_then(|e| Some(self.from_graph[*e].data()))
    }
}

impl<W, R> GraphExtras for Network<W, R>
where
    W: Clone,
    R: Clone + Zero,
{
    fn edge_count(&self) -> usize {
        self.from_graph.edge_count()
    }

    fn node_count(&self) -> usize {
        self.from_graph.node_count()
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
            for graph_edge_ref in self.edges_directed(graph_node_id, Direction::Outgoing) {
                let graph_edge_target = graph_edge_ref.to.clone();
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

impl<'a, W, R> PrintableGraph<'a> for Network<W, R>
where
    W: fmt::Display + Clone,
    R: fmt::Display + Clone + Zero,
{
    fn print_nodes(&self) {
        for arti in self
            .from_graph
            .raw_nodes()
            .iter()
            .map(|node| node.petgraph_node_data())
        {
            println!("{}", arti);
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::types::{Osrank, Weight};

    fn network_fixture() -> Network<Weight, Osrank> {
        let mut network = Network::default();

        for node in &["p1", "p2", "p3"] {
            let a = Artifact::new_project(node.to_string(), 100);
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
                types::EdgeData {
                    edge_type: types::EdgeType::Dependency,
                    weight: edge.2,
                },
            )
        }

        network
    }

    #[test]
    // We test that `neighbors` returns _all_ the neighbors of a node,
    // which means nodes derived from _both_ incoming & outgoing edges.
    fn neighbours_returns_outgoing_and_incoming() {
        let graph = network_fixture();
        let all_neighbours = graph
            .neighbors(&"p1".to_string())
            .map(|n| n.id())
            .collect::<Vec<&String>>();
        assert_eq!(all_neighbours, vec!["p2", "p3"]);
    }

    #[test]
    // We test that `edges_directed` returns only the outgoing edges when
    // the `Outgoing` direction is specified.
    fn edges_directed_outgoing() {
        let graph = network_fixture();
        let all_outgoing = graph
            .edges_directed(&"p1".to_string(), Direction::Outgoing)
            .into_iter()
            .map(|eref| graph.get_node(eref.to).and_then(|n| Some(n.id())))
            .collect::<Vec<Option<&String>>>();

        // There are two outgoing edges to p2 in the fixture graph.
        assert_eq!(
            all_outgoing,
            vec![Some(&"p2".to_string()), Some(&"p2".to_string())]
        );
    }

    #[test]
    // We test that `edges_directed` returns only the *incoming* edges when
    // the `Incoming` direction is specified.
    fn edges_directed_incoming() {
        let graph = network_fixture();
        let all_outgoing = graph
            .edges_directed(&"p1".to_string(), Direction::Incoming)
            .into_iter()
            .map(|eref| graph.get_node(eref.from).and_then(|n| Some(n.id())))
            .collect::<Vec<Option<&String>>>();
        assert_eq!(all_outgoing, vec![Some(&"p3".to_string())]);
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
    fn artifact_get_set_metadata_roundtrip(meta: types::NodeData<f64>) {
        let mut a = Artifact::new_account("foo", 0);
        *a.data_mut() = meta.clone();
        assert_eq!(*a.data(), meta);
    }
}
