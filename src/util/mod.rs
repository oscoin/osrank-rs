extern crate quickcheck as qc;
pub mod quickcheck;

use num_traits::Zero;
use oscoin_graph_api::{types, Edge, Graph, GraphWriter, Id, Node};
use qc::{Arbitrary, Gen};
use std::fmt;

#[derive(Clone)]
pub struct Pretty<A> {
    pub unpretty: A,
}

impl<A: fmt::Debug> fmt::Debug for Pretty<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:#?}", self.unpretty)
    }
}

impl<A: Arbitrary> Arbitrary for Pretty<A> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Pretty {
            unpretty: Arbitrary::arbitrary(g),
        }
    }
}

// General helper functions to build graphs slightly easily.

pub fn add_projects<G, E, W>(graph: &mut G, ns: impl Iterator<Item = (E, u32)>)
where
    <G as Graph>::Node: Node<types::NodeData<W>>,
    G: GraphWriter<NodeData = types::NodeData<W>>,
    W: Zero,
    E: Into<Id<G::Node>>,
{
    for (project_id, contribs) in ns.into_iter() {
        graph.add_node(
            project_id.into(),
            types::NodeData {
                node_type: types::NodeType::Project {
                    contributions_from_all_users: contribs,
                },
                rank: Zero::zero(),
            },
        )
    }
}

pub fn add_users<G, E, W>(graph: &mut G, ns: impl Iterator<Item = (E, u32)>)
where
    <G as Graph>::Node: Node<types::NodeData<W>>,
    G: GraphWriter<NodeData = types::NodeData<W>>,
    W: Zero,
    E: Into<Id<G::Node>>,
{
    for (user_id, contribs) in ns.into_iter() {
        graph.add_node(
            user_id.into(),
            types::NodeData {
                node_type: types::NodeType::User {
                    contributions_to_all_projects: contribs,
                },
                rank: Zero::zero(),
            },
        )
    }
}

pub fn add_edges<G, S, T, W>(
    graph: &mut G,
    es: impl Iterator<Item = (Id<G::Edge>, S, T, types::EdgeData<W>)>,
) where
    <G as Graph>::Edge: Edge<W, Id<G::Node>, types::EdgeData<W>>,
    G: GraphWriter<EdgeData = types::EdgeData<W>>,
    <G as Graph>::Edge: Edge<<G as Graph>::Weight, Id<G::Node>, types::EdgeData<W>>,
    S: Into<Id<G::Node>>,
    T: Into<Id<G::Node>>,
{
    for (edge_id, source, target, data) in es.into_iter() {
        graph.add_edge(edge_id, &source.into(), &target.into(), data)
    }
}
