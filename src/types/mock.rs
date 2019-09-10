#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate oscoin_graph_api;

use crate::types::network::{Artifact, DependencyType, Network};
use crate::util::quickcheck::frequency;
use oscoin_graph_api::{GraphObject, GraphWriter};
use quickcheck::{Arbitrary, Gen};
use rand::Rng;

pub type MockNetwork = Network<f64>;

/// Equivalent to `newtype Mock a = Mock a` in Haskell.
///
/// Useful to define some trait which operates over mocks implementation only.
pub struct Mock<A> {
    pub unmock: A,
}

#[derive(Debug)]
struct ArbitraryEdge<'a> {
    source: &'a String,
    target: &'a String,
    id: usize,
    weight: f64,
    data: DependencyType<f64>,
}

impl Arbitrary for MockNetwork {
    // Tries to generate an arbitrary Network.
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let mut graph = Network::default();
        let nodes: Vec<Artifact<String>> = Arbitrary::arbitrary(g);

        let edges = arbitrary_normalised_edges_from(g, &nodes);

        for n in &nodes {
            graph.add_node(n.id().clone(), n.data().clone())
        }

        for e in edges {
            graph.add_edge(e.id, e.source, e.target, e.weight, e.data)
        }

        graph
    }
}

#[derive(Clone)]
enum NewEdgeAction {
    SkipNode,
    UseNode,
}

impl Arbitrary for NewEdgeAction {
    // Tries to generate an arbitrary Network.
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let choices = vec![(80, NewEdgeAction::UseNode), (20, NewEdgeAction::SkipNode)];
        frequency(g, choices)
    }
}

/// Attempts to generate a vector of random edges that respect the osrank
/// invariant, i.e. that the sum of the weight of the outgoing ones from a
/// certain node is 1.
fn arbitrary_normalised_edges_from<'a, G: Gen + Rng>(
    g: &mut G,
    nodes: &'a [Artifact<String>],
) -> Vec<ArbitraryEdge<'a>> {
    let mut edges = Vec::new();
    let mut id_counter = 0;

    for node in nodes {
        let action: NewEdgeAction = Arbitrary::arbitrary(g);
        match action {
            NewEdgeAction::SkipNode => continue,
            NewEdgeAction::UseNode => {
                // Pick a set of random nodes (it can include this node as
                // well) and generate a bunch of edges between them.

                let edges_num = g.gen_range(1, 6); // Up to 5 outgoing edges
                let node_ixs = (0..edges_num)
                    .map(|_| g.gen_range(0, nodes.len()))
                    .collect::<Vec<usize>>();

                for ix in node_ixs {
                    let w = 1.0 / f64::from(edges_num);

                    edges.push(ArbitraryEdge {
                        id: id_counter,
                        source: &node.id(),
                        target: &nodes[ix].id(),
                        weight: w,
                        data: DependencyType::Influence(w),
                    });

                    id_counter += 1;
                }
            }
        }
    }

    edges
}
