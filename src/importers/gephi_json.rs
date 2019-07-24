extern crate num_traits;
extern crate serde;
extern crate serde_json;

use crate::protocol_traits::graph::Graph;
use crate::types::{ArtifactType, DependencyType, Network, Weight};
use num_traits::Zero;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct GephiNode {
    id: String,
    a: Vec<Value>,
}

#[derive(Debug, Deserialize)]
pub struct GephiEdge {
    s: usize,
    t: usize,
    w: f64,
}

/// Construct a Network out of a Gephi json file. Only projects are supported
/// for now.
pub fn from_gephi_json(nodes: Vec<GephiNode>, edges: Vec<GephiEdge>) -> Network<Weight> {
    let mut network = Network::default();
    for node in nodes {
        let project_meta = ArtifactType::Project {
            osrank: Zero::zero(),
        };
        network.add_node(node.id, project_meta);
    }

    for (ix, edge) in edges.iter().enumerate() {
        //FIXME(adn) This should take into account normalisation and
        //redistribution of the weights etc.
        let depends_on = DependencyType::Depend(Weight::new(1, 1));
        network.add_edge(&edge.s.to_string(), &edge.t.to_string(), ix, depends_on)
    }

    network
}
