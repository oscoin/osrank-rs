extern crate serde;
extern crate serde_json;

use crate::types::{Artifact, Dependency, Network, ProjectAttributes, Weight};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct GephiNode {
    id: String,
    a: Vec<Value>,
}

#[derive(Debug, Deserialize)]
pub struct GephiEdge {
    s: u32,
    t: u32,
    w: f64,
}

/// Construct a Network out of a Gephi json file. Only projects are supported
/// for now.
pub fn from_gephi_json(nodes: Vec<GephiNode>, edges: Vec<GephiEdge>) -> Network {
    let mut network = Network::default();
    for node in nodes {
        let project = Artifact::Project(ProjectAttributes {
            id: node.id,
            osrank: None,
        });
        network.add_artifact(project);
    }

    for edge in edges {
        //FIXME(adn) This should take into account normalisation and
        //redistribution of the weights etc.
        let depends_on = Dependency::Depend(Weight::new(1, 1));
        network.unsafe_add_dependency(edge.s, edge.t, depends_on)
    }

    network
}
