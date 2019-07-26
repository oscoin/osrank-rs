extern crate osrank;

use std::fs::File;

use self::osrank::importers::gephi_json::{from_gephi_json, GephiEdge, GephiNode};

fn main() -> Result<(), Box<std::error::Error>> {
    let cargo_nodes = File::open("data/cargo_nodes.json")?;
    let cargo_edges = File::open("data/cargo_edges.json")?;
    let gephi_nodes: Vec<GephiNode> = serde_json::from_reader(cargo_nodes)?;
    let gephi_edges: Vec<GephiEdge> = serde_json::from_reader(cargo_edges)?;

    println!("Building a network from a file..");
    let cargo_network = from_gephi_json(gephi_nodes, gephi_edges);

    // Render to file
    println!("Rendering the network to a file..");
    cargo_network.to_graphviz_dot("rust_network.dot".as_ref())?;
    Ok(())
}
