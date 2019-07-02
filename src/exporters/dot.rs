extern crate petgraph;

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use crate::types::Network;
use petgraph::dot::{Config, Dot};

/// Debug-only function to render a Network into a Graphiz dot file.
pub fn to_graphviz_dot(output_path: &Path, network: &Network) -> Result<(), Box<std::io::Error>> {
    let mut dot_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output_path)?;
    dot_file.write_fmt(format_args!(
        "{}",
        Dot::with_config(&network.from_graph, &[Config::EdgeNoLabel])
    ))?;
    Ok(())
}
