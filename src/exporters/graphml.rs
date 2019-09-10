#![allow(unknown_lints)]
#![warn(clippy::all)]

use oscoin_graph_api::{Direction, EdgeRef, Graph, GraphObject};

use std::convert::TryInto;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

static GRAPHML_META: &str = r###"<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"  
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:y="http://www.yworks.com/xml/graphml"
    xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
     http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
     <key for="node" id="node_style" yfiles.type="nodegraphics"/>
"###;

static GRAPHML_FOOTER: &str = "</graphml>";

#[derive(Debug)]
pub enum ExportError {
    IOError(std::io::Error),
}

impl From<std::io::Error> for ExportError {
    fn from(err: std::io::Error) -> ExportError {
        ExportError::IOError(err)
    }
}

pub trait IntoGraphMlXml {
    fn render(&self) -> String;
}

struct GexfAttribute<K, V> {
    attr_for: K,
    attr_value: V,
}

impl IntoGraphMlXml for String {
    fn render(&self) -> String {
        self.clone()
    }
}

impl IntoGraphMlXml for usize {
    fn render(&self) -> String {
        format!("{}", self)
    }
}

impl<K, V> IntoGraphMlXml for GexfAttribute<K, V>
where
    K: IntoGraphMlXml,
    V: IntoGraphMlXml,
{
    fn render(&self) -> String {
        format!(
            "<attvalue for=\"{}\" value=\"{}\"/>",
            self.attr_for.render(),
            self.attr_value.render()
        )
    }
}

impl<V> IntoGraphMlXml for Option<V>
where
    V: IntoGraphMlXml,
{
    fn render(&self) -> String {
        match self {
            None => String::new(),
            Some(v) => v.render(),
        }
    }
}

struct GraphMlNode<I> {
    id: I,
    label: Option<String>,
    attrs: Vec<GexfAttribute<String, String>>,
}

impl<I> IntoGraphMlXml for GraphMlNode<I>
where
    I: IntoGraphMlXml,
{
    fn render(&self) -> String {
        format!(
            r###"<node id="{}">
                <data key="node_style">
                  <y:ShapeNode>
                    <y:NodeLabel>{}</y:NodeLabel>
                 </y:ShapeNode>
                </data>
</node>"###,
            self.id.render(),
            self.label.render()
        )
    }
}

struct GexfEdge<I, N> {
    id: I,
    source: N,
    target: N,
}

impl<I, N> IntoGraphMlXml for GexfEdge<I, N>
where
    I: IntoGraphMlXml,
    N: IntoGraphMlXml,
{
    fn render(&self) -> String {
        format!(
            "<edge id=\"{}\" source=\"{}\" target=\"{}\"/>",
            self.id.render(),
            self.source.render(),
            self.target.render(),
        )
    }
}

/// Converts a `Graph::Node` into some GRAPHML tags.
fn write_node<N>(node: &N, out: &mut File) -> Result<(), ExportError>
where
    N: GraphObject,
    N::Id: IntoGraphMlXml + Clone + TryInto<String>,
{
    let gexf_node = GraphMlNode {
        id: node.id().clone(),
        label: Some(
            node.id()
                .clone()
                .try_into()
                .unwrap_or(String::from("Unlabeled node")),
        ),
        attrs: Vec::new(),
    };

    out.write_all(gexf_node.render().as_bytes())?;
    Ok(())
}

/// Converts a `Graph::Edge` into some GRAPHML tags.
fn write_edge<N, E>(edge: &EdgeRef<N, E>, out: &mut File) -> Result<(), ExportError>
where
    E: IntoGraphMlXml + Clone,
    N: IntoGraphMlXml + Clone,
{
    let gexf_edge = GexfEdge {
        id: edge.id.clone(),
        source: edge.from.clone(),
        target: edge.to.clone(),
    };

    out.write_all(gexf_edge.render().as_bytes())?;
    Ok(())
}

/// Converts the `Graph` into some GRAPHML tags.
fn write_graph<G>(g: &G, out: &mut File) -> Result<(), ExportError>
where
    G: Graph,
    <G::Node as GraphObject>::Id: IntoGraphMlXml + Clone + TryInto<String>,
    <G::Edge as GraphObject>::Id: IntoGraphMlXml + Clone,
{
    let mut all_edges = Vec::new();

    out.write_all(b"<graph id=\"osrank\" edgedefault=\"directed\">\n")?;

    for n in g.nodes() {
        write_node(n, out)?;
        out.write_all(b"\n")?;
        all_edges.extend(g.edges_directed(n.id(), Direction::Outgoing))
    }

    for e in &all_edges {
        write_edge(e, out)?;
        out.write_all(b"\n")?;
    }

    out.write_all(b"</graph>")?;

    Ok(())
}

/// Exports a graph `G` to a `.graphml` file.
///
/// This file can then be imported into one of the many graph visualisers,
/// like [Gephi](https://gephi.org/).
/// For a more exhaustive explanation of GraphML, refers to the
/// [official documentation](http://graphml.graphdrawing.org/).
pub fn export_graph<G>(g: &G, out: &Path) -> Result<(), ExportError>
where
    G: Graph,
    <G::Node as GraphObject>::Id: IntoGraphMlXml + Clone + TryInto<String>,
    <G::Edge as GraphObject>::Id: IntoGraphMlXml + Clone,
{
    let mut out_file = OpenOptions::new().write(true).create_new(true).open(out)?;

    out_file.write_all(GRAPHML_META.as_bytes())?;
    write_graph(g, &mut out_file)?;
    out_file.write_all(GRAPHML_FOOTER.as_bytes())?;

    Ok(())
}
