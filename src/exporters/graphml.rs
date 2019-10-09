#![allow(unknown_lints)]
#![warn(clippy::all)]

use crate::types::mock::KeyValueAnnotator;
use oscoin_graph_api::{Direction, Edge, EdgeRef, Graph, GraphObject};

use super::{size_from_rank, NodeType, Rank, RgbColor};

use num_traits::Zero;
use std::convert::TryInto;
use std::fs::{File, OpenOptions};
use std::hash::Hash;
use std::io::Write;
use std::path::Path;

/// The static header for a `.graphml` file.
///
/// NOTE: Most attributes can be interpreted by a specific viewer only.
/// Here we did choose gephi as it seems to do a better job than Cytoscape
/// in preserving key information like the color of nodes.
static GRAPHML_META: &str = r###"<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"  
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
     http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
     <key for="edge" id="edge_weight" attr.name="weight" attr.type="double"/>
     <key attr.name="label" attr.type="string" for="node" id="label"/>
     <key attr.name="rank" attr.type="float" for="node" id="rank">
       <default>0.0</default>
     </key>
     <key attr.name="r" attr.type="int" for="node" id="r">
       <default>0</default>
     </key>
     <key attr.name="g" attr.type="int" for="node" id="g">
       <default>0</default>
     </key>
     <key attr.name="b" attr.type="int" for="node" id="b">
       <default>255</default>
     </key>
     <key attr.name="size" attr.type="float" for="node" id="size">
       <default>10.0</default>
     </key>
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

impl IntoGraphMlXml for String {
    fn render(&self) -> String {
        self.clone()
    }
}

impl IntoGraphMlXml for f64 {
    fn render(&self) -> String {
        format!("{}", self)
    }
}

impl IntoGraphMlXml for usize {
    fn render(&self) -> String {
        format!("{}", self)
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

impl IntoGraphMlXml for RgbColor {
    fn render(&self) -> String {
        format!(
            r###"
          <data key="r">{}</data>
          <data key="g">{}</data>
          <data key="b">{}</data>"###,
            self.red, self.green, self.blue
        )
    }
}

pub struct NodeAttrs {
    label: String,
    fill_color: RgbColor,
    size: f64,
    rank: f64,
}

impl IntoGraphMlXml for NodeAttrs {
    fn render(&self) -> String {
        format!(
            r###"<data key="label">{}</data>
                 <data key="size">{}</data>
                 <data key="rank">{}</data>
                 {}
                "###,
            self.label,
            self.size,
            self.rank,
            self.fill_color.render()
        )
    }
}

struct GraphMlNode<I> {
    id: I,
    node_style: NodeAttrs,
}

impl<I> IntoGraphMlXml for GraphMlNode<I>
where
    I: IntoGraphMlXml,
{
    fn render(&self) -> String {
        format!(
            r###"<node id="{}">{}</node>"###,
            self.id.render(),
            self.node_style.render()
        )
    }
}

struct GexfEdge<I, N, W> {
    id: I,
    source: N,
    target: N,
    weight: W,
}

impl<I, N, W> IntoGraphMlXml for GexfEdge<I, N, W>
where
    I: IntoGraphMlXml,
    N: IntoGraphMlXml,
    W: IntoGraphMlXml,
{
    fn render(&self) -> String {
        format!(
            r###"<edge id="{}" source="{}" target="{}">
                   <data key="edge_weight">{}</data>
</edge>"###,
            self.id.render(),
            self.source.render(),
            self.target.render(),
            self.weight.render()
        )
    }
}

/// Converts a `Graph::Node` into some GRAPHML tags.
fn write_node<N, V>(
    node: &N,
    annotator: &KeyValueAnnotator<<N as GraphObject>::Id, V>,
    out: &mut File,
) -> Result<(), ExportError>
where
    N: GraphObject,
    V: Into<Rank<f64>> + Zero + Clone,
    <N as GraphObject>::Data: Clone + Into<NodeType> + Into<Rank<f64>>,
    <N as GraphObject>::Id: IntoGraphMlXml + Clone + TryInto<String> + Eq + Hash,
{
    let data: <N as GraphObject>::Data = (*node).data().clone();
    let rank: Rank<f64> = annotator
        .annotator
        .get(&node.id())
        .and_then(|r| Some((*r).clone()))
        .unwrap_or_else(V::zero)
        .into();
    let node_type: NodeType = data.clone().into();
    let lbl = node
        .id()
        .clone()
        .try_into()
        .unwrap_or_else(|_| String::from("Unlabeled node"));

    let gexf_node = GraphMlNode {
        id: node.id().clone(),
        node_style: NodeAttrs {
            label: lbl,
            size: size_from_rank(rank),
            rank: rank.rank,
            fill_color: node_type.into(),
        },
    };

    out.write_all(gexf_node.render().as_bytes())?;
    Ok(())
}

/// Converts a `Graph::Edge` into some GRAPHML tags.
fn write_edge<G>(
    g: &G,
    edge: &EdgeRef<<G::Node as GraphObject>::Id, <G::Edge as GraphObject>::Id>,
    out: &mut File,
) -> Result<(), ExportError>
where
    G: Graph,
    <G::Node as GraphObject>::Id: IntoGraphMlXml + Clone,
    <G::Edge as GraphObject>::Id: IntoGraphMlXml + Clone,
    <G as Graph>::Weight: IntoGraphMlXml + Zero,
{
    let gexf_edge: GexfEdge<
        <G::Edge as GraphObject>::Id,
        <G::Node as GraphObject>::Id,
        <G as Graph>::Weight,
    > = GexfEdge {
        id: edge.id.clone(),
        source: edge.from.clone(),
        target: edge.to.clone(),
        weight: g
            .get_edge(edge.id)
            .and_then(|e| Some(e.weight()))
            .unwrap_or_else(<G as Graph>::Weight::zero),
    };

    out.write_all(gexf_edge.render().as_bytes())?;
    Ok(())
}

/// Converts the `Graph` into some GRAPHML tags.
fn write_graph<G, V>(
    g: &G,
    annotator: &KeyValueAnnotator<<G::Node as GraphObject>::Id, V>,
    out: &mut File,
) -> Result<(), ExportError>
where
    G: Graph,
    V: Into<Rank<f64>> + Zero + Clone,
    <G::Node as GraphObject>::Data: Clone + Into<NodeType> + Into<Rank<f64>>,
    <G::Node as GraphObject>::Id: IntoGraphMlXml + Clone + TryInto<String> + Eq + Hash,
    <G::Edge as GraphObject>::Id: IntoGraphMlXml + Clone,
    <G as Graph>::Weight: IntoGraphMlXml + Zero,
{
    let mut all_edges = Vec::new();

    out.write_all(b"<graph id=\"osrank\" edgedefault=\"directed\">\n")?;

    for n in g.nodes() {
        write_node(n, annotator, out)?;
        out.write_all(b"\n")?;
        all_edges.extend(g.edges_directed(n.id(), Direction::Outgoing))
    }

    for e in &all_edges {
        write_edge(g, e, out)?;
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
pub fn export_graph<G, V>(
    g: &G,
    annotator: &KeyValueAnnotator<<G::Node as GraphObject>::Id, V>,
    out: &Path,
) -> Result<(), ExportError>
where
    G: Graph,
    V: Into<Rank<f64>> + Zero + Clone,
    <G::Node as GraphObject>::Data: Clone + Into<NodeType> + Into<Rank<f64>>,
    <G::Node as GraphObject>::Id: IntoGraphMlXml + Clone + TryInto<String> + Eq + Hash,
    <G::Edge as GraphObject>::Id: IntoGraphMlXml + Clone,
    <G as Graph>::Weight: IntoGraphMlXml + Zero,
{
    let mut out_file = OpenOptions::new().write(true).create_new(true).open(out)?;

    out_file.write_all(GRAPHML_META.as_bytes())?;
    write_graph(g, annotator, &mut out_file)?;
    out_file.write_all(GRAPHML_FOOTER.as_bytes())?;

    Ok(())
}
