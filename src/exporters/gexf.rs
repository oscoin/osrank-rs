#![allow(unknown_lints)]
#![warn(clippy::all)]

use num_traits::Zero;
use oscoin_graph_api::{types, Direction, EdgeRef, Graph, GraphObject, Node};

use super::{size_from_rank, Exporter, Rank, RgbColor};
use crate::types::mock::{KeyValueAnnotator, MockNetwork};
use crate::types::Osrank;

use std::convert::TryInto;
use std::fs::{File, OpenOptions};
use std::hash::Hash;
use std::io::Write;
use std::path::Path;

static GEXF_META: &str = r###"<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2" xmlns:viz="http://www.gexf.net/1.2draft/viz" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.gexf.net/1.1draft http://www.gexf.net/1.1draft/gexf.xsd">
  <meta lastmodifieddate="2011-09-05">
    <creator>Gephi 0.8</creator>
    <description></description>
  </meta>
  <graph defaultedgetype="directed">
  <attributes class="node" mode="static">
    <attribute id="osrank" title="osrank" type="float">
        <default>0.0</default>
    </attribute>
  </attributes>
"###;

static GEXF_FOOTER: &str = "</gexf>";

#[derive(Debug)]
pub enum ExportError {
    IOError(std::io::Error),
}

impl From<std::io::Error> for ExportError {
    fn from(err: std::io::Error) -> ExportError {
        ExportError::IOError(err)
    }
}

pub trait IntoGexfXml {
    fn render(&self) -> String;
}

impl IntoGexfXml for String {
    fn render(&self) -> String {
        self.clone()
    }
}

impl IntoGexfXml for usize {
    fn render(&self) -> String {
        format!("{}", self)
    }
}

impl<V> IntoGexfXml for Option<V>
where
    V: IntoGexfXml,
{
    fn render(&self) -> String {
        match self {
            None => String::new(),
            Some(v) => v.render(),
        }
    }
}

impl IntoGexfXml for RgbColor {
    fn render(&self) -> String {
        format!(
            r###"<viz:color r="{}" g="{}" b="{}"></viz:color>"###,
            self.red, self.green, self.blue
        )
    }
}

pub struct NodeStyle {
    fill_color: RgbColor,
    size: f64,
}

impl IntoGexfXml for NodeStyle {
    fn render(&self) -> String {
        format!(
            r###"<viz:size value="{}"></viz:size>
              {}"###,
            self.size,
            self.fill_color.render()
        )
    }
}

pub struct NodeAttrs {
    rank: f64,
}

impl IntoGexfXml for NodeAttrs {
    fn render(&self) -> String {
        format!(
            r###"<attvalues>
                <attvalue for="osrank" value="{}"/>
              </attvalues>
              "###,
            self.rank,
        )
    }
}

struct GexfNode<I> {
    id: I,
    label: Option<String>,
    node_style: NodeStyle,
    node_attrs: NodeAttrs,
}

impl<I> IntoGexfXml for GexfNode<I>
where
    I: IntoGexfXml,
{
    fn render(&self) -> String {
        let mb_label = match &self.label {
            None => String::new(),
            Some(l) => format!(r###"label="{}""###, l),
        };
        format!(
            r###"<node id="{}" {}>
                 {}
                 {}
        </node>"###,
            self.id.render(),
            mb_label.render(),
            self.node_style.render(),
            self.node_attrs.render()
        )
    }
}

struct GexfEdge<I, N> {
    id: I,
    source: N,
    target: N,
}

impl<I, N> IntoGexfXml for GexfEdge<I, N>
where
    I: IntoGexfXml,
    N: IntoGexfXml,
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

/// Converts a `Graph::Node` into some GEXF tags.
fn write_node<N, V>(
    node: &N,
    annotator: &KeyValueAnnotator<<N as GraphObject>::Id, V>,
    out: &mut File,
) -> Result<(), ExportError>
where
    N: GraphObject<Data = types::NodeData<V>>,
    N::Id: IntoGexfXml + Clone + TryInto<String> + Eq + Hash,
    V: Into<Rank<f64>> + Zero + Clone,
{
    let data: <N as GraphObject>::Data = (*node).data().clone();
    let node_type: types::NodeType = data.clone().node_type;
    let rank: Rank<f64> = annotator
        .annotator
        .get(&node.id())
        .and_then(|r| Some((*r).clone()))
        .unwrap_or_else(V::zero)
        .into();
    let gexf_node = GexfNode {
        id: node.id().clone(),
        label: node.id().clone().try_into().ok(),
        node_style: NodeStyle {
            fill_color: node_type.into(),
            size: size_from_rank(rank),
        },
        node_attrs: NodeAttrs { rank: rank.rank },
    };

    out.write_all(gexf_node.render().as_bytes())?;
    Ok(())
}

/// Converts a `Graph::Edge` into some GEXF tags.
fn write_edge<N, E>(edge: &EdgeRef<N, E>, out: &mut File) -> Result<(), ExportError>
where
    E: IntoGexfXml + Clone,
    N: IntoGexfXml + Clone,
{
    let gexf_edge = GexfEdge {
        id: edge.id.clone(),
        source: edge.from.clone(),
        target: edge.to.clone(),
    };

    out.write_all(gexf_edge.render().as_bytes())?;
    Ok(())
}

/// Converts the `Graph` into some GEXF tags.
fn write_graph<G, V>(
    g: &G,
    annotator: &KeyValueAnnotator<<G::Node as GraphObject>::Id, V>,
    out: &mut File,
) -> Result<(), ExportError>
where
    G: Graph<NodeData = types::NodeData<V>>,
    <G as Graph>::Node: Node<types::NodeData<V>>,
    <G::Node as GraphObject>::Id: IntoGexfXml + Clone + TryInto<String> + Eq + Hash,
    <G::Edge as GraphObject>::Id: IntoGexfXml + Clone,
    V: Into<Rank<f64>> + Zero + Clone,
{
    let mut all_edges = Vec::new();

    out.write_all(b"<nodes>\n")?;

    for n in g.nodes() {
        write_node(n, annotator, out)?;
        out.write_all(b"\n")?;
        all_edges.extend(g.edges_directed(n.id(), Direction::Outgoing))
    }

    out.write_all(b"</nodes>\n<edges>")?;

    for e in &all_edges {
        write_edge(e, out)?;
        out.write_all(b"\n")?;
    }

    out.write_all(b"</edges>\n</graph>")?;

    Ok(())
}

/// Exports a graph `G` to a `.gexf` file.
///
/// This file can then be imported into one of the many graph visualisers,
/// like [Gephi](https://gephi.org/).
fn export_graph_impl<G, V>(
    g: &G,
    annotator: &KeyValueAnnotator<<G::Node as GraphObject>::Id, V>,
    out: &Path,
) -> Result<(), ExportError>
where
    G: Graph<NodeData = types::NodeData<V>>,
    <G as Graph>::Node: Node<types::NodeData<V>>,
    <G::Node as GraphObject>::Id: IntoGexfXml + Clone + TryInto<String> + Eq + Hash,
    <G::Edge as GraphObject>::Id: IntoGexfXml + Clone,
    V: Into<Rank<f64>> + Zero + Clone,
{
    let mut out_file = OpenOptions::new().write(true).create_new(true).open(out)?;

    out_file.write_all(GEXF_META.as_bytes())?;
    write_graph(g, annotator, &mut out_file)?;
    out_file.write_all(GEXF_FOOTER.as_bytes())?;

    Ok(())
}

pub struct GexfExporter<'a, G, V>
where
    G: Graph,
{
    graph: &'a G,
    annotator: &'a KeyValueAnnotator<<G::Node as GraphObject>::Id, V>,
    out_path: &'a str,
}

impl<'a, G, V> GexfExporter<'a, G, V>
where
    G: Graph,
{
    pub fn new(
        graph: &'a G,
        annotator: &'a KeyValueAnnotator<<G::Node as GraphObject>::Id, V>,
        out_path: &'a str,
    ) -> Self {
        GexfExporter {
            graph,
            annotator,
            out_path,
        }
    }
}

impl<'a> Exporter for GexfExporter<'a, MockNetwork<f64>, Osrank> {
    type ExporterOutput = ();
    type ExporterError = ExportError;
    fn export(self) -> Result<Self::ExporterOutput, Self::ExporterError> {
        let pth = self.out_path.to_owned() + ".gexf";
        let out_with_ext = Path::new(&pth);
        export_graph_impl(self.graph, self.annotator, &out_with_ext)
    }
}
