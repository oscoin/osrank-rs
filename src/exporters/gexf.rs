#![allow(unknown_lints)]
#![warn(clippy::all)]

use crate::protocol_traits::graph::{EdgeReference, Graph, GraphObject};
use crate::types::network::Artifact;

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

static GEXF_META: &str = r###"
<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd" version="1.2">
    <meta lastmodifieddate="2009-03-20">
        <creator>Gephi.org</creator>
        <description>A Web network</description>
    </meta>
"###;

static GEXF_FOOTER: &str = "</gexf>";

pub enum ExportError {
    IOError(std::io::Error),
}

impl From<std::io::Error> for ExportError {
    fn from(err: std::io::Error) -> ExportError {
        ExportError::IOError(err)
    }
}

pub trait IntoGefxXml {
    fn render(&self) -> String;
}

struct GexfAttribute<K, V> {
    attr_for: K,
    attr_value: V,
}

impl IntoGefxXml for String {
    fn render(&self) -> String {
        self.clone()
    }
}

impl<K, V> IntoGefxXml for GexfAttribute<K, V>
where
    K: IntoGefxXml,
    V: IntoGefxXml,
{
    fn render(&self) -> String {
        format!(
            "<attvalue for=\"{}\" value=\"{}\"/>",
            self.attr_for.render(),
            self.attr_value.render()
        )
    }
}

impl<V> IntoGefxXml for Option<V>
where
    V: IntoGefxXml,
{
    fn render(&self) -> String {
        match self {
            None => String::new(),
            Some(v) => v.render(),
        }
    }
}

struct GexfNode<I> {
    id: I,
    label: Option<String>,
    attrs: Vec<GexfAttribute<String, String>>,
}

impl<I> IntoGefxXml for GexfNode<I>
where
    I: IntoGefxXml,
{
    fn render(&self) -> String {
        let mut values = String::new();

        for a in &self.attrs {
            values.push_str(a.render().as_str())
        }

        format!(
            "<node id=\"{}\" label=\"{}\">\\n<attvalues>{}</attvalues>\\n</node>",
            self.id.render(),
            self.label.render(),
            values
        )
    }
}

/// Converts a `Graph::Node` into some GEXF tags.
fn write_node<N>(node: &N, out: &mut File) -> Result<(), ExportError>
where
    N: GraphObject,
    N::Id: IntoGefxXml + Clone,
{
    let gexf_node = GexfNode {
        id: node.id().clone(),
        label: None,
        attrs: Vec::new(),
    };

    out.write_all(gexf_node.render().as_bytes())?;
    Ok(())
}

/// Converts a `Graph::Edge` into some GEXF tags.
fn write_edge<N, E>(edge: &EdgeReference<N, E>, out: &mut File) -> Result<(), ExportError>
where
    E: IntoGefxXml + Clone,
{
    Ok(())
}

/// Converts the `Graph` into some GEXF tags.
fn write_graph<G>(g: &G, out: &mut File) -> Result<(), ExportError>
where
    G: Graph,
    <G::Node as GraphObject>::Id: IntoGefxXml + Clone,
    <G::Edge as GraphObject>::Id: IntoGefxXml + Clone,
{
    let mut all_edges = Vec::new();

    out.write_all("<nodes>".as_bytes())?;

    for n in g.nodes() {
        write_node(n, out)?;
        all_edges.extend(g.neighbours(n.id()))
    }

    out.write_all("</nodes>\\n<edges>".as_bytes())?;

    for e in &all_edges {
        write_edge(e, out)?;
    }

    out.write_all("</edges>".as_bytes())?;

    Ok(())
}

pub fn export_graph<G>(g: &G, out: &Path) -> Result<(), ExportError>
where
    G: Graph,
    <G::Node as GraphObject>::Id: IntoGefxXml + Clone,
    <G::Edge as GraphObject>::Id: IntoGefxXml + Clone,
{
    let mut out_file = OpenOptions::new().write(true).create_new(true).open(out)?;

    out_file.write_all(GEXF_META.as_bytes())?;
    write_graph(g, &mut out_file)?;
    out_file.write_all(GEXF_FOOTER.as_bytes())?;

    Ok(())
}
