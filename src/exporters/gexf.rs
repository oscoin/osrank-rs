#![allow(unknown_lints)]
#![warn(clippy::all)]

use crate::protocol_traits::graph::{EdgeReference, Graph, GraphObject};

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

static GEXF_META: &str = r###"<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.1draft" version="1.1" xmlns:viz="http://www.gexf.net/1.1draft/viz" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.gexf.net/1.1draft http://www.gexf.net/1.1draft/gexf.xsd">
  <meta lastmodifieddate="2011-09-05">
    <creator>Gephi 0.8</creator>
    <description></description>
  </meta>
  <graph defaultedgetype="directed">
  <attributes class="node" mode="static">
    <attribute id="modularity_class" title="Modularity Class" type="integer">
      <default>0</default>
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

impl IntoGefxXml for usize {
    fn render(&self) -> String {
        format!("{}", self)
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
            r###"<node id="{}" label="{}">
        <viz:size value="10"></viz:size>
        <viz:color r="168" g="168" b="29"></viz:color>
        <attvalues>{}</attvalues>
        </node>"###,
            self.id.render(),
            self.label.render(),
            values
        )
    }
}

struct GexfEdge<I, N> {
    id: I,
    source: N,
    target: N,
}

impl<I, N> IntoGefxXml for GexfEdge<I, N>
where
    I: IntoGefxXml,
    N: IntoGefxXml,
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
fn write_node<N>(node: &N, out: &mut File) -> Result<(), ExportError>
where
    N: GraphObject,
    N::Id: IntoGefxXml + Clone,
{
    let gexf_node = GexfNode {
        id: node.id().clone(),
        label: Some("test".to_string()),
        attrs: Vec::new(),
    };

    out.write_all(gexf_node.render().as_bytes())?;
    Ok(())
}

/// Converts a `Graph::Edge` into some GEXF tags.
fn write_edge<N, E>(edge: &EdgeReference<N, E>, out: &mut File) -> Result<(), ExportError>
where
    E: IntoGefxXml + Clone,
    N: IntoGefxXml + Clone,
{
    let gexf_edge = GexfEdge {
        id: edge.id.clone(),
        source: edge.source.clone(),
        target: edge.target.clone(),
    };

    out.write_all(gexf_edge.render().as_bytes())?;
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

    out.write_all(b"<nodes>\n")?;

    for n in g.nodes() {
        write_node(n, out)?;
        out.write_all(b"\n")?;
        all_edges.extend(g.neighbours(n.id()))
    }

    out.write_all(b"</nodes>\n<edges>")?;

    for e in &all_edges {
        write_edge(e, out)?;
        out.write_all(b"\n")?;
    }

    out.write_all(b"</edges>\n</graph>")?;

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
