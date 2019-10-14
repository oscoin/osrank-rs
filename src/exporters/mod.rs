pub mod dot;
/// Exports a Graph into GEXF (Gephi Exchange Format).
pub mod gexf;
/// Exports a Graph into GraphML.
pub mod graphml;

use crate::types::network::ArtifactType;
use crate::types::Osrank;
use fraction::ToPrimitive;
use itertools::Itertools;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::iter::IntoIterator;
use std::marker::PhantomData;

pub trait Exporter {
    type ExporterOutput;
    type ExporterError;
    fn export(self) -> Result<Self::ExporterOutput, Self::ExporterError>;
}

pub enum CsvExporterError {
    IOError(std::io::Error),
}

impl From<std::io::Error> for CsvExporterError {
    fn from(err: std::io::Error) -> CsvExporterError {
        CsvExporterError::IOError(err)
    }
}

/// Given a (id,rank) iterator, write into a `.csv` file the (sorted) rank,
/// from the highest to the lowest.
pub fn export_rank_to_csv<K, V>(
    annotator: impl IntoIterator<Item = (K, V), IntoIter = <HashMap<K, V> as IntoIterator>::IntoIter>,
    out_path: &str,
) -> Result<(), CsvExporterError>
where
    V: ToPrimitive + PartialOrd + std::fmt::Display,
    K: std::fmt::Display,
{
    let mut output_csv = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out_path)?;

    for (node_id, rank) in annotator
        .into_iter()
        .sorted_by(|(_, v1), (_, v2)| v2.partial_cmp(v1).unwrap())
    {
        output_csv.write_all(format!("{} {:.32}\n", node_id, rank).as_str().as_bytes())?;
    }

    Ok(())
}

/// A rank for a node.
///
/// The `PhantomData` stores the type we need to convert _from_.
#[derive(Debug, Clone, Copy)]
pub struct Rank<T> {
    rank: f64,
    from_type: PhantomData<T>,
}

pub struct RgbColor {
    pub red: usize,
    pub green: usize,
    pub blue: usize,
}

/// Simple function to make a node bigger relative to its osrank.
pub fn size_from_rank(r: Rank<f64>) -> f64 {
    if r.rank <= 0.00005 {
        10.0
    } else {
        10.0 + (1000.0 * r.rank)
    }
}

// (adn) Compatibility shim necessary to be able to distinguish between projects
// and accounts. Hopefully it will go away as soon as we move to upstream
// `graph_api`.
pub enum NodeType {
    Project,
    Account,
}

impl std::convert::From<NodeType> for RgbColor {
    fn from(f: NodeType) -> Self {
        match f {
            NodeType::Project { .. } => RgbColor {
                red: 0,
                green: 0,
                blue: 255,
            },
            NodeType::Account { .. } => RgbColor {
                red: 255,
                green: 0,
                blue: 0,
            },
        }
    }
}

// Traits necessary to satisfy upstream constraints

impl std::convert::From<ArtifactType> for NodeType {
    fn from(atype: ArtifactType) -> Self {
        match atype {
            ArtifactType::Project { .. } => NodeType::Project,
            ArtifactType::Account { .. } => NodeType::Account,
        }
    }
}

impl std::convert::From<ArtifactType> for Rank<f64> {
    fn from(atype: ArtifactType) -> Self {
        Rank {
            rank: atype.get_osrank().to_f64().unwrap_or(0.0),
            from_type: PhantomData,
        }
    }
}

impl std::convert::From<Osrank> for Rank<f64> {
    fn from(r: Osrank) -> Self {
        Rank {
            rank: r.to_f64().unwrap_or(0.0),
            from_type: PhantomData,
        }
    }
}
