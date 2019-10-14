pub mod dot;
/// Exports a Graph into GEXF (Gephi Exchange Format).
pub mod gexf;
/// Exports a Graph into GraphML.
pub mod graphml;

use crate::types::network::ArtifactType;
use crate::types::Osrank;
use fraction::ToPrimitive;
use std::marker::PhantomData;

pub trait Exporter {
    type ExporterOutput;
    type ExporterError;
    fn export_graph(self) -> Result<Self::ExporterOutput, Self::ExporterError>;
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
