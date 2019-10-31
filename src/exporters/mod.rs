/// Exports the rank of a graph into CSV.
pub mod csv;
pub mod dot;
/// Exports a Graph into GEXF (Gephi Exchange Format).
pub mod gexf;
/// Exports a Graph into GraphML.
pub mod graphml;

use crate::types::Osrank;
use fraction::ToPrimitive;
use oscoin_graph_api::types;
use std::marker::PhantomData;

pub trait Exporter {
    type ExporterOutput;
    type ExporterError;
    fn export(self) -> Result<Self::ExporterOutput, Self::ExporterError>;
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

impl std::convert::From<types::NodeType> for RgbColor {
    fn from(f: types::NodeType) -> Self {
        match f {
            types::NodeType::Project { .. } => RgbColor {
                red: 0,
                green: 0,
                blue: 255,
            },
            types::NodeType::User { .. } => RgbColor {
                red: 255,
                green: 0,
                blue: 0,
            },
        }
    }
}

// Traits necessary to satisfy upstream constraints
impl std::convert::Into<Rank<f64>> for types::NodeData<Osrank> {
    fn into(self) -> Rank<f64> {
        Rank {
            rank: self.rank.rank.to_f64().unwrap_or(0.0),
            from_type: PhantomData,
        }
    }
}

impl std::convert::Into<Rank<f64>> for Osrank {
    fn into(self) -> Rank<f64> {
        Rank {
            rank: self.to_f64().unwrap_or(0.0),
            from_type: PhantomData,
        }
    }
}
