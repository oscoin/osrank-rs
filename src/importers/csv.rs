#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate num_traits;
extern crate serde;
extern crate sprs;

use crate::linalg::SparseMatrix;
use crate::protocol_traits::graph::Graph;
use crate::protocol_traits::ledger::LedgerView;
use core::fmt;
use num_traits::{Num, One};
use serde::Deserialize;
use sprs::{TriMat, TriMatBase};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;

//
// Types
//

type ProjectId = u32;
type ProjectName = String;
type Contributor = String;
// We need a way to map external indexes with an incremental index suitable
// for matrix manipulation, so we define this LocalMatrixIndex exactly for
// this purpose.
type LocalMatrixIndex = usize;

pub type DependencyMatrix<N> = SparseMatrix<N>;
pub type ContributionMatrix<N> = SparseMatrix<N>;
pub type MaintenanceMatrix<N> = SparseMatrix<N>;

#[derive(Debug, Deserialize)]
pub struct DepMetaRow {
    pub id: u32,
    pub name: String,
    pub platform: String,
}

pub struct DependenciesMetadata {
    pub ids: HashSet<ProjectId>,
    pub labels: Vec<ProjectName>,
    pub project2index: HashMap<ProjectId, LocalMatrixIndex>,
}

impl DependenciesMetadata {
    pub fn new() -> Self {
        DependenciesMetadata {
            ids: Default::default(),
            labels: Default::default(),
            project2index: Default::default(),
        }
    }
}

pub struct ContributionsMetadata {
    pub contributors: HashSet<Contributor>,
    pub contributor2index: HashMap<Contributor, LocalMatrixIndex>,
}

impl ContributionsMetadata {
    pub fn new() -> Self {
        ContributionsMetadata {
            contributors: Default::default(),
            contributor2index: Default::default(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ContribRow {
    pub project_id: ProjectId,
    pub contributor: String,
    pub repo: String,
    pub contributions: u32,
    pub project_name: ProjectName,
}

#[derive(Debug, Deserialize)]
pub struct DepRow {
    pub from: ProjectId,
    pub to: ProjectId,
}

//
// Errors
//

#[derive(Debug)]
pub enum CsvImportError {
    // Returned in case of generic I/O error.
    IOError(std::io::Error),

    // Returned when the CSV deserialisation failed.
    CsvDeserialisationError(csv::Error),
}

impl fmt::Display for CsvImportError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CsvImportError::IOError(e) => {
                write!(f, "i/o error when reading/writing on the CSV file {}", e)
            }
            CsvImportError::CsvDeserialisationError(e) => {
                write!(f, "Deserialisation failed on a CSV row {}", e)
            }
        }
    }
}

impl From<std::io::Error> for CsvImportError {
    fn from(err: std::io::Error) -> CsvImportError {
        CsvImportError::IOError(err)
    }
}

impl From<csv::Error> for CsvImportError {
    fn from(err: csv::Error) -> CsvImportError {
        CsvImportError::CsvDeserialisationError(err)
    }
}

pub fn from_csv_files<G, L>(
    deps_csv_file: &Path,
    deps_meta_csv_file: &Path,
    contrib_csv_file: &Path,
    ledger_view: &L,
) -> Result<G, CsvImportError>
where
    G: Graph,
    L: LedgerView,
{
    unimplemented!()
}

/// Creates a (sparse) adjacency matrix for the dependencies.
/// Corresponds to the "cargo-dep-adj.csv" from the Python scripts.
pub fn new_dependency_adjacency_matrix<N>(
    deps_meta: &DependenciesMetadata,
    deps_csv: csv::Reader<File>,
) -> Result<DependencyMatrix<N>, CsvImportError>
where
    N: Num + Clone,
{
    let mut dep_adj: TriMat<N> = TriMatBase::new((deps_meta.ids.len(), deps_meta.ids.len()));

    // Iterate through the dependencies, populating the matrix.
    for result in deps_csv.into_records().filter_map(|e| e.ok()) {
        let row: DepRow = result.deserialize(None)?;
        if let (Some(from_index), Some(to_index)) = (
            deps_meta.project2index.get(&row.from),
            deps_meta.project2index.get(&row.to),
        ) {
            dep_adj.add_triplet(*from_index as usize, *to_index as usize, One::one());
        }
    }

    Ok(dep_adj.to_csr())
}

/// Creates a (sparse) adjacency matrix for the contributions.
/// Corresponds to the "cargo-contrib-adj.csv" from the Python scripts.
pub fn new_contribution_adjacency_matrix<F, N>(
    deps_meta: &DependenciesMetadata,
    contribs_meta: &ContributionsMetadata,
    contribs_csv: csv::Reader<F>,
    mk_contribution: Box<dyn Fn(u32) -> N>,
) -> Result<ContributionMatrix<N>, CsvImportError>
where
    F: Read,
    N: Num + Clone,
{
    // Creates a sparse matrix of projects x contributors
    let mut contrib_adj: TriMat<N> =
        TriMatBase::new((deps_meta.ids.len(), contribs_meta.contributors.len()));

    for result in contribs_csv.into_records().filter_map(|e| e.ok()) {
        let row: ContribRow = result.deserialize(None)?;

        if let Some((row_ix, col_ix)) =
            deps_meta
                .project2index
                .get(&row.project_id)
                .and_then(|row_ix| {
                    contribs_meta
                        .contributor2index
                        .get(&row.contributor)
                        .map(|col| (*row_ix, *col))
                })
        {
            contrib_adj.add_triplet(row_ix, col_ix, mk_contribution(row.contributions))
        }
    }

    Ok(contrib_adj.to_csr())
}
