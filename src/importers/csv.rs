#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate csv;
extern crate num_traits;
extern crate serde;
extern crate sprs;

use crate::adjacency::new_network_matrix;
use crate::linalg::SparseMatrix;
use crate::protocol_traits::graph::Graph;
use crate::protocol_traits::ledger::LedgerView;
use crate::types::{AccountAttributes, Artifact, Dependency, Network, ProjectAttributes};
use core::fmt;
use num_traits::{Num, One, Zero};
use serde::Deserialize;
use sprs::{CsMat, TriMat, TriMatBase};
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

impl Default for DependenciesMetadata {
    fn default() -> Self {
        DependenciesMetadata::new()
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

impl Default for ContributionsMetadata {
    fn default() -> Self {
        ContributionsMetadata::new()
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

pub fn import_network<G, L>(
    deps_csv_file: &Path,
    deps_meta_csv_file: &Path,
    contrib_csv_file: &Path,
    //TODO(and) We want to consider maintainers at some point.
    _maintainers_csv_file: Option<&Path>,
    ledger_view: &L,
) -> Result<Network<f64>, CsvImportError>
where
    L: LedgerView,
{
    let deps_csv = csv::Reader::from_reader(File::open(deps_csv_file)?);
    let deps_meta_csv = csv::Reader::from_reader(File::open(deps_meta_csv_file)?);
    let contribs_csv_first_pass = csv::Reader::from_reader(File::open(contrib_csv_file)?);
    let mut deps_meta = DependenciesMetadata::new();
    let mut contribs_meta = ContributionsMetadata::new();

    let mut graph = Network::default();

    // Iterate once over the dependencies metadata and store the name and id.
    // We need to maintain some sort of mapping between the order of visit
    // (which will be used as the index in the matrix) and the project id.
    for result in deps_meta_csv.into_records().filter_map(|e| e.ok()) {
        let row: DepMetaRow = result.deserialize(None)?;
        let prj_id = row.name.clone();
        deps_meta.ids.insert(row.id);
        deps_meta.labels.push(row.name);
        deps_meta
            .project2index
            .insert(row.id, deps_meta.ids.len() - 1);

        // Add the projects as nodes in the graph.
        graph.add_node(Artifact::Project(ProjectAttributes {
            id: prj_id,
            osrank: Zero::zero(),
        }));
    }

    // Iterate once over the contributions and build a matrix where
    // rows are the project names and columns the (unique) contributors.
    for result in contribs_csv_first_pass
        .into_records()
        .filter_map(|e| e.ok())
    {
        let row: ContribRow = result.deserialize(None)?;
        let c = row.contributor.clone();
        let contrib_id = row.contributor.clone();

        if contribs_meta.contributors.get(&row.contributor).is_none() {
            contribs_meta.contributors.insert(row.contributor);
            contribs_meta
                .contributor2index
                .insert(c, contribs_meta.contributors.len() - 1);

            graph.add_node(Artifact::Account(AccountAttributes {
                id: contrib_id.to_string(),
                osrank: Zero::zero(),
            }));
        }
    }

    //"Rewind" the file as we need a second pass.
    let contribs_csv = csv::Reader::from_reader(File::open(contrib_csv_file)?);

    let dep_adj_matrix = new_dependency_adjacency_matrix(&deps_meta, deps_csv)?;
    let con_adj_matrix = new_contribution_adjacency_matrix(
        &deps_meta,
        &contribs_meta,
        contribs_csv,
        Box::new(f64::from),
    )?;

    //FIXME(adn) For now the maintenance matrix is empty.
    let maintainers_matrix = CsMat::zero((dep_adj_matrix.rows(), con_adj_matrix.cols()));

    let network_matrix = new_network_matrix(
        &dep_adj_matrix,
        &con_adj_matrix,
        &maintainers_matrix,
        &ledger_view.get_hyperparams(),
    );

    //FIXME(adn) Here we have a precision problem: we _have_ to convert the
    //weights from fractions to f64 to avoid arithmetic overflows, but yet here
    //it's nice to work with fractions.
    for (source, row_vec) in network_matrix.outer_iterator().enumerate() {
        for (target, weight) in row_vec.iter().enumerate() {
            graph.add_edge(source, target, Dependency::Influence(*weight.1))
        }
    }

    // Build a graph out of the matrix.
    Ok(graph)
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
