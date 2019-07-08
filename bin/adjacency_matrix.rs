extern crate clap;
extern crate failure;
extern crate ndarray;
extern crate serde;
extern crate sprs;

use clap::{App, Arg};
use failure::Fail;
use serde::Deserialize;
use sprs::{CsMat, TriMat, TriMatBase};

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::Write;

//
// Types
//

#[derive(Debug, Fail)]
enum AppError {
    // Returned in case of generic I/O error.
    #[fail(display = "i/o error when reading/writing on the CSV file {}", _0)]
    IOError(std::io::Error),

    // Returned when the CSV deserialisation failed.
    #[fail(display = "Deserialisation failed on a CSV row {}", _0)]
    CsvDeserialisationError(csv::Error),
}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> AppError {
        AppError::IOError(err)
    }
}

impl From<csv::Error> for AppError {
    fn from(err: csv::Error) -> AppError {
        AppError::CsvDeserialisationError(err)
    }
}

type ProjectId = u32;
type ProjectName = String;
type Contributor = String;
// We need a way to map external indexes with an incremental index suitable
// for matrix manipulation, so we define this LocalMatrixIndex exactly for
// this purpose.
type LocalMatrixIndex = u64;

#[derive(Debug, Deserialize)]
struct DepMetaRow {
    id: u32,
    name: String,
    platform: String,
}

struct DependenciesMetadata {
    ids: HashSet<ProjectId>,
    labels: HashSet<ProjectName>,
    project2index: HashMap<ProjectId, LocalMatrixIndex>,
}

impl DependenciesMetadata {
    fn new() -> Self {
        DependenciesMetadata {
            ids: Default::default(),
            labels: Default::default(),
            project2index: Default::default(),
        }
    }
}

struct ContributionsMetadata {
    contributors: HashSet<Contributor>,
    contributor2index: HashMap<Contributor, LocalMatrixIndex>,
}

impl ContributionsMetadata {
    fn new() -> Self {
        ContributionsMetadata {
            contributors: Default::default(),
            contributor2index: Default::default(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ContribRow {
    contributor: String,
    repo: String,
    contributions: u64,
    project_name: ProjectName,
}

#[derive(Debug, Deserialize)]
struct DepRow {
    from: ProjectId,
    to: ProjectId,
}

//
// Functions
//

/// Creates a (sparse) adjacency matrix for the dependencies.
/// Corresponds to the "cargo-dep-adj.csv" from the Python scripts.
fn new_dependency_adjacency_matrix(
    deps_meta: &DependenciesMetadata,
    deps_csv: csv::Reader<File>,
) -> Result<CsMat<u8>, AppError> {
    let mut dep_adj: TriMat<u8> = TriMatBase::new((deps_meta.ids.len(), deps_meta.ids.len()));

    // Iterate through the dependencies, populating the matrix.
    for result in deps_csv.into_records().filter_map(|e| e.ok()) {
        let row: DepRow = result.deserialize(None)?;
        match (
            deps_meta.project2index.get(&row.from),
            deps_meta.project2index.get(&row.to),
        ) {
            (Some(from_index), Some(to_index)) => {
                dep_adj.add_triplet(*from_index as usize, *to_index as usize, 1);
            }
            _ => (),
        }
    }

    Ok(dep_adj.to_csr())
}

/// Creates a (sparse) adjacency matrix for the contributions.
/// Corresponds to the "cargo-contrib-adj.csv" from the Python scripts.
fn new_contribution_adjacency_matrix() -> Result<CsMat<u8>, AppError> {
    unimplemented!()
}

fn debug_sparse_matrix_to_csv(matrix: &CsMat<u8>, out_path: &str) -> Result<(), AppError> {
    let mut output_csv = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out_path)?;

    for row in matrix.to_dense().genrows() {
        if let Some((last, els)) = row.as_slice().and_then(|e| e.split_last()) {
            for cell in els {
                output_csv.write(format!("{},", *cell).as_str().as_bytes())?;
            }
            output_csv.write(format!("{}", *last).as_str().as_bytes())?;
        }
        output_csv.write(b"\n")?;
    }

    Ok(())
}

/// Naive porting of the Python algorithm.
fn build_adjacency_matrix(
    deps_file: &str,
    deps_meta_file: &str,
    contrib_file: &str,
    _out_path: &str,
) -> Result<(), AppError> {
    let deps_csv = csv::Reader::from_reader(File::open(deps_file)?);
    let deps_meta_csv = csv::Reader::from_reader(File::open(deps_meta_file)?);
    let contribs_csv = csv::Reader::from_reader(File::open(contrib_file)?);

    let mut deps_meta = DependenciesMetadata::new();
    let mut contribs_meta = ContributionsMetadata::new();

    // Iterate once over the dependencies metadata and store the name and id.
    // We need to maintain some sort of mapping between the order of visit
    // (which will be used as the index in the matrix) and the project id.
    for result in deps_meta_csv.into_records().filter_map(|e| e.ok()) {
        let row: DepMetaRow = result.deserialize(None)?;
        deps_meta.ids.insert(row.id);
        deps_meta.labels.insert(row.name);
        deps_meta
            .project2index
            .insert(row.id, deps_meta.ids.len() as u64 - 1);
    }

    // Iterate once over the contributions and build a matrix where
    // rows are the project names and columns the (unique) contributors.
    for result in contribs_csv.into_records().filter_map(|e| e.ok()) {
        let row: ContribRow = result.deserialize(None)?;
        let c = row.contributor.clone();
        contribs_meta.contributors.insert(row.contributor);
        contribs_meta
            .contributor2index
            .insert(c, contribs_meta.contributors.len() as u64 - 1);
    }

    // Create the dependency matrix (data/cargo-dep-adj.csv in the original
    // script)
    let dep_adj_matrix = new_dependency_adjacency_matrix(&deps_meta, deps_csv)?;
    // let con_adj_matrix = new_contribution_adjacency_matrix()?;

    // Just for fun/debug: write this as a CSV file.
    debug_sparse_matrix_to_csv(&dep_adj_matrix, "data/cargo-dep-adj.csv")?;

    Ok(())
}

fn main() -> Result<(), AppError> {
    let matches = App::new("Port of the adjacency matrix calculation from Jupyter")
        .arg(
            Arg::with_name("dependencies")
                .long("deps")
                .help("Path to the <platform>_dependencies.csv file")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("dependencies-with-metadata")
                .long("deps-meta")
                .help("Path to the <platform>_dependencies_meta.csv file")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("contributions")
                .long("contribs")
                .help("Path to the <platform>_contributions.csv file")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("output-path")
                .short("o")
                .help("Path to the output .csv file")
                .takes_value(true)
                .required(true),
        )
        .get_matches();

    build_adjacency_matrix(
        matches
            .value_of("dependencies")
            .expect("dependencies csv file not given."),
        matches
            .value_of("dependencies-with-metadata")
            .expect("dependencies with metadata csv file not given."),
        matches
            .value_of("contributions")
            .expect("contributions csv file not given."),
        matches
            .value_of("output-path")
            .expect("output csv file not specified."),
    )
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;
    use sprs::{TriMat, TriMatBase};

    #[test]
    // See: https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29
    fn sparse_matrixes_ok() {
        let mut dep_adj: TriMat<u8> = TriMatBase::new((4, 4));

        dep_adj.add_triplet(1, 0, 5);
        dep_adj.add_triplet(1, 1, 8);
        dep_adj.add_triplet(2, 2, 3);
        dep_adj.add_triplet(3, 1, 6);

        // Dense matrix
        let dense = arr2(&[[0, 0, 0, 0], [5, 8, 0, 0], [0, 0, 3, 0], [0, 6, 0, 0]]);

        assert_eq!(dense, dep_adj.to_csr().to_dense());
    }
}
