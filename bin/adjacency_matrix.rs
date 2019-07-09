extern crate clap;
extern crate failure;
extern crate ndarray;
extern crate num_traits;
extern crate osrank;
extern crate serde;
extern crate sprs;

use clap::{App, Arg};
use failure::Fail;
use osrank::types::{HyperParams, Weight};
use serde::Deserialize;
use sprs::binop::scalar_mul_mat;
use sprs::{hstack, vstack, CsMat, CsMatView, TriMat, TriMatBase};

use num_traits::{Num, One};
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
type SparseMatrix = CsMat<osrank::types::Weight>;
type DependencyMatrix = SparseMatrix;
type ContributionMatrix = SparseMatrix;
type MaintenanceMatrix = SparseMatrix;

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

/// Normalises the rows for the input matrix.
fn normalise_rows_mut<N>(matrix: &mut CsMat<N>)
where
    N: Num + Copy,
{
    for mut row_vec in matrix.outer_iterator_mut() {
        let mut ixs = Vec::new();
        let norm = row_vec.iter().fold(N::zero(), |acc, v| {
            ixs.push(v.0);
            acc + *(v.1)
        });
        if norm != N::zero() {
            for ix in ixs {
                row_vec[ix] = row_vec[ix] / norm;
            }
        }
    }
}

fn normalise_rows<'a, N>(matrix: &'a CsMat<N>) -> CsMatView<'a, N>
where
    N: Num + Copy,
{
    unimplemented!()
}

/// Creates a (sparse) adjacency matrix for the dependencies.
/// Corresponds to the "cargo-dep-adj.csv" from the Python scripts.
fn new_dependency_adjacency_matrix(
    deps_meta: &DependenciesMetadata,
    deps_csv: csv::Reader<File>,
) -> Result<DependencyMatrix, AppError> {
    let mut dep_adj: TriMat<Weight> = TriMatBase::new((deps_meta.ids.len(), deps_meta.ids.len()));

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
fn new_contribution_adjacency_matrix(
    contribs_meta: &ContributionsMetadata,
    contribs_csv: csv::Reader<File>,
) -> Result<ContributionMatrix, AppError> {
    // // Creates a sparse matrix of projects x contributors
    // let mut dep_adj: TriMat<u8> = TriMatBase::new((contribs_meta.contributors.len(), deps_meta.ids.len()));

    // // Iterate through the dependencies, populating the matrix.
    // for result in deps_csv.into_records().filter_map(|e| e.ok()) {
    //     let row: DepRow = result.deserialize(None)?;
    //     if let (Some(from_index), Some(to_index)) = (
    //         deps_meta.project2index.get(&row.from),
    //         deps_meta.project2index.get(&row.to),
    //     ) {
    //         dep_adj.add_triplet(*from_index as usize, *to_index as usize, 1);
    //     }
    // }

    // Ok(dep_adj.to_csr())

    unimplemented!()
}

fn new_network_matrix(
    dep_matrix: &DependencyMatrix,
    contrib_matrix: &ContributionMatrix,
    maintainer_matrix: &MaintenanceMatrix,
    hyperparams: HyperParams,
) -> Result<SparseMatrix, AppError> {
    let contrib_t = contrib_matrix.clone().transpose_into();
    let contrib_t_norm = normalise_rows(&contrib_t);
    let maintainer_t = maintainer_matrix.clone().transpose_into();
    let maintainer_norm = normalise_rows(&maintainer_matrix);

    let project_to_project =
        scalar_mul_mat(&normalise_rows(&dep_matrix), hyperparams.depend_factor);
    let project_to_account = &scalar_mul_mat(&maintainer_norm, hyperparams.maintain_factor)
        + &scalar_mul_mat(&normalise_rows(&contrib_matrix), hyperparams.contrib_factor);
    let account_to_project = &scalar_mul_mat(
        &(&maintainer_t * &contrib_t_norm),
        hyperparams.maintain_prime_factor,
    ) + &scalar_mul_mat(&contrib_t_norm, hyperparams.contrib_prime_factor);

    let account_to_account: SparseMatrix =
        CsMat::zero((contrib_matrix.cols(), contrib_matrix.cols()));

    // Join the matrixes together
    let q1_q2 = hstack(&vec![project_to_project.view(), project_to_account.view()]);
    let q3_q4 = hstack(&vec![account_to_project.view(), account_to_account.view()]);

    Ok(vstack(&vec![q1_q2.view(), q3_q4.view()]))
}

fn debug_sparse_matrix_to_csv(matrix: &SparseMatrix, out_path: &str) -> Result<(), AppError> {
    let mut output_csv = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out_path)?;

    for row in matrix.to_dense().genrows() {
        if let Some((last, els)) = row.as_slice().and_then(|e| e.split_last()) {
            for cell in els {
                output_csv.write_all(
                    format!("{},", (*cell).as_f64().unwrap())
                        .as_str()
                        .as_bytes(),
                )?;
            }
            output_csv.write_all(format!("{}", (*last).as_f64().unwrap()).as_str().as_bytes())?;
        }
        output_csv.write_all(b"\n")?;
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
    let mut contribs_csv = csv::Reader::from_reader(File::open(contrib_file)?);

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
    for result in contribs_csv.records().filter_map(|e| e.ok()) {
        let row: ContribRow = result.deserialize(None)?;
        let c = row.contributor.clone();
        contribs_meta.contributors.insert(row.contributor);
        contribs_meta
            .contributor2index
            .insert(c, contribs_meta.contributors.len() as u64 - 1);
    }

    //TODO(adn) For now the maintenance matrix is empty.

    let mut dep_adj_matrix = new_dependency_adjacency_matrix(&deps_meta, deps_csv)?;
    let mut con_adj_matrix = new_contribution_adjacency_matrix(&contribs_meta, contribs_csv)?;
    let mut maintenance_matrix = CsMat::zero((con_adj_matrix.cols(), con_adj_matrix.cols()));

    let network_matrix = new_network_matrix(
        &mut dep_adj_matrix,
        &mut con_adj_matrix,
        &mut maintenance_matrix,
        HyperParams::default(),
    )?;

    // Just for fun/debug: write this as a CSV file.
    debug_sparse_matrix_to_csv(&network_matrix, "data/cargo-all-adj.csv")?;

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
    use sprs::{CsMat, TriMat, TriMatBase};

    #[test]
    // Check that this implementation is the same as the Python script
    fn normalise_rows_equal_python() {
        let mut input = CsMat::csr_from_dense(
            arr2(&[[10., 20., 30.], [7., 8., 9.], [5., 9., 4.]]).view(),
            0.0,
        );

        super::normalise_rows_mut(&mut input);

        let expected = arr2(&[
            [0.16666666666666666, 0.3333333333333333, 0.5],
            [0.2916666666666667, 0.3333333333333333, 0.375],
            [0.2777777777777778, 0.5, 0.2222222222222222],
        ]);

        assert_eq!(input.to_dense(), expected);
    }

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
