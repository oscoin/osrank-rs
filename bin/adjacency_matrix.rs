extern crate clap;
extern crate failure;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate num_traits;
extern crate osrank;
extern crate serde;
extern crate sprs;

use clap::{App, Arg};
use failure::Fail;
use ndarray::Array2;
use ndarray_linalg::solve::Inverse;
use osrank::types::{HyperParams, Weight};
use serde::Deserialize;
use sprs::binop::{add_mat_same_storage, scalar_mul_mat};
use sprs::{hstack, vstack, CsMat, TriMat, TriMatBase};

use core::fmt::Debug;
use num_traits::{Num, One, Signed, Zero};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};

//
// Utility Traits
//

pub trait DisplayAsF64 {
    fn to_f64(self) -> f64;
}

impl DisplayAsF64 for f64 {
    fn to_f64(self: f64) -> f64 {
        self
    }
}

impl DisplayAsF64 for Weight {
    fn to_f64(self: Weight) -> f64 {
        self.as_f64().unwrap()
    }
}

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
type LocalMatrixIndex = usize;
type SparseMatrix<N> = CsMat<N>;
type DenseMatrix<N> = Array2<N>;
type DependencyMatrix<N> = SparseMatrix<N>;
type ContributionMatrix<N> = SparseMatrix<N>;
type MaintenanceMatrix<N> = SparseMatrix<N>;

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
    project_id: ProjectId,
    contributor: String,
    repo: String,
    contributions: u32,
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

/// Simple but naive implementation of pagerank that requires inverting an
/// nxn matrix.
/// See: http://michaelnielsen.org/blog/lectures-on-the-google-technology-stack-1-introduction-to-pagerank/
/// NOTE(adn) This function is not inside osrank::algorithm as it requires
/// external libraries and therefore might not be suitable for WASM compilation,
/// if ever needed.
pub fn pagerank_naive(
    g: &DenseMatrix<f64>,
    alpha: f64,
    outbound_links_factor: f64,
) -> DenseMatrix<f64> {
    let prop_visiting = 1.0 - alpha;
    let prop_teleporting = alpha;

    // Based on the formula:
    //
    // page_rank_matrix = prop_teleporting * ((I-prop_visiting*G)**-1)*((1.0/n)*ones(n,1))
    //

    let eye_matrix = Array2::eye(g.cols());
    let m1 = CsMat::csr_from_dense(
        (eye_matrix - (prop_visiting * g))
            .view()
            .inv()
            .expect("matrix inversion failed")
            .view(),
        0.0,
    );

    let e_matrix = scalar_mul_mat(
        &CsMat::csr_from_dense((Array2::ones((g.rows(), 1))).view(), 0.0),
        outbound_links_factor,
    );

    scalar_mul_mat(&(&m1 * &e_matrix), prop_teleporting).to_dense()
}

// Iterative algorithm taken from
// https://en.wikipedia.org/wiki/PageRank#Simplified_algorithm
pub fn pagerank_naive_iterative(
    dense: &DenseMatrix<f64>,
    damping_factor: f64,
    outbound_links_factor: f64,
) -> DenseMatrix<f64> {
    let prop_teleporting = 1.0 - damping_factor;

    let m = CsMat::csr_from_dense(dense.view(), 0.0);

    // At t = 0, the rank for all the nodes is the same.
    let mut rank = CsMat::csr_from_dense(
        (outbound_links_factor * Array2::ones((m.rows(), 1))).view(),
        0.0,
    );

    let e = CsMat::csr_from_dense(
        ((prop_teleporting / m.rows() as f64) * Array2::ones((m.rows(), 1))).view(),
        0.0,
    );

    let mut previous_rank = rank.clone();

    for _ix in 0..100 {
        rank = add_mat_same_storage(&scalar_mul_mat(&(&m * &rank), damping_factor), &e);

        if previous_rank == rank {
            break;
        }

        previous_rank = rank.clone()
    }

    rank.to_dense()
}

pub fn assert_rows_normalised<N>(matrix: &CsMat<N>, epsilon: N)
where
    N: One + Num + Copy + Debug + DisplayAsF64 + PartialOrd,
{
    for (row_ix, row_vec) in matrix.outer_iterator().enumerate() {
        let norm = row_vec.iter().fold(N::zero(), |acc, v| acc + *(v.1));

        let between_range = norm > N::one() - epsilon && norm < N::one() + epsilon;

        if norm != N::zero() && !between_range {
            panic!(
                "The matrix is not normalised correctly for {:#?}. Sum was: {:#?}",
                row_ix,
                norm.to_f64()
            );
        }
    }
}

pub fn assert_cols_normalised<N>(dense: &DenseMatrix<N>, epsilon: N)
where
    N: One + Num + Copy + Debug + DisplayAsF64 + PartialOrd,
{
    for col_ix in 0..dense.cols() {
        let norm = (0..dense.rows()).fold(N::zero(), |acc, ix| acc + dense[[ix, col_ix]]);
        let between_range = norm > N::one() - epsilon && norm < N::one() + epsilon;

        if !between_range {
            panic!(
                "The matrix is not normalised correctly for column {:#?}. Sum was: {:#?}",
                col_ix,
                norm.to_f64()
            );
        }
    }
}

/// Normalises the columns for the input (transposed) matrix in preparation
/// for pagerank, by turning each zero-column into probability distributions.
pub fn pagerank_normalise<N>(sparse: &SparseMatrix<N>, outbound_links_factor: N) -> DenseMatrix<N>
where
    N: Zero + PartialEq + Clone + Copy + Signed + PartialOrd,
{
    let mut dense = sparse.to_dense();
    let matrix_rows = dense.rows();

    for col_ix in 0..dense.cols() {
        let norm = (0..dense.rows()).fold(N::zero(), |acc, ix| acc + dense[[ix, col_ix]]);

        if norm == N::zero() {
            for ix in 0..matrix_rows {
                dense[[ix, col_ix]] = outbound_links_factor;
            }
        }
    }

    dense
}

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

/// Normalises the rows of a Matrix.
/// N.B. It returns a brand new Matrix, therefore it performs a copy.
/// FIXME(adn) Is there a way to yield only a (partial) view copying only
/// the values?
fn normalise_rows<N>(matrix: &SparseMatrix<N>) -> SparseMatrix<N>
where
    N: Num + Copy,
{
    let mut cloned = matrix.clone();
    normalise_rows_mut(&mut cloned);
    cloned
}

fn transpose_storage<N>(matrix: &SparseMatrix<N>) -> SparseMatrix<N>
where
    N: Num + Copy + Signed + PartialOrd + Default,
{
    CsMat::csr_from_dense(matrix.to_dense().t().view(), N::zero())
}

/// Creates a (sparse) adjacency matrix for the dependencies.
/// Corresponds to the "cargo-dep-adj.csv" from the Python scripts.
fn new_dependency_adjacency_matrix<N>(
    deps_meta: &DependenciesMetadata,
    deps_csv: csv::Reader<File>,
) -> Result<DependencyMatrix<N>, AppError>
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
/// NOTE(adn) We need to cleanup this matrix to make sure we end up with
/// a matrix such that:
/// 1. The number of rows and columns is equal to the dependency matrix
/// 2. The data occurs in the same logical order of the dependency matrix.
///    To say this differently, if the dependency matrix lists A,B,C as its
///    first projects, the contribution matrix will need to list A,B,C in
///    exactly the same order, without gaps.
fn new_contribution_adjacency_matrix<F, N>(
    deps_meta: &DependenciesMetadata,
    contribs_meta: &ContributionsMetadata,
    contribs_csv: csv::Reader<F>,
    mk_contribution: Box<dyn Fn(u32) -> N>,
) -> Result<ContributionMatrix<N>, AppError>
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

// Inefficient implementation of the Hadamard multiplication that operates
// on the dense representation.
fn hadamard_mul<N>(lhs: SparseMatrix<N>, rhs: &SparseMatrix<N>) -> SparseMatrix<N>
where
    N: Zero + PartialOrd + Signed + Clone,
{
    CsMat::csr_from_dense((lhs.to_dense() * rhs.to_dense()).view(), Zero::zero())
}

fn new_network_matrix<N>(
    dep_matrix: &DependencyMatrix<N>,
    contrib_matrix: &ContributionMatrix<N>,
    maintainer_matrix: &MaintenanceMatrix<N>,
    hyperparams: HyperParams,
) -> Result<SparseMatrix<N>, AppError>
where
    N: Num + Copy + Default + From<osrank::types::Weight> + PartialOrd + Signed,
{
    let contrib_t = transpose_storage(&contrib_matrix);
    let contrib_t_norm = normalise_rows(&contrib_t);

    let maintainer_t = maintainer_matrix.clone().transpose_into();
    let maintainer_norm = normalise_rows(&maintainer_matrix);

    let project_to_project = scalar_mul_mat(
        &normalise_rows(&dep_matrix),
        hyperparams.depend_factor.into(),
    );
    let project_to_account = &scalar_mul_mat(&maintainer_norm, hyperparams.maintain_factor.into())
        + &scalar_mul_mat(
            &normalise_rows(&contrib_matrix),
            hyperparams.contrib_factor.into(),
        );
    let account_to_project =
        &hadamard_mul(
            scalar_mul_mat(&maintainer_t, hyperparams.maintain_prime_factor.into()),
            &contrib_t_norm,
        ) + &scalar_mul_mat(&contrib_t_norm, hyperparams.contrib_prime_factor.into());

    let account_to_account: SparseMatrix<N> =
        CsMat::zero((contrib_matrix.cols(), contrib_matrix.cols()));

    // Join the matrixes together
    let q1_q2 = hstack(&vec![project_to_project.view(), project_to_account.view()]);
    let q3_q4 = hstack(&vec![account_to_project.view(), account_to_account.view()]);

    Ok(normalise_rows(&vstack(&vec![q1_q2.view(), q3_q4.view()])))
}

fn debug_sparse_matrix_to_csv<N>(matrix: &CsMat<N>, out_path: &str) -> Result<(), AppError>
where
    N: DisplayAsF64 + Zero + Clone + Copy,
{
    debug_dense_matrix_to_csv(&matrix.to_dense(), out_path)
}

fn debug_dense_matrix_to_csv<N>(matrix: &DenseMatrix<N>, out_path: &str) -> Result<(), AppError>
where
    N: DisplayAsF64 + Zero + Clone + Copy,
{
    let mut output_csv = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out_path)?;

    for row in matrix.genrows() {
        if let Some((last, els)) = row.as_slice().and_then(|e| e.split_last()) {
            for cell in els {
                output_csv.write_all(format!("{},", cell.to_f64()).as_str().as_bytes())?;
            }
            output_csv.write_all(format!("{}", last.to_f64()).as_str().as_bytes())?;
        }
        output_csv.write_all(b"\n")?;
    }

    Ok(())
}

fn debug_pagerank_to_csv(matrix: &DenseMatrix<f64>, out_path: &str) -> Result<(), AppError> {
    let mut output_csv = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out_path)?;

    for row in matrix.genrows() {
        if let Some((last, els)) = row.as_slice().and_then(|e| e.split_last()) {
            for cell in els {
                output_csv.write_all(format!("{:.32},", cell).as_str().as_bytes())?;
            }
            output_csv.write_all(format!("{:.32}", last).as_str().as_bytes())?;
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
    let contribs_csv_first_pass = csv::Reader::from_reader(File::open(contrib_file)?);

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
            .insert(row.id, deps_meta.ids.len() - 1);
    }

    // Iterate once over the contributions and build a matrix where
    // rows are the project names and columns the (unique) contributors.
    for result in contribs_csv_first_pass
        .into_records()
        .filter_map(|e| e.ok())
    {
        let row: ContribRow = result.deserialize(None)?;
        let c = row.contributor.clone();
        contribs_meta.contributors.insert(row.contributor);
        contribs_meta
            .contributor2index
            .insert(c, contribs_meta.contributors.len() - 1);
    }

    //"Rewind" the file as we need a second pass.
    let contribs_csv = csv::Reader::from_reader(File::open(contrib_file)?);

    //TODO(adn) For now the maintenance matrix is empty.

    println!("Assembling the dependency matrix...");
    let dep_adj_matrix = new_dependency_adjacency_matrix(&deps_meta, deps_csv)?;
    println!(
        "Generated a matrix of {}x{}",
        dep_adj_matrix.rows(),
        dep_adj_matrix.cols()
    );
    println!("Assembling the contribution matrix...");
    let con_adj_matrix = new_contribution_adjacency_matrix(
        &deps_meta,
        &contribs_meta,
        contribs_csv,
        Box::new(f64::from),
    )?;
    println!(
        "Generated a matrix of {}x{}",
        con_adj_matrix.rows(),
        con_adj_matrix.cols()
    );
    let maintenance_matrix = CsMat::zero((dep_adj_matrix.rows(), con_adj_matrix.cols()));

    println!("Assembling the network matrix...");
    let network_matrix = new_network_matrix(
        &dep_adj_matrix,
        &con_adj_matrix,
        &maintenance_matrix,
        HyperParams::default(),
    )?;

    println!("assert_normalised(network_matrix)");
    assert_rows_normalised(&network_matrix, 0.0001);

    println!(
        "Generated a matrix of {}x{}",
        network_matrix.rows(),
        network_matrix.cols()
    );

    let outbound_links_factor = 1.0 / f64::from(network_matrix.rows() as u32);

    println!("outbound_links_factor: {:.32}", outbound_links_factor);

    println!("Transposing the network matrix...");
    let network_matrix_t = transpose_storage(&network_matrix);
    println!("Normalise into a probability distribution...");
    let network_t_norm = pagerank_normalise(&network_matrix_t, outbound_links_factor);

    assert_cols_normalised(&network_t_norm, 0.0001);

    println!("Computing the pagerank...");
    let pagerank_matrix = pagerank_naive_iterative(&network_t_norm, 0.85, outbound_links_factor);

    println!("Write the matrix to file (skipped for now)");
    // Just for fun/debug: write this as a CSV file.
    debug_pagerank_to_csv(&pagerank_matrix, "data/cargo-page-rank.csv")?;

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
    use csv;
    use ndarray::arr2;
    use num_traits::{One, Zero};
    use osrank::types::{HyperParams, Weight};
    use pretty_assertions::assert_eq;
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
    // Check that this implementation is the same as the Python script
    fn hadamard_mul_works() {
        let input = CsMat::csr_from_dense(
            arr2(&[[8, 4, 1, 0], [4, 0, 8, 5], [2, 4, 6, 5], [2, 2, 2, 5]]).view(),
            0,
        );

        let input2 = input.clone();
        let result = super::hadamard_mul(input, &input2);

        let expected = arr2(&[
            [64, 16, 1, 0],
            [16, 0, 64, 25],
            [4, 16, 36, 25],
            [4, 4, 4, 25],
        ]);

        assert_eq!(result.to_dense(), expected);

        // Assert that it is NOT the same as the one provided by `sprs`.

        let input3 = CsMat::csr_from_dense(
            arr2(&[[8, 4, 1, 0], [4, 0, 8, 5], [2, 4, 6, 5], [2, 2, 2, 5]]).view(),
            0,
        );

        let result2 = &input3 * &input3;

        assert_ne!(result2.to_dense(), expected);
    }

    #[test]
    fn transpose_works() {
        let z = Zero::zero();

        let c = CsMat::csr_from_dense(
            arr2(&[
                [Weight::new(100, 1), z, z],
                [z, Weight::new(30, 1), z],
                [z, Weight::new(60, 1), Weight::new(20, 1)],
            ])
            .view(),
            z,
        );

        // Transpose the matrix
        let actual = c.clone().transpose_into();

        let expected = arr2(&[
            [Weight::new(100, 1), z, z],
            [z, Weight::new(30, 1), Weight::new(60, 1)],
            [z, z, Weight::new(20, 1)],
        ]);

        assert_eq!(actual.to_dense(), expected);
    }

    #[test]
    // Check that we can normalise by rows after a transposition.
    fn normalise_after_transpose() {
        let z = Zero::zero();
        let o = One::one();

        let c = CsMat::csr_from_dense(
            arr2(&[
                [Weight::new(100, 1), z, z],
                [z, Weight::new(30, 1), z],
                [z, Weight::new(60, 1), Weight::new(20, 1)],
            ])
            .view(),
            z,
        );

        // Transpose the matrix and normalise it.

        let actual = super::normalise_rows(&super::transpose_storage(&c));

        let expected = arr2(&[
            [o, z, z],
            [z, Weight::new(1, 3), Weight::new(2, 3)],
            [z, z, o],
        ]);

        assert_eq!(actual.to_dense(), expected);
    }

    #[test]
    // Check that this implementation is the same as the one described in
    // the basic model.
    fn new_network_equal_basic_model() {
        let z = Zero::zero();
        let o = One::one();
        let d = CsMat::csr_from_dense(arr2(&[[z, o, z], [z, z, z], [o, o, z]]).view(), z);

        let c = CsMat::csr_from_dense(
            arr2(&[
                [Weight::new(100, 1), z, z],
                [z, Weight::new(30, 1), z],
                [z, Weight::new(60, 1), Weight::new(20, 1)],
            ])
            .view(),
            z,
        );
        let m = CsMat::csr_from_dense(arr2(&[[o, z, z], [z, o, z], [z, o, z]]).view(), z);

        let network = super::new_network_matrix(&d, &c, &m, HyperParams::default()).unwrap();

        let expected = arr2(&[
            [z, Weight::new(4, 7), z, Weight::new(3, 7), z, z],
            [z, z, z, z, o, z],
            [
                Weight::new(2, 7),
                Weight::new(2, 7),
                z,
                z,
                Weight::new(11, 28),
                Weight::new(1, 28),
            ],
            [o, z, z, z, z, z],
            [z, Weight::new(1, 3), Weight::new(2, 3), z, z, z],
            [z, z, o, z, z, z],
        ]);

        assert_eq!(network.to_dense(), expected);
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

    #[test]
    // Tests which simulates the following situation:
    // * We have 3 projects: oscoin, osrank and radicle;
    // * We have 3 contributors: foo, bar, baz;
    // * foo & bar both contributes to oscoin;
    // * baz contributes only to osrank;
    // * The radicle project has no contributions.
    fn test_contribution_matrix() {
        let contrib_csv: String = String::from(
            r###"
ID,MAINTAINER,REPO,CONTRIBUTIONS,NAME
10,github@foo,https://github.com/oscoin/oscoin,118,oscoin
10,github@bar,https://github.com/oscoin/ocoin,32,oscoin
15,github@baz,https://github.com/oscoin/osrank,10,osrank
"###,
        );

        let dep_meta = super::DependenciesMetadata {
            ids: [10, 15, 7].iter().cloned().collect(),
            labels: [
                String::from("oscoin"),
                String::from("osrank"),
                String::from("radicle"),
            ]
            .iter()
            .cloned()
            .collect(),
            project2index: [(10, 0), (15, 1), (7, 2)].iter().cloned().collect(),
        };

        let contribs = super::ContributionsMetadata {
            contributors: [
                String::from("github@foo"),
                String::from("github@bar"),
                String::from("github@baz"),
            ]
            .iter()
            .cloned()
            .collect(),
            contributor2index: [
                (String::from("github@foo"), 0 as usize),
                (String::from("github@bar"), 1 as usize),
                (String::from("github@baz"), 2 as usize),
            ]
            .iter()
            .cloned()
            .collect(),
        };

        let contribs_records = csv::ReaderBuilder::new()
            .flexible(true)
            .from_reader(contrib_csv.as_bytes());

        let actual = super::new_contribution_adjacency_matrix(
            &dep_meta,
            &contribs,
            contribs_records,
            Box::new(|c| Weight::new(c, 1)),
        )
        .unwrap();

        let expected = arr2(&[
            [Weight::new(118, 1), Weight::new(32, 1), Weight::zero()],
            [Weight::zero(), Weight::zero(), Weight::new(10, 1)],
            [Weight::zero(), Weight::zero(), Weight::zero()],
        ]);

        assert_eq!(actual.to_dense(), expected);
    }

    #[test]
    fn pagerank_naive_f64() {
        let input = arr2(&[[0.5, 0.5, 0.], [0.5, 0., 0.], [0., 0.5, 1.0]]);
        let alpha = 0.15;
        let outbound_links_factor = 1.0 / 3.0;

        let actual = super::pagerank_naive(&input, alpha, outbound_links_factor);

        let expected = arr2(&[
            [0.18066561014263074],
            [0.12678288431061807],
            [0.6925515055467512],
        ]);

        assert_eq!(actual, expected);
    }

    #[test]
    fn pagerank_naive_iterative_f64() {
        let input = arr2(&[[0.5, 0.5, 0.], [0.5, 0., 0.], [0., 0.5, 1.0]]);
        let alpha = 0.85;
        let outbound_links_factor = 1.0 / 3.0;

        let actual = super::pagerank_naive_iterative(&input, alpha, outbound_links_factor);

        let expected = arr2(&[
            [0.18066561014263083],
            [0.1267828843106181],
            [0.6925515055467515],
        ]);

        assert_eq!(actual, expected);
    }

    #[test]
    fn pagerank_normalise_works() {
        let z = Weight::zero();
        let o = Weight::one();
        let input = CsMat::csr_from_dense(
            arr2(&[
                [z, z, z],
                [Weight::new(1, 2), z, Weight::new(1, 2)],
                [o, z, z],
            ])
            .t(),
            Weight::zero(),
        );

        let transposed = arr2(&[
            [z, Weight::new(1, 2), o],
            [z, z, z],
            [z, Weight::new(1, 2), z],
        ]);

        assert_eq!(input.to_dense(), transposed);

        let actual = super::pagerank_normalise(&input, Weight::new(1, 3));

        let expected = arr2(&[
            [Weight::new(1, 3), Weight::new(1, 2), o],
            [Weight::new(1, 3), z, z],
            [Weight::new(1, 3), Weight::new(1, 2), z],
        ]);

        assert_eq!(actual, expected);
    }
}
