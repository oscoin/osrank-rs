#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate clap;
extern crate failure;
extern crate ndarray;
extern crate num_traits;
extern crate osrank;
extern crate serde;
extern crate sprs;

use clap::{App, Arg};
use itertools::Itertools;
use ndarray::Array2;
use osrank::adjacency::new_network_matrix;
use osrank::collections::{Rank, WithLabels};
use osrank::importers::csv::{
    new_contribution_adjacency_matrix, new_dependency_adjacency_matrix, ContribRow,
    ContributionsMetadata, CsvImportError, DepMetaRow, DependenciesMetadata,
};
use osrank::linalg::{transpose_storage, DenseMatrix, SparseMatrix};
use osrank::types::{HyperParams, Weight};
use sprs::binop::{add_mat_same_storage, scalar_mul_mat};
use sprs::CsMat;

use core::fmt::Debug;
use num_traits::{Num, One, Signed, Zero};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::rc::Rc;

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
// Functions
//

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

pub fn debug_sparse_matrix_to_csv<N>(
    matrix: &CsMat<N>,
    out_path: &str,
) -> Result<(), CsvImportError>
where
    N: DisplayAsF64 + Zero + Clone + Copy,
{
    debug_dense_matrix_to_csv(&matrix.to_dense(), out_path)
}

pub fn debug_dense_matrix_to_csv<N>(
    matrix: &DenseMatrix<N>,
    out_path: &str,
) -> Result<(), CsvImportError>
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

fn debug_pagerank_to_csv(rank: &Rank<f64>, out_path: &str) -> Result<(), CsvImportError> {
    let mut output_csv = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out_path)?;

    for (project_name, pagerank) in rank
        .into_iter()
        .sorted_by(|(_, v1), (_, v2)| v2.partial_cmp(v1).unwrap())
    {
        output_csv.write_all(
            format!("{} {:.32}\n", project_name, pagerank)
                .as_str()
                .as_bytes(),
        )?;
    }

    Ok(())
}

/// Naive porting of the Python algorithm.
fn build_adjacency_matrix(
    deps_file: &str,
    deps_meta_file: &str,
    contrib_file: &str,
    _out_path: &str,
) -> Result<(), CsvImportError> {
    let deps_csv = csv::Reader::from_reader(File::open(deps_file)?);
    let deps_meta_csv = csv::Reader::from_reader(File::open(deps_meta_file)?);
    let contribs_csv_first_pass = csv::Reader::from_reader(File::open(contrib_file)?);

    let mut deps_meta = DependenciesMetadata::new();
    let mut contribs_meta = ContributionsMetadata::new();
    let mut contrib_rows = Vec::default();

    // Iterate once over the dependencies metadata and store the name and id.
    // We need to maintain some sort of mapping between the order of visit
    // (which will be used as the index in the matrix) and the project id.
    for result in deps_meta_csv.into_records().filter_map(|e| e.ok()) {
        let row: DepMetaRow = result.deserialize(None)?;
        deps_meta.ids.insert(row.id);
        deps_meta.labels.push(row.name);
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
        let contributor = Rc::new(row.contributor.clone());

        contribs_meta.contributors.insert(Rc::clone(&contributor));
        contribs_meta.contributor2index.insert(
            Rc::clone(&contributor),
            contribs_meta.contributors.len() - 1,
        );
        contrib_rows.push(row);
    }

    //TODO(adn) For now the maintenance matrix is empty.

    println!("Assembling the dependency matrix...");
    let dep_adj_matrix = new_dependency_adjacency_matrix(&deps_meta, deps_csv)?;
    println!(
        "Generated a matrix of {}x{}",
        dep_adj_matrix.rows(),
        dep_adj_matrix.cols()
    );
    println!("Assembling the contribution matrix...");
    let con_adj_matrix =
        new_contribution_adjacency_matrix(&deps_meta, &contribs_meta, Box::new(f64::from))?;
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
        &HyperParams::default(),
    );

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

    let pagerank_matrix_labeled =
        Rank::from(pagerank_matrix.labeled((deps_meta.labels.to_vec().as_slice(), &[])))
            .unwrap_or_else(|e| panic!(e));

    println!("Write the matrix to file (skipped for now)");

    // Just for fun/debug: write this as a CSV file.
    debug_pagerank_to_csv(&pagerank_matrix_labeled, "data/cargo-page-rank.csv")?;

    Ok(())
}

fn main() -> Result<(), CsvImportError> {
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
    use num_traits::{One, Zero};
    use osrank::adjacency::new_network_matrix;
    use osrank::importers::csv::{ContribRow, ContributionsMetadata, DependenciesMetadata};
    use osrank::linalg::{hadamard_mul, normalise_rows, normalise_rows_mut};
    use osrank::types::{HyperParams, Weight};
    use pretty_assertions::assert_eq;
    use sprs::{CsMat, TriMat, TriMatBase};
    use std::rc::Rc;

    #[test]
    // Check that this implementation is the same as the Python script
    fn normalise_rows_equal_python() {
        let mut input = CsMat::csr_from_dense(
            arr2(&[[10., 20., 30.], [7., 8., 9.], [5., 9., 4.]]).view(),
            0.0,
        );

        normalise_rows_mut(&mut input);

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
        let result = hadamard_mul(input, &input2);

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

        let actual = normalise_rows(&super::transpose_storage(&c));

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

        let network = new_network_matrix(&d, &c, &m, &HyperParams::default());

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
        let dep_meta = DependenciesMetadata {
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

        let github_foo = Rc::new(String::from("github@foo"));
        let github_bar = Rc::new(String::from("github@bar"));
        let github_baz = Rc::new(String::from("github@baz"));

        let contribs = ContributionsMetadata {
            rows: vec![
                ContribRow {
                    project_id: 10,
                    contributor: String::from("github@foo"),
                    repo: String::from("https://github.com/oscoin/oscoin"),
                    contributions: 118,
                    project_name: String::from("oscoin"),
                },
                ContribRow {
                    project_id: 10,
                    contributor: String::from("github@bar"),
                    repo: String::from("https://github.com/oscoin/oscoin"),
                    contributions: 32,
                    project_name: String::from("oscoin"),
                },
                ContribRow {
                    project_id: 15,
                    contributor: String::from("github@baz"),
                    repo: String::from("https://github.com/osrank/osrank"),
                    contributions: 10,
                    project_name: String::from("osank"),
                },
            ],
            contributors: [
                Rc::clone(&github_foo),
                Rc::clone(&github_bar),
                Rc::clone(&github_baz),
            ]
            .iter()
            .cloned()
            .collect(),
            contributor2index: [
                (github_foo, 0 as usize),
                (github_bar, 1 as usize),
                (github_baz, 2 as usize),
            ]
            .iter()
            .cloned()
            .collect(),
        };

        let actual = super::new_contribution_adjacency_matrix(
            &dep_meta,
            &contribs,
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
