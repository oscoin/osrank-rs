#![allow(unknown_lints)]
#![warn(clippy::all)]

#[macro_use]
extern crate log;
extern crate env_logger;

#[macro_use]
extern crate failure_derive;

extern crate clap;
extern crate failure;
extern crate ndarray;
extern crate num_traits;
extern crate osrank;
extern crate serde;
extern crate sprs;

#[macro_use(quickcheck)]
extern crate quickcheck_macros;

use clap::{App, Arg};
use itertools::Itertools;
use ndarray::Array2;
use osrank::adjacency::new_network_matrix;
use osrank::collections::{Rank, WithLabels};
use osrank::exporters::csv::CsvExporterError;
use osrank::exporters::Exporter;
use osrank::importers::csv::{
    new_contribution_adjacency_matrix, new_dependency_adjacency_matrix, ContribRow,
    ContributionsMetadata, CsvImportError, DepMetaRow, DependenciesMetadata, DisplayAsF64,
};
use osrank::linalg::{transpose_storage_csr, transpose_storage_naive, DenseMatrix, SparseMatrix};
use osrank::types::mock::MockAnnotatorCsvExporter;
use osrank::types::HyperParams;
use sprs::binop::{add_mat_same_storage, scalar_mul_mat};
use sprs::CsMat;

use core::fmt::Debug;
use num_traits::{Num, One, Signed, Zero};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::rc::Rc;

#[derive(Debug, Fail)]
enum AppError {
    // Returned in case of generic I/O error.
    #[fail(display = "i/o error when reading/writing on the CSV file {}", _0)]
    IOError(std::io::Error),
    #[fail(display = "import error when reading/writing on the CSV file {}", _0)]
    ImportError(CsvImportError),
    #[fail(display = "export error when reading/writing on the CSV file {}", _0)]
    ExportError(CsvExporterError),
}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> AppError {
        AppError::IOError(err)
    }
}

impl From<CsvExporterError> for AppError {
    fn from(err: CsvExporterError) -> AppError {
        AppError::ExportError(err)
    }
}

fn run_osrank(
    deps_file: &str,
    deps_meta_file: &str,
    contrib_file: &str,
    out_path: &str,
) -> Result<(), AppError> {
    // Export the ranks into a csv file.
    let annotator = unimplemented!("todo");
    let rank_exporter = MockAnnotatorCsvExporter::new(annotator, out_path);
    rank_exporter.export()?;
    Ok(())
}

fn main() -> Result<(), AppError> {
    env_logger::init();
    let matches = App::new("Run the Osrank algorithm and collect the result in a csv file.")
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
                .help("Path to the output .csv file which will contain the ranks")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("algorithm")
                .short("a")
                .help("The type of algorithm to use (naive|incremental).")
                .takes_value(true)
                .default_value("naive")
                .required(false),
        )
        .get_matches();

    run_osrank(
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
