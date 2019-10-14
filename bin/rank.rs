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

use clap::{App, Arg};
use core::fmt::Debug;
use oscoin_graph_api::{Graph, GraphAlgorithm, GraphObject};
use std::fs::File;

use osrank::algorithm::naive::{OsrankNaiveAlgorithm, OsrankNaiveMockContext};
use osrank::algorithm::{Normalised, OsrankError};
use osrank::exporters::csv::CsvExporterError;
use osrank::exporters::Exporter;
use osrank::importers::csv::{import_network, CsvImportError};
use osrank::protocol_traits::ledger::{LedgerView, MockLedger};
use osrank::types;
use osrank::types::mock::{Mock, MockAnnotator, MockAnnotatorCsvExporter, MockNetwork};
use osrank::types::walk::SeedSet;

#[derive(Debug, Fail)]
enum AppError {
    // Returned in case of generic I/O error.
    #[fail(display = "i/o error when reading/writing on the CSV file {}", _0)]
    IOError(std::io::Error),
    #[fail(display = "import error when reading/writing on the CSV file {}", _0)]
    ImportError(CsvImportError),
    #[fail(display = "export error when reading/writing on the CSV file {}", _0)]
    ExportError(CsvExporterError),
    #[fail(
        display = "export when running the Osrank algorithm on the graph {}",
        _0
    )]
    AlgorithmError(OsrankError),
}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> AppError {
        AppError::IOError(err)
    }
}

impl From<CsvImportError> for AppError {
    fn from(err: CsvImportError) -> AppError {
        AppError::ImportError(err)
    }
}

impl From<CsvExporterError> for AppError {
    fn from(err: CsvExporterError) -> AppError {
        AppError::ExportError(err)
    }
}

impl From<OsrankError> for AppError {
    fn from(err: OsrankError) -> AppError {
        AppError::AlgorithmError(err)
    }
}

#[derive(Debug)]
pub enum OsrankAlgorithm {
    Naive,
    Incremental,
}

fn run_osrank(
    deps_file: &str,
    deps_meta_file: &str,
    contrib_file: &str,
    out_path: &str,
    osrank_algo: OsrankAlgorithm,
    ledger: MockLedger,
    seed_set: Option<SeedSet<<<MockNetwork as Graph>::Node as GraphObject>::Id>>,
) -> Result<(), AppError> {
    let deps_csv_file = File::open(deps_file)?;
    let deps_meta_csv_file = File::open(deps_meta_file)?;
    let contribs_csv_file = File::open(contrib_file)?;

    debug!("Importing the network...");

    let ss = match &seed_set {
        None => None,
        Some(r) => Some(r),
    };

    let (algo, mut ctx, network) = match osrank_algo {
        OsrankAlgorithm::Naive => {
            debug!("Selecting the naive algorithm...");
            let a: Mock<
                OsrankNaiveAlgorithm<
                    Normalised<MockNetwork>,
                    MockLedger,
                    MockAnnotator<Normalised<MockNetwork>>,
                >,
            > = Mock {
                unmock: OsrankNaiveAlgorithm::default(),
            };
            let mut ctx = OsrankNaiveMockContext::default();
            ctx.seed_set = ss;
            ctx.ledger_view = ledger;
            let network = import_network::<MockNetwork, MockLedger, File>(
                csv::Reader::from_reader(deps_csv_file),
                csv::Reader::from_reader(deps_meta_csv_file),
                csv::Reader::from_reader(contribs_csv_file),
                None,
                &ctx.ledger_view,
            )?;

            (a, ctx, network)
        }
        OsrankAlgorithm::Incremental => {
            debug!("Selecting the incremental (not-implemented) algorithm...");
            let a: Mock<
                OsrankNaiveAlgorithm<
                    Normalised<MockNetwork>,
                    MockLedger,
                    MockAnnotator<Normalised<MockNetwork>>,
                >,
            > = Mock {
                unmock: OsrankNaiveAlgorithm::default(),
            };
            let mut ctx = OsrankNaiveMockContext::default();
            ctx.seed_set = ss;
            ctx.ledger_view = ledger;
            let network = import_network::<MockNetwork, MockLedger, File>(
                csv::Reader::from_reader(deps_csv_file),
                csv::Reader::from_reader(deps_meta_csv_file),
                csv::Reader::from_reader(contribs_csv_file),
                None,
                &ctx.ledger_view,
            )?;

            (a, ctx, network)
        }
    };

    debug!(
        "{}",
        format!("Calculating the osrank ({:#?} algorithm)...", osrank_algo)
    );

    let initial_seed = [0; 32];
    let mut annotator: MockAnnotator<Normalised<MockNetwork>> = Default::default();

    algo.execute(&mut ctx, &network, &mut annotator, initial_seed)?;

    debug!("Exporting the ranks into a .csv file ...");
    // Export the ranks into a csv file.
    let rank_exporter = MockAnnotatorCsvExporter::new(annotator, out_path);
    rank_exporter.export()?;

    debug!("Done.");
    Ok(())
}

fn parse_algorithm(algo_str: &str) -> Option<OsrankAlgorithm> {
    match algo_str {
        "naive" => Some(OsrankAlgorithm::Naive),
        "incremental" => Some(OsrankAlgorithm::Incremental),
        _ => None,
    }
}

fn parse_seed_set(
    _path_to_seed_file: &str,
) -> Option<SeedSet<<<MockNetwork as Graph>::Node as GraphObject>::Id>> {
    None
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
        .arg(
            Arg::with_name("tau")
                .help("The value of 'tau', i.e. the pruning threshold for the trustrank phase. (Default: 0.0)")
                .takes_value(true)
                .default_value("0.0")
                .required(false),
        )
        .arg(
            Arg::with_name("iter")
                .short("i")
                .help("The number of iterations (R) for each random walk (Default: 10)")
                .takes_value(true)
                .default_value("10")
                .required(false),
        )
        .arg(
            Arg::with_name("accounts-damping-factor")
                .help("The damping factor for accounts (Default: 0.85)")
                .takes_value(true)
                .default_value("0.85")
                .required(false),
        )
        .arg(
            Arg::with_name("projects-damping-factor")
                .help("The damping factor for projects (Default: 0.85)")
                .takes_value(true)
                .default_value("0.85")
                .required(false),
        )
        .get_matches();

    let tau = matches
        .value_of("tau")
        .and_then(|s: &str| s.parse::<types::Tau>().ok())
        .unwrap_or(0.0);

    let r = matches
        .value_of("iter")
        .and_then(|s: &str| s.parse::<types::R>().ok())
        .unwrap_or(10);

    let acc_damping_factor = matches
        .value_of("accounts-damping-factor")
        .and_then(|s: &str| s.parse::<f64>().ok())
        .unwrap_or(0.85);

    let prj_damping_factor = matches
        .value_of("projects-damping-factor")
        .and_then(|s: &str| s.parse::<f64>().ok())
        .unwrap_or(0.85);

    let damping_factors = types::DampingFactors {
        project: prj_damping_factor,
        account: acc_damping_factor,
    };

    let mut ledger_view = MockLedger::default();
    ledger_view.set_tau(tau);
    ledger_view.set_random_walks_num(r);
    ledger_view.set_damping_factors(damping_factors);

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
        matches
            .value_of("algorithm")
            .and_then(parse_algorithm)
            .expect("Failed to parse algorithm. Possible choices: naive|incremental."),
        ledger_view,
        matches.value_of("seed-set").and_then(parse_seed_set),
    )
}
