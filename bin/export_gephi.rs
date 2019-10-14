#![allow(unknown_lints)]
#![warn(clippy::all)]

#[macro_use]
extern crate log;
extern crate env_logger;

extern crate clap;

use clap::{App, Arg};

use oscoin_graph_api::GraphAlgorithm;
use osrank::algorithm::naive::{OsrankNaiveAlgorithm, OsrankNaiveMockContext};
use osrank::algorithm::{Normalised, OsrankError};
use osrank::exporters::gexf::GexfExporter;
use osrank::exporters::graphml::GraphMlExporter;
use osrank::exporters::{gexf, graphml, Exporter};
use osrank::importers::csv::{import_network, CsvImportError};
use osrank::protocol_traits::ledger::{LedgerView, MockLedger};
use osrank::types::mock::{Mock, MockAnnotator, MockNetwork};

use std::fs::File;

#[derive(Debug)]
enum AppError {
    AlgorithmError(OsrankError),
    ImportError(CsvImportError),
    GexfExportError,
    GraphMlExportError,
}

impl From<OsrankError> for AppError {
    fn from(err: OsrankError) -> AppError {
        AppError::AlgorithmError(err)
    }
}

impl From<CsvImportError> for AppError {
    fn from(err: CsvImportError) -> AppError {
        AppError::ImportError(err)
    }
}

impl From<gexf::ExportError> for AppError {
    fn from(_err: gexf::ExportError) -> AppError {
        AppError::GexfExportError
    }
}

impl From<graphml::ExportError> for AppError {
    fn from(_err: graphml::ExportError) -> AppError {
        AppError::GraphMlExportError
    }
}

fn main() -> Result<(), AppError> {
    env_logger::init();

    let matches = App::new("Export a Graph into a GEFX xml.")
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
                .help("Path to the output .gexf file")
                .takes_value(true)
                .required(true),
        )
        .get_matches();

    let deps = matches
        .value_of("dependencies")
        .expect("dependencies csv file not given.");
    let deps_meta = matches
        .value_of("dependencies-with-metadata")
        .expect("dependencies with metadata csv file not given.");
    let contribs = matches
        .value_of("contributions")
        .expect("contributions csv file not given.");
    let out = matches
        .value_of("output-path")
        .expect("output csv file not specified.");

    let deps_csv_file = File::open(deps).unwrap();
    let deps_meta_csv_file = File::open(deps_meta).unwrap();
    let contribs_csv_file = File::open(contribs).unwrap();

    debug!("Importing the network...");

    let algo: Mock<
        OsrankNaiveAlgorithm<
            Normalised<MockNetwork>,
            MockLedger,
            MockAnnotator<Normalised<MockNetwork>>,
        >,
    > = Mock {
        unmock: OsrankNaiveAlgorithm::default(),
    };
    let mut ctx = OsrankNaiveMockContext::default();
    ctx.ledger_view.set_random_walks_num(10);

    let mut network = import_network::<MockNetwork, MockLedger, File>(
        csv::Reader::from_reader(deps_csv_file),
        csv::Reader::from_reader(deps_meta_csv_file),
        csv::Reader::from_reader(contribs_csv_file),
        None,
        &ctx.ledger_view,
    )?;

    debug!("Calculating the osrank (mock naive algorithm)...");

    let initial_seed = [0; 32];
    let mut annotator: MockAnnotator<Normalised<MockNetwork>> = Default::default();

    algo.execute(&mut ctx, &mut network, &mut annotator, initial_seed)?;

    debug!("Exporting the network to .gexf ...");

    let gexf = GexfExporter::new(&network, &annotator, &out);
    gexf.export_graph()?;

    debug!("Exporting the network to .graphml ...");
    let graphml = GraphMlExporter::new(&network, &annotator, &out);
    graphml.export_graph()?;

    debug!("Done.");

    Ok(())
}
