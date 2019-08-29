#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate clap;

use clap::{App, Arg};

use osrank::exporters::gexf;
use osrank::importers::csv::import_network;
use osrank::protocol_traits::ledger::MockLedger;
use osrank::types::mock::MockNetwork;

use std::fs::File;
use std::path::Path;

fn main() -> Result<(), CsvImportError> {
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

    let mut deps_csv_file = File::open(deps).unwrap();
    let mut deps_meta_csv_file = File::open(deps_meta).unwrap();
    let mut contribs_csv_file = File::open(contribs).unwrap();

    let mock_ledger = MockLedger::default();
    let network = import_network::<MockNetwork, MockLedger, File>(
        csv::Reader::from_reader(deps_csv_file),
        csv::Reader::from_reader(deps_meta_csv_file),
        csv::Reader::from_reader(contribs_csv_file),
        None,
        &mock_ledger,
    );

    gexf::export_graph(&network, &Path::new(out));
}
