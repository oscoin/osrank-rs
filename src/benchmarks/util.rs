#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate itertools;
extern crate oscoin_graph_api;
extern crate rand;
extern crate rand_xoshiro;

use crate::algorithm::naive::{random_walk, OsrankNaiveAlgorithm, OsrankNaiveMockContext};
use crate::importers::csv::import_network;
use crate::protocol_traits::ledger::{LedgerView, MockLedger};
use crate::types::mock::{Mock, MockAnnotator, MockNetwork};
use crate::types::network::{ArtifactType, DependencyType, Network};
use crate::types::Weight;
use itertools::Itertools;
use num_traits::Zero;
use oscoin_graph_api::{GraphAlgorithm, GraphWriter};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::fs::File;
use std::io::{BufRead, BufReader};

const NIGHTLY: &str = "nightly";
const DEV: &str = "dev";

pub fn nightly(name: &str) -> String {
    format!("({}) {}", NIGHTLY, name)
}

pub fn dev(name: &str) -> String {
    format!("({}) {}", DEV, name)
}

pub fn construct_osrank_naive_algorithm<'a>() -> (
    Mock<OsrankNaiveAlgorithm<'a, MockNetwork, MockLedger, MockAnnotator<MockNetwork>>>,
    MockAnnotator<MockNetwork>,
    OsrankNaiveMockContext<'a, MockAnnotator<MockNetwork>, MockNetwork>,
) {
    let algo: Mock<OsrankNaiveAlgorithm<MockNetwork, MockLedger, MockAnnotator<MockNetwork>>> =
        Mock {
            unmock: OsrankNaiveAlgorithm::default(),
        };

    let ctx = OsrankNaiveMockContext::default();

    (algo, Default::default(), ctx)
}

pub fn construct_network_small() -> MockNetwork {
    let mut network = Network::default();
    for node in &["p1", "p2", "p3"] {
        network.add_node(
            node.to_string(),
            ArtifactType::Project {
                osrank: Zero::zero(),
            },
        )
    }
    for node in &["a1", "a2", "a3"] {
        network.add_node(
            node.to_string(),
            ArtifactType::Account {
                osrank: Zero::zero(),
            },
        )
    }
    let edges = [
        ("p1", "a1", Weight::new(3, 7)),
        ("a1", "p1", Weight::new(1, 1)),
        ("p1", "p2", Weight::new(4, 7)),
        ("p2", "a2", Weight::new(1, 1)),
        ("a2", "p2", Weight::new(1, 3)),
        ("a2", "p3", Weight::new(2, 3)),
        ("p3", "a2", Weight::new(11, 28)),
        ("p3", "a3", Weight::new(1, 28)),
        ("p3", "p1", Weight::new(2, 7)),
        ("p3", "p2", Weight::new(2, 7)),
        ("a3", "p3", Weight::new(1, 1)),
    ];
    for edge in &edges {
        network.add_edge(
            2,
            &edge.0.to_string(),
            &edge.1.to_string(),
            edge.2.as_f64().unwrap(),
            DependencyType::Influence(edge.2.as_f64().unwrap()),
        )
    }
    network
}

pub fn construct_network(meta_num: usize, contributions_num: usize) -> MockNetwork {
    let deps_reader = BufReader::new(File::open("data/cargo_dependencies.csv").unwrap())
        .split(b'\n')
        .map(|l| l.unwrap())
        .intersperse(vec![b'\n'])
        .flatten()
        .collect::<Vec<u8>>();

    let deps_meta_reader = BufReader::new(File::open("data/cargo_dependencies_meta.csv").unwrap())
        .split(b'\n')
        .map(|l| l.unwrap())
        .take(meta_num)
        .intersperse(vec![b'\n']) // re-add the '\n'
        .flatten()
        .collect::<Vec<u8>>();

    let contribs_reader = BufReader::new(File::open("data/cargo_contributions.csv").unwrap())
        .split(b'\n')
        .map(|l| l.unwrap())
        .take(contributions_num)
        .intersperse(vec![b'\n']) // re-add the '\n'
        .flatten()
        .collect::<Vec<u8>>();

    let mock_ledger = MockLedger::default();
    import_network::<MockNetwork, MockLedger, _>(
        csv::Reader::from_reader(deps_reader.as_slice()),
        csv::Reader::from_reader(deps_meta_reader.as_slice()),
        csv::Reader::from_reader(contribs_reader.as_slice()),
        None,
        &mock_ledger,
    )
    .unwrap()
}

pub fn run_osrank_naive(network: &MockNetwork, iter: u32, initial_seed: [u8; 32]) {
    let (algo, mut annotator, mut ctx) = construct_osrank_naive_algorithm();
    ctx.ledger_view.set_random_walks_num(iter);
    algo.execute(&mut ctx, &network, &mut annotator, initial_seed)
        .unwrap();
}

pub fn run_random_walk(network: &MockNetwork, iter: u32, initial_seed: [u8; 32]) {
    let mut mock_ledger = MockLedger::default();

    mock_ledger.set_random_walks_num(iter);
    random_walk::<MockLedger, MockNetwork, Xoshiro256StarStar>(
        None,
        &network,
        &mock_ledger,
        &Xoshiro256StarStar::from_seed(initial_seed),
    )
    .unwrap();
}
