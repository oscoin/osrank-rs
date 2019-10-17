#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate itertools;
extern crate oscoin_graph_api;
extern crate rand;
extern crate rand_xoshiro;

use crate::algorithm::naive::{random_walk, OsrankNaiveAlgorithm, OsrankNaiveMockContext};
use crate::importers::csv::import_network;
use crate::protocol_traits::ledger;
use crate::protocol_traits::ledger::LedgerView;
use crate::types::mock;
use crate::types::mock::Mock;
use crate::types::network::Network;
use crate::types::Weight;
use crate::util::{add_edges, add_projects, add_users};
use itertools::Itertools;
use oscoin_graph_api::{types, GraphAlgorithm};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

type MockNetwork = mock::MockNetwork<f64>;
type MockLedger = ledger::MockLedger<f64>;
type MockAnnotator<N> = mock::MockAnnotator<N>;

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
    OsrankNaiveMockContext<'a, f64, MockAnnotator<MockNetwork>, MockNetwork>,
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

    add_projects(
        &mut network,
        [("p1", 10), ("p2", 20), ("p3", 30)].iter().cloned(),
    );
    add_users(
        &mut network,
        [("a1", 100), ("a2", 90), ("a3", 80)].iter().cloned(),
    );

    let edges = [
        (
            0,
            "p1",
            "a1",
            types::EdgeData {
                weight: Weight::new(3, 7).as_f64().unwrap(),
                edge_type: types::EdgeType::ProjectToUserContribution(100),
            },
        ),
        (
            1,
            "a1",
            "p1",
            types::EdgeData {
                weight: Weight::new(1, 1).as_f64().unwrap(),
                edge_type: types::EdgeType::UserToProjectContribution(100),
            },
        ),
        (
            2,
            "p1",
            "p2",
            types::EdgeData {
                weight: Weight::new(4, 7).as_f64().unwrap(),
                edge_type: types::EdgeType::Dependency,
            },
        ),
        (
            3,
            "p2",
            "a2",
            types::EdgeData {
                weight: Weight::new(1, 1).as_f64().unwrap(),
                edge_type: types::EdgeType::ProjectToUserContribution(30),
            },
        ),
        (
            4,
            "a2",
            "p2",
            types::EdgeData {
                weight: Weight::new(1, 3).as_f64().unwrap(),
                edge_type: types::EdgeType::UserToProjectContribution(30),
            },
        ),
        (
            5,
            "a2",
            "p3",
            types::EdgeData {
                weight: Weight::new(2, 3).as_f64().unwrap(),
                edge_type: types::EdgeType::UserToProjectContribution(60),
            },
        ),
        (
            6,
            "p3",
            "a2",
            types::EdgeData {
                weight: Weight::new(11, 28).as_f64().unwrap(),
                edge_type: types::EdgeType::ProjectToUserContribution(60),
            },
        ),
        (
            7,
            "p3",
            "a3",
            types::EdgeData {
                weight: Weight::new(1, 28).as_f64().unwrap(),
                edge_type: types::EdgeType::ProjectToUserContribution(20),
            },
        ),
        (
            8,
            "p3",
            "p1",
            types::EdgeData {
                weight: Weight::new(2, 7).as_f64().unwrap(),
                edge_type: types::EdgeType::Dependency,
            },
        ),
        (
            9,
            "p3",
            "p2",
            types::EdgeData {
                weight: Weight::new(2, 7).as_f64().unwrap(),
                edge_type: types::EdgeType::Dependency,
            },
        ),
        (
            10,
            "a3",
            "p3",
            types::EdgeData {
                weight: Weight::new(1, 1).as_f64().unwrap(),
                edge_type: types::EdgeType::UserToProjectContribution(20),
            },
        ),
    ];

    add_edges(&mut network, edges.iter().cloned());

    network
}

const OSRANK_NIGHTLY_NETWORK_DEPS: &str = "OSRANK_NIGHTLY_NETWORK_DEPS";
const OSRANK_NIGHTLY_NETWORK_DEPS_META: &str = "OSRANK_NIGHTLY_NETWORK_DEPS_META";
const OSRANK_NIGHTLY_NETWORK_CONTRIBUTIONS: &str = "OSRANK_NIGHTLY_NETWORK_CONTRIBUTIONS";

pub fn construct_network(meta_num: usize, contributions_num: usize) -> MockNetwork {
    // Due to the fact we need to construct this network using some real, large data, and
    // due to the fact this data lives elsewhere now (in the osrank-rs-ecosystems repo) we have
    // to pass it externally. Unfortunately there seems to be no easy way to customise
    // Criterion's CLI to pass extra arguments, so passing this via env vars seems the cleanest
    // approach.

    let cargo_deps = env::var(OSRANK_NIGHTLY_NETWORK_DEPS).ok();
    let cargo_deps_meta = env::var(OSRANK_NIGHTLY_NETWORK_DEPS_META).ok();
    let cargo_contributions = env::var(OSRANK_NIGHTLY_NETWORK_CONTRIBUTIONS).ok();

    match (cargo_deps, cargo_deps_meta, cargo_contributions) {
        (Some(deps), Some(meta), Some(contribs)) => {
            let deps_reader = BufReader::new(File::open(deps).unwrap())
                .split(b'\n')
                .map(|l| l.unwrap())
                .intersperse(vec![b'\n'])
                .flatten()
                .collect::<Vec<u8>>();

            let deps_meta_reader = BufReader::new(File::open(meta).unwrap())
                .split(b'\n')
                .map(|l| l.unwrap())
                .take(meta_num)
                .intersperse(vec![b'\n']) // re-add the '\n'
                .flatten()
                .collect::<Vec<u8>>();

            let contribs_reader = BufReader::new(File::open(contribs).unwrap())
                .split(b'\n')
                .map(|l| l.unwrap())
                .take(contributions_num)
                .intersperse(vec![b'\n']) // re-add the '\n'
                .flatten()
                .collect::<Vec<u8>>();

            let mock_ledger = MockLedger::default();
            import_network::<MockNetwork, _>(
                csv::Reader::from_reader(deps_reader.as_slice()),
                csv::Reader::from_reader(deps_meta_reader.as_slice()),
                csv::Reader::from_reader(contribs_reader.as_slice()),
                None,
                &mock_ledger.get_hyperparams(),
            )
            .unwrap()
        }
        _ => {
            let msg = format!(
                r###"The nightly benchmarks requires the following env var to be set:
 - {} -> must contain a valid path to a `<platform>_dependencies.csv` file.
 - {} -> must contain a valid path to a `<platform>_dependencies_meta.csv` file.
 - {} -> must contain a valid path to a `<platform>_contributions.csv` file.

 Please specify these env vars and try again.
            "###,
                OSRANK_NIGHTLY_NETWORK_DEPS,
                OSRANK_NIGHTLY_NETWORK_DEPS_META,
                OSRANK_NIGHTLY_NETWORK_CONTRIBUTIONS
            );
            panic!(msg)
        }
    }
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
