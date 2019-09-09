#![macro_use]
extern crate criterion;
extern crate oscoin_graph_api;
extern crate rand;
extern crate rand_xorshift;

use crate::rand::SeedableRng;
use criterion::*;
use itertools::Itertools;
use num_traits::Zero;
use oscoin_graph_api::{GraphAlgorithm, GraphWriter};
use osrank::algorithm::{random_walk, rank_network, OsrankNaiveAlgorithm, OsrankNaiveMockContext};
use osrank::importers::csv::import_network;
use osrank::protocol_traits::graph::GraphExtras;
use osrank::protocol_traits::ledger::{LedgerView, MockLedger};
use osrank::types::mock::{Mock, MockNetwork};
use osrank::types::network::{ArtifactType, DependencyType, Network};
use osrank::types::Weight;
use rand_xorshift::XorShiftRng;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn construct_osrank_naive_algorithm<'a>() -> (
    Mock<OsrankNaiveAlgorithm<'a, MockNetwork, MockLedger>>,
    OsrankNaiveMockContext<'a, MockNetwork>,
) {
    let algo: Mock<OsrankNaiveAlgorithm<MockNetwork, MockLedger>> = Mock {
        unmock: OsrankNaiveAlgorithm::default(),
    };

    let ctx = OsrankNaiveMockContext::default();

    (algo, ctx)
}

fn construct_network_small() -> MockNetwork {
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

fn construct_network(meta_num: usize, contributions_num: usize) -> MockNetwork {
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

fn run_osrank_naive(mut network: &mut MockNetwork, iter: u32, initial_seed: [u8; 16]) {
    let (algo, mut ctx) = construct_osrank_naive_algorithm();
    ctx.ledger_view.set_random_walks_num(iter);
    algo.execute(&mut ctx, &mut network, initial_seed).unwrap();
}

fn run_random_walk(network: &MockNetwork, iter: u32, initial_seed: [u8; 16]) {
    let mut mock_ledger = MockLedger::default();

    mock_ledger.set_random_walks_num(iter);
    random_walk::<MockLedger, MockNetwork, XorShiftRng>(
        None,
        &network,
        &mock_ledger,
        &mut XorShiftRng::from_seed(initial_seed.clone()),
    )
    .unwrap();
}

// Benchmarks intended to be run for development are appended with `(dev) `.
// `cargo bench -- dev` will only run them.
fn bench_osrank_naive_on_small_network(c: &mut Criterion) {
    let mut network = construct_network_small();
    c.bench_function("(dev) osrank by random seed", move |b| {
        b.iter(|| {
            let rand_vec: [u8; 16] = rand::random();
            run_osrank_naive(&mut network, 1, rand_vec)
        })
    });
}

// run with a lower sample size to speed things up
fn bench_osrank_naive_on_sample_csv(c: &mut Criterion) {
    let mut network = construct_network(1_000, 10_000);
    let info = format!(
        "(dev) osrank with {:?} nodes, iter: 1",
        &network.node_count()
    );
    c.bench(
        &info,
        Benchmark::new("sample size 10", move |b| {
            b.iter(|| run_osrank_naive(&mut network, 1, [0; 16]))
        })
        .sample_size(10),
    );
}

fn bench_random_walk_on_csv(c: &mut Criterion) {
    let network = construct_network(1_000, 10_000);
    let info = format!(
        "(dev) random walks with {:?} nodes, iter: 1",
        &network.node_count()
    );
    c.bench_function(&info, move |b| {
        b.iter(|| run_random_walk(&network, 1, [0; 16]))
    });
}

fn bench_rank_network(c: &mut Criterion) {
    let mut network = construct_network(1_000, 10_000);

    let (_algo, mut ctx) = construct_osrank_naive_algorithm();
    ctx.ledger_view.set_random_walks_num(1);

    let walks = random_walk::<MockLedger, MockNetwork, XorShiftRng>(
        None,
        &network,
        &ctx.ledger_view,
        &mut XorShiftRng::from_seed([0; 16]),
    )
    .unwrap()
    .walks;
    let info = format!(
        "(dev) bench network of {:?} nodes with {:?} walks",
        &network.node_count(),
        &walks.len()
    );
    c.bench(
        &info,
        Benchmark::new("sample size 10", move |b| {
            b.iter(|| rank_network(&walks, &mut network, &ctx.ledger_view, &ctx.set_osrank))
        })
        .sample_size(10),
    );
}

// The following benchmarks are very slow and intended to be run on a nighlty CI. For local testing
// run the benchmarks with `cargo bench -- dev` to avoid them
fn bench_nightly_osrank_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("(nightly) Increasing node count");
    group.sample_size(10);

    for iter in &[1, 10, 100, 1_000] {
        for count in &[1_001, 2_501, 5_001, 7_501, 10_001, 15_001] {
            let mut network = construct_network(*count as usize, 0);
            let nodes = &network.node_count();
            group.bench_function(
                BenchmarkId::new(format!("Iter {}", &iter), nodes),
                move |b| b.iter(|| run_osrank_naive(&mut network, *iter as u32, [0; 16])),
            );
        }

        for count in &[5_000, 10_000, 19_370] {
            let mut network = construct_network(16_220, *count as usize);
            let nodes = &network.node_count();
            group.bench_function(
                BenchmarkId::new(format!("Iter {}", &iter), nodes),
                move |b| b.iter(|| run_osrank_naive(&mut network, *iter as u32, [0; 16])),
            );
        }
    }
    group.finish();
}

fn bench_nightly_random_walk(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("(nightly) Random walks with increasing iterator and node count");
    group.sample_size(10);

    for iter in &[1, 10, 100, 1_000] {
        for count in &[1_001, 2_501, 5_001, 7_501, 10_001, 15_001] {
            let network = construct_network(*count as usize, 0);
            let nodes = &network.node_count();
            group.bench_function(
                BenchmarkId::new(format!("Iter {}", &iter), nodes),
                move |b| b.iter(|| run_random_walk(&network, *iter, [0; 16])),
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_osrank_naive_on_small_network,
    bench_osrank_naive_on_sample_csv,
    bench_random_walk_on_csv,
    bench_rank_network,
    bench_nightly_osrank_naive,
    bench_nightly_random_walk
);
criterion_main!(benches);
