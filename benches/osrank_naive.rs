#![macro_use]
extern crate criterion;
extern crate rand;
extern crate rand_xorshift;

use criterion::{Criterion, Benchmark};
use criterion::criterion_group;
use criterion::criterion_main;
use osrank::protocol_traits::graph::{Graph, GraphObject};
use osrank::protocol_traits::ledger::{MockLedger, LedgerView};
use osrank::types::{Artifact, ArtifactType, Dependency, DependencyType, Network, Weight};
use osrank::algorithm::osrank_naive;
use osrank::algorithm::random_walk;
use osrank::importers::csv::import_network;
use num_traits::Zero;
use rand_xorshift::XorShiftRng;
use crate::rand::SeedableRng;
use std::fs::File;

type MockNetwork = Network<f64>;

fn construct_network_small() -> Network<f64> {
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
        ("a1", "p2", Weight::new(3, 7)), ("p1", "p2", Weight::new(1, 1)),
        ("p1", "p2", Weight::new(4, 7)), ("p2", "a2", Weight::new(1, 1)),
        ("a2", "p2", Weight::new(1, 3)), ("a2", "p3", Weight::new(2, 3)),
        ("p3", "a2", Weight::new(11, 28)), ("p3", "a3", Weight::new(1, 28)),
        ("p3", "p1", Weight::new(2, 7)), ("p3", "p2", Weight::new(2, 7)),
        ("a3", "p3", Weight::new(1, 1))
    ];
    for edge in &edges {
        network.add_edge(
            &edge.0.to_string(),
            &edge.1.to_string(),
            2,
            DependencyType::Influence(edge.2.as_f64().unwrap()),
        )
    }
    network
}

fn construct_network() -> Network<f64> {
    let deps_csv_file = File::open("data/cargo_dependencies.csv").unwrap();
    let deps_meta_csv_file = File::open("data/cargo_dependencies_meta_bench_sample.csv").unwrap();
    let contribs_csv_file = File::open("data/cargo_contributions_bench_sample.csv").unwrap();
    let mock_ledger = MockLedger::default();
    import_network::<MockNetwork, MockLedger, File>( csv::Reader::from_reader(deps_csv_file)
                                , csv::Reader::from_reader(deps_meta_csv_file)
                                , csv::Reader::from_reader(contribs_csv_file)
                                , None
                                , &mock_ledger).unwrap()
}

fn run_osrank_naive(mut network: &mut Network<f64>, iter: u32, initial_seed: [u8; 16]) {
    let mut mock_ledger = MockLedger::default();
    mock_ledger.set_random_walks_num(iter);
    let get_weight = Box::new(|m: &DependencyType<f64>| *m.get_weight());
    let set_osrank = Box::new(|node: &Artifact<String>, rank| match node.get_metadata() {
        ArtifactType::Project { osrank: _ } => ArtifactType::Project { osrank: rank },
        ArtifactType::Account { osrank: _ } => ArtifactType::Account { osrank: rank },
    });
    osrank_naive::<MockLedger, MockNetwork, XorShiftRng>(
        None,
        &mut network,
        &mock_ledger,
        initial_seed,
        get_weight,
        set_osrank);
}

fn run_random_walk(network: &Network<f64>, iter: u32, initial_seed: [u8; 16]) {
    let mut mock_ledger = MockLedger::default();
    mock_ledger.set_random_walks_num(iter);
    let get_weight:Box<Fn(&<Dependency<usize, f64> as GraphObject>::Metadata) -> f64>
        = Box::new(|m: &DependencyType<f64>| *m.get_weight());
    random_walk::<MockLedger, MockNetwork, XorShiftRng>(
        None,
        &network,
        &mock_ledger,
        XorShiftRng::from_seed(initial_seed.clone()),
        &get_weight);
}

fn bench_osrank_naive_on_small_network(c: &mut Criterion) {
    let mut network = construct_network_small();
    c.bench_function("osrank by random seed", move |b| b.iter(|| {
        let rand_vec : [u8; 16] = rand::random();
        run_osrank_naive(&mut network, 1, rand_vec)
    }));
}

// run with a lower sample size to speed things up
fn bench_osrank_naive_on_sample_csv(c: &mut Criterion) {
    let mut network = construct_network();
    let info = format!("osrank with {:?} nodes, iter: 1", &network.node_count());
    c.bench(&info,
        Benchmark::new("sample size 10", move |b| b.iter(|| {run_osrank_naive(&mut network, 1, [0; 16])}))
            .sample_size(10)
    );
}

fn bench_random_walk_on_csv(c: &mut Criterion) {
    let network = construct_network();
    let info = format!("random walks with {:?} nodes, iter: 1", &network.node_count());
    c.bench_function(&info, move |b| b.iter(|| {
        run_random_walk(&network, 1, [0; 16])
    }));
}

fn bench_count_visits(c: &mut Criterion) {
    let network = construct_network();
    let mut mock_ledger = MockLedger::default();
    mock_ledger.set_random_walks_num(1);
    let get_weight:Box<Fn(&<Dependency<usize, f64> as GraphObject>::Metadata) -> f64>
        = Box::new(|m: &DependencyType<f64>| *m.get_weight());
    let walks = random_walk::<MockLedger, MockNetwork, XorShiftRng>(
                    None,
                    &network,
                    &mock_ledger,
                    XorShiftRng::from_seed([0; 16]),
                    &get_weight).unwrap().walks;
    let info = format!("count visits of {:?} nodes by {:?} walks", &network.node_count(), &walks.len());
    c.bench(&info,
        Benchmark::new("sample size 10", move |b| b.iter(|| {
            for node in network.nodes(){
                &walks.count_visits(node.id().clone());
            }
        })).sample_size(10)
    );
}

criterion_group!(
    benches,
    bench_osrank_naive_on_small_network,
    bench_osrank_naive_on_sample_csv,
    bench_random_walk_on_csv,
    bench_count_visits
);
criterion_main!(benches);
