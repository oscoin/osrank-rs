#![macro_use]
extern crate criterion;
extern crate rand;
extern crate rand_xorshift;

use criterion::{Criterion, Benchmark};
use criterion::criterion_group;
use criterion::criterion_main;
use osrank::protocol_traits::graph::{Graph, GraphObject};
use osrank::protocol_traits::ledger::MockLedger;
use osrank::protocol_traits::ledger::LedgerView;
use osrank::types::{Artifact, ArtifactType, Dependency, DependencyType, Network, Weight};
use osrank::algorithm::osrank_naive;
use osrank::algorithm::random_walk;
use num_traits::Zero;
use rand_xorshift::XorShiftRng;
use crate::rand::SeedableRng;
use std::fs::File;
use osrank::importers::csv::import_network;

type MockNetwork = Network<f64>;

fn construct_network_small() -> Network<f64> {
    let mut network = Network::default();
    network.add_node(
        "p1".to_string(),
        ArtifactType::Project {
            osrank: Zero::zero(),
        },
    );
    network.add_node(
        "p2".to_string(),
        ArtifactType::Project {
            osrank: Zero::zero(),
        },
    );
    network.add_node(
        "p3".to_string(),
        ArtifactType::Project {
            osrank: Zero::zero(),
        },
    );
    network.add_node(
        "a1".to_string(),
        ArtifactType::Project {
            osrank: Zero::zero(),
        },
    );
    network.add_node(
        "a2".to_string(),
        ArtifactType::Project {
            osrank: Zero::zero(),
        },
    );
    network.add_node(
        "a3".to_string(),
        ArtifactType::Project {
            osrank: Zero::zero(),
        },
    );
    network.add_edge(
        &"p1".to_string(),
        &"a1".to_string(),
        0,
        DependencyType::Influence(Weight::new(3, 7).as_f64().unwrap()),
    );
    network.add_edge(
        &"a1".to_string(),
        &"p1".to_string(),
        1,
        DependencyType::Influence(Weight::new(1, 1).as_f64().unwrap()),
    );
    network.add_edge(
        &"p1".to_string(),
        &"p2".to_string(),
        2,
        DependencyType::Influence(Weight::new(4, 7).as_f64().unwrap()),
    );
    network.add_edge(
        &"p2".to_string(),
        &"a2".to_string(),
        3,
        DependencyType::Influence(Weight::new(1, 1).as_f64().unwrap()),
    );
    network.add_edge(
        &"a2".to_string(),
        &"p2".to_string(),
        4,
        DependencyType::Influence(Weight::new(1, 3).as_f64().unwrap()),
    );
    network.add_edge(
        &"a2".to_string(),
        &"p3".to_string(),
        5,
        DependencyType::Influence(Weight::new(2, 3).as_f64().unwrap()),
    );
    network.add_edge(
        &"p3".to_string(),
        &"a2".to_string(),
        6,
        DependencyType::Influence(Weight::new(11, 28).as_f64().unwrap()),
    );
    network.add_edge(
        &"p3".to_string(),
        &"a3".to_string(),
        7,
        DependencyType::Influence(Weight::new(1, 28).as_f64().unwrap()),
    );
    network.add_edge(
        &"p3".to_string(),
        &"p1".to_string(),
        8,
        DependencyType::Influence(Weight::new(2, 7).as_f64().unwrap()),
    );
    network.add_edge(
        &"p3".to_string(),
        &"p2".to_string(),
        9,
        DependencyType::Influence(Weight::new(2, 7).as_f64().unwrap()),
    );
    network.add_edge(
        &"a3".to_string(),
        &"p3".to_string(),
        10,
        DependencyType::Influence(Weight::new(1, 1).as_f64().unwrap()),
    );
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
    let mut network = construct_network();
    let info = format!("random walks with {:?} nodes, iter: 1", &network.node_count());
    c.bench_function(&info, move |b| b.iter(|| {
        run_random_walk(&network, 1, [0; 16])
    }));
}

criterion_group!(
    benches,
    bench_osrank_naive_on_small_network,
    bench_random_walk_on_csv,
    bench_osrank_naive_on_sample_csv
);
criterion_main!(benches);
