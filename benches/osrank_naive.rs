#![macro_use]
extern crate criterion;
extern crate rand;
extern crate rand_xorshift;

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use osrank::protocol_traits::graph::{Graph, GraphObject};
use osrank::protocol_traits::ledger::MockLedger;
use osrank::types::{Artifact, ArtifactType, DependencyType, Network, Weight};
use osrank::algorithm::osrank_naive;
use num_traits::Zero;
use rand_xorshift::XorShiftRng;

type MockNetwork = Network<f64>;

fn construct_network() -> Network<f64> {
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

fn bench_osrank_naive(c: &mut Criterion) {
    let mut network = construct_network();
    c.bench_function("osrank 10 iterations", move |b| b.iter(|| {
        let mock_ledger = MockLedger::default();
        let get_weight = Box::new(|m: &DependencyType<f64>| *m.get_weight());
        let set_osrank = Box::new(|node: &Artifact<String>, rank| match node.get_metadata() {
            ArtifactType::Project { osrank: _ } => ArtifactType::Project { osrank: rank },
            ArtifactType::Account { osrank: _ } => ArtifactType::Account { osrank: rank },
        });
        let initial_seed = [0; 16];
        osrank_naive::<MockLedger, MockNetwork, XorShiftRng>(
            None,
            &mut network,
            &mock_ledger,
            10,
            initial_seed,
            get_weight,
            set_osrank)
    }));
}

criterion_group!(benches, bench_osrank_naive);
criterion_main!(benches);
