#![macro_use]
extern crate criterion;
extern crate oscoin_graph_api;
extern crate rand;
extern crate rand_xoshiro;

use criterion::*;

use osrank::algorithm::naive::{random_walk, rank_network};
use osrank::algorithm::Normalised;
use osrank::benchmarks::util::{
    construct_network, construct_network_small, construct_osrank_naive_algorithm, dev,
    run_osrank_naive, run_random_walk,
};
use osrank::protocol_traits::graph::GraphExtras;
use osrank::protocol_traits::ledger::{LedgerView, MockLedger};
use osrank::types::mock::MockNetwork;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

// Benchmarks intended to be run for development are appended with `(dev) `.
// `cargo bench -- dev` will only run them.
fn bench_osrank_naive_on_small_network(c: &mut Criterion) {
    let mut network = construct_network_small();
    c.bench_function(&dev("osrank by random seed"), move |b| {
        b.iter(|| {
            let rand_vec: [u8; 32] = rand::random();
            run_osrank_naive(&mut network, 1, rand_vec)
        })
    });
}

// run with a lower sample size to speed things up
fn bench_osrank_naive_on_sample_csv(c: &mut Criterion) {
    let mut network = construct_network(1_000, 10_000);
    let info = &dev(format!("osrank with {:?} nodes, iter: 1", &network.node_count()).as_str());
    c.bench(
        &info,
        Benchmark::new("sample size 10", move |b| {
            b.iter(|| run_osrank_naive(&mut network, 1, [0; 32]))
        })
        .sample_size(10),
    );
}

fn bench_random_walk_on_csv(c: &mut Criterion) {
    let network = construct_network(1_000, 10_000);
    let info = &dev(format!(
        "random walks with {:?} nodes, iter: 1",
        &network.node_count()
    )
    .as_str());
    c.bench_function(&info, move |b| {
        b.iter(|| run_random_walk(&network, 1, [0; 32]))
    });
}

fn bench_rank_network(c: &mut Criterion) {
    let mut network = construct_network(1_000, 10_000);

    let (_algo, mut annotator, mut ctx) = construct_osrank_naive_algorithm();
    ctx.ledger_view.set_random_walks_num(1);

    let walks = random_walk::<MockLedger, Normalised<MockNetwork>, Xoshiro256StarStar>(
        None,
        &network,
        &ctx.ledger_view,
        &Xoshiro256StarStar::from_seed([0; 32]),
    )
    .unwrap()
    .walks;
    let info = &dev(format!(
        "bench network of {:?} nodes with {:?} walks",
        &network.node_count(),
        &walks.len()
    )
    .as_str());
    c.bench(
        &info,
        Benchmark::new("sample size 10", move |b| {
            b.iter(|| {
                rank_network(
                    &walks,
                    &mut network,
                    &ctx.ledger_view,
                    &mut annotator,
                    &ctx.to_annotation,
                )
            })
        })
        .sample_size(10),
    );
}

criterion_group!(
    benches,
    bench_osrank_naive_on_small_network,
    bench_osrank_naive_on_sample_csv,
    bench_random_walk_on_csv,
    bench_rank_network,
);
criterion_main!(benches);
