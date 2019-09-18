#![macro_use]
extern crate criterion;
extern crate oscoin_graph_api;
extern crate rand;
extern crate rand_xorshift;

use criterion::*;

use osrank::benchmarks::util::{construct_network, nightly, run_osrank_naive, run_random_walk};
use osrank::protocol_traits::graph::GraphExtras;

// The following benchmarks are very slow and intended to be run on a nighlty CI. For local testing
// run the benchmarks with `cargo bench -- dev` to avoid them
fn bench_nightly_osrank_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group(&nightly("Increasing node count"));
    group.sample_size(10);

    // for iter in &[1, 10, 100, 1_000] {
    for iter in &[1, 10] {
        for count in &[1_001, 2_501, 5_001, 7_501, 10_001, 15_001] {
            let mut network = construct_network(*count as usize, 0);
            let nodes = &network.node_count();
            group.bench_function(
                BenchmarkId::new(format!("Iter {}", &iter), nodes),
                move |b| b.iter(|| run_osrank_naive(&mut network, *iter as u32, [0; 32])),
            );
        }

        // for count in &[5_000, 10_000, 19_370] {
        //     let mut network = construct_network(16_220, *count as usize);
        //     let nodes = &network.node_count();
        //     group.bench_function(
        //         BenchmarkId::new(format!("Iter {}", &iter), nodes),
        //         move |b| b.iter(|| run_osrank_naive(&mut network, *iter as u32, [0; 32])),
        //     );
        // }
    }
    group.finish();
}

fn bench_nightly_random_walk(c: &mut Criterion) {
    let mut group = c.benchmark_group(&nightly(
        "Random walks with increasing iterator and node count",
    ));
    group.sample_size(10);

    for iter in &[1, 10, 100, 1_000] {
        for count in &[1_001, 2_501, 5_001, 7_501, 10_001, 15_001] {
            let network = construct_network(*count as usize, 0);
            let nodes = &network.node_count();
            group.bench_function(
                BenchmarkId::new(format!("Iter {}", &iter), nodes),
                move |b| b.iter(|| run_random_walk(&network, *iter, [0; 32])),
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_nightly_osrank_naive // bench_nightly_random_walk
);
criterion_main!(benches);
