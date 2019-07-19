extern crate either;
extern crate ndarray;
extern crate sprs;
extern crate rand;
extern crate petgraph;

use rand::Rng;
use rand::distributions::WeightedError;
use rand::seq::SliceRandom;
use std::iter::FromIterator;
use crate::types::{Network, RandomWalks, RandomWalk, SeedSet, Artifact, ProjectAttributes, Dependency, Weight, Osrank, DampingFactors};
use petgraph::visit::EdgeRef;
use fraction::Fraction;

#[derive(Debug)]
pub enum OsrankError {}

/// A view over the whole network.
/// The spirit behind this time would be to efficiently capture only a subset
/// of a potentially very big Network graph, by storing only the data we are
/// interested working with at any given stage.
// #[derive(Debug, PartialEq, Eq)]
// pub struct NetworkView {}
//
// impl NetworkView {
//     pub fn from_network(network: &Network) -> Self {
//         unimplemented!();
//     }
// }

type NetworkView = Network;

#[derive(Debug)]
pub struct WalkResult {
    network_view: NetworkView,
    walks: RandomWalks,
}

pub fn random_walk(
    seed_set: Option<SeedSet>,
    network: &NetworkView,
    damping_factors: DampingFactors,
    iter: i32,
) -> Result<WalkResult, OsrankError> {
    match seed_set {
        Some(_) => unimplemented!(),
        None => {
            let mut walks = RandomWalks::new();
            for i in network.from_graph.node_indices() {
                for _ in 0..iter {
                    let mut walk = RandomWalk::new();
                    walk.add_next(i);
                    let mut current_node = i;
                    // TODO distinguish account/project
                    // TODO Should there be a safeguard so this doesn't run forever?
                    while rand::thread_rng().gen::<f64>() < damping_factors.project {
                        let neighbors = Vec::from_iter(network.from_graph.edges(current_node));
                        match neighbors.choose_weighted(&mut rand::thread_rng(), |item| item.weight().get_weight().as_f64().unwrap()) {
                            Ok(next_edge) => {
                                walk.add_next(next_edge.target());
                                current_node = next_edge.target();
                            },
                            Err(WeightedError::NoItem) => break,
                            Err(error) => panic!("Problem with the neighbors: {:?}", error),
                        }
                    }
                    walks.add_walk(walk);
                }
            }
            // TODO return actual NetworkView
            let res = WalkResult {
                network_view: NetworkView::default(),
                walks,
            };
            Ok(res)
        }
    }
}

/// Naive version of the algorithm that given a full Network and a precomputed
/// set W of random walks, iterates over each edge of the Network and computes
/// the osrank.
// pub fn osrank_naive(seed_set: SeedSet, network: &mut Network) -> Result<(), OsrankError> {
//     // Phase1, rank the network and produce a NetworkView.
//     let phase1 = random_walk(Some(seed_set), &NetworkView::from_network(network))?;
//     // Phase2, compute the osrank only on the NetworkView
//     let mut phase2 = random_walk(None, &phase1.network_view)?;
//     rank_network(&phase2.walks, &mut phase2.network_view)
// }

pub fn rank_network(
    random_walks: &RandomWalks,
    network_view: &mut NetworkView,
    damping_factors: DampingFactors,
) -> Result<(), OsrankError> {
    for node_idx in network_view.from_graph.node_indices() {
        let total_walks = random_walks.len();
        let node_visits = &random_walks.count_visits(node_idx);
        let rank = Fraction::from(damping_factors.project) * Osrank::new(*node_visits as u32, (total_walks * network_view.from_graph.node_indices().count()) as u32);
        network_view.from_graph[node_idx].set_osrank(Some(rank)) ;
    }
    Ok(())
}

#[test]
fn everything_ok() {
    // build the example network, for now all nodes are projects
    let mut network = NetworkView::default();
    let p1 = Artifact::Project(ProjectAttributes {
        id: "p1".to_string(),
        osrank: None,
    });
    let p2 = Artifact::Project(ProjectAttributes {
        id: "p2".to_string(),
        osrank: None,
    });
    let p3 = Artifact::Project(ProjectAttributes {
        id: "p3".to_string(),
        osrank: None,
    });
    let a1 = Artifact::Project(ProjectAttributes {
        id: "a1".to_string(),
        osrank: None,
    });
    let a2 = Artifact::Project(ProjectAttributes {
        id: "a2".to_string(),
        osrank: None,
    });
    let a3 = Artifact::Project(ProjectAttributes {
        id: "a3".to_string(),
        osrank: None,
    });
    network.add_artifact(p1);
    network.add_artifact(p2);
    network.add_artifact(p3);
    network.add_artifact(a1);
    network.add_artifact(a2);
    network.add_artifact(a3);
    network.unsafe_add_dependency(0,3,Dependency::Depend(Weight::new(3, 7)));
    network.unsafe_add_dependency(3,0,Dependency::Depend(Weight::new(1, 1)));
    network.unsafe_add_dependency(0,1,Dependency::Depend(Weight::new(4, 7)));
    network.unsafe_add_dependency(1,4,Dependency::Depend(Weight::new(1, 1)));
    network.unsafe_add_dependency(4,1,Dependency::Depend(Weight::new(1, 3)));
    network.unsafe_add_dependency(4,2,Dependency::Depend(Weight::new(2, 3)));
    network.unsafe_add_dependency(2,4,Dependency::Depend(Weight::new(11, 28)));
    network.unsafe_add_dependency(2,5,Dependency::Depend(Weight::new(1, 28)));
    network.unsafe_add_dependency(2,0,Dependency::Depend(Weight::new(2, 7)));
    network.unsafe_add_dependency(2,1,Dependency::Depend(Weight::new(2, 7)));
    network.unsafe_add_dependency(5,2,Dependency::Depend(Weight::new(1, 1)));

    assert_eq!(network.from_graph.edge_count(), 11);
    let walked = random_walk(None, &network, DampingFactors::default(), 10).unwrap();
    assert_eq!(rank_network(&walked.walks, &mut network, DampingFactors::default()).unwrap(),());
}
