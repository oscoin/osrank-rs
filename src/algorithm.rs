extern crate either;
extern crate ndarray;
extern crate sprs;

use crate::types::{Network, RandomWalks, RandomWalk, SeedSet, Artifact, ProjectAttributes, Dependency, Weight};

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
) -> Result<WalkResult, OsrankError> {
    unimplemented!()
}

pub fn random_walk_no_seed(
    network: &NetworkView,
) -> Result<WalkResult, OsrankError> {
    let mut walks = RandomWalks::new();
    for i in network.from_graph.node_indices() {
        // TODO(mb) number of iterations must be a variable
        for j in 0..4 {
            let mut walk = RandomWalk::new();
            walk.add_next(i);
            let mut current_node = i;
            // TODO number of steps for now has to be replaced by damping factor termination
            for t in 0..4 {
                // TODO choose next node at random
                match network.from_graph.neighbors(current_node).nth(0) {
                    Some(node_idx) => {
                        walk.add_next(node_idx);
                        current_node = node_idx
                    },
                    _ => break
                }
            }
            walks.add_walk(walk);
        }
    }
    // TODO return actual NetworkView
    let res = WalkResult {
        network_view: NetworkView::default(),
        walks: walks,
    };
    println!("{:?}", &res.walks);
    Ok(res)
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
) -> Result<(), OsrankError> {
    for node_idx in network_view.from_graph.node_indices() {
        let node_visits = random_walks.count_visits(node_idx);
        println!("counted {:?} for idx {:?}", node_visits, &node_idx);
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
    let mut walked = random_walk_no_seed(&network).unwrap();
    assert_eq!(rank_network(&walked.walks, &mut network).unwrap(),());
    assert_eq!(walked.walks.len(), 4*6);
}
