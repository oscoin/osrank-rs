#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate ndarray;
extern crate petgraph;
extern crate rand;
extern crate sprs;

use crate::protocol_traits::graph::Graph;
use crate::protocol_traits::ledger::LedgerView;
use crate::types::{Artifact, Dependency, Osrank, RandomWalk, RandomWalks, SeedSet};
use fraction::Fraction;
use rand::distributions::WeightedError;
use rand::seq::SliceRandom;
use rand::Rng;

#[derive(Debug)]
pub enum OsrankError {}

#[derive(Debug)]
pub struct WalkResult<'a, G, I> {
    network_view: &'a G,
    walks: RandomWalks<I>,
}

// FIXME(adn) It should be possible to make this code parametric over
// Dependency<W>, for I have ran into a cryptic error about the SampleBorrow
// trait not be implemented, and wasn't able to immediately make the code
// typecheck.
pub fn random_walk<L, G>(
    seed_set: Option<SeedSet>,
    network: &'a G,
    ledger_view: &L,
    iter: i32,
) -> Result<WalkResult<'a, G, G::NodeId>, OsrankError>
where
    L: LedgerView,
    G: Graph<EdgeMetadata = Dependency<f64>>,
    G::NodeId: PartialEq,
{
    match seed_set {
        Some(_) => unimplemented!(),
        None => {
            let mut walks = RandomWalks::new();
            for i in network.node_ids() {
                for _ in 0..iter {
                    let mut walk = RandomWalk::new();
                    walk.add_next(i);
                    let mut current_node = i;
                    // TODO distinguish account/project
                    // TODO Should there be a safeguard so this doesn't run forever?
                    while rand::thread_rng().gen::<f64>()
                        < ledger_view.get_damping_factors().project
                    {
                        let neighbors = network.neighbours(&current_node);
                        match neighbors.choose_weighted(&mut rand::thread_rng(), |item| {
                            network
                                .lookup_edge_metadata(&item.id)
                                .and_then(|m| Some(m.get_weight()))
                                .unwrap()
                        }) {
                            Ok(next_edge) => {
                                walk.add_next(next_edge.target);
                                current_node = next_edge.target;
                            }
                            Err(WeightedError::NoItem) => break,
                            Err(error) => panic!("Problem with the neighbors: {:?}", error),
                        }
                    }
                    walks.add_walk(walk);
                }
            }

            let res = WalkResult {
                network_view: network,
                walks,
            };

            Ok(res)
        }
    }
}

/// Naive version of the algorithm that given a full Network and a precomputed
/// set W of random walks, iterates over each edge of the Network and computes
/// the osrank.
pub fn osrank_naive<L, G>(
    seed_set: Option<SeedSet>,
    network: &mut G,
    ledger_view: &L,
    iter: i32,
) -> Result<(), OsrankError>
where
    L: LedgerView,
    G: Graph<NodeMetadata = Artifact, EdgeMetadata = Dependency>,
    G::NodeId: PartialEq,
{
    match seed_set {
        Some(_) => {
            // Phase1, rank the network and produce a NetworkView.
            let phase1 = random_walk(seed_set, &*network, ledger_view, iter)?;
            // Phase2, compute the osrank only on the NetworkView
            let phase2 = random_walk(None, &*phase1.network_view, ledger_view, iter)?;
            rank_network(&phase2.walks, &mut *network, ledger_view)
        },
        None => {
            // Compute osrank on the full NetworkView
            let create_walks = random_walk(None, &*network, ledger_view, iter)?;
            rank_network(&create_walks.walks, &mut *network, ledger_view)
        }
    }
}

pub fn rank_network<L, G>(
    random_walks: &RandomWalks<G::NodeId>,
    network_view: &mut G,
    ledger_view: &L,
) -> Result<(), OsrankError>
where
    L: LedgerView,
    G: Graph<NodeMetadata = Artifact>,
    G::NodeId: PartialEq,
{
    for node_idx in network_view.node_ids() {
        let total_walks = random_walks.len();
        let node_visits = &random_walks.count_visits(node_idx);
        let rank = Fraction::from(1.0 - ledger_view.get_damping_factors().project)
            * Osrank::new(
                *node_visits as u32,
                total_walks as u32,
            );

        network_view.update_node_metadata(
            &node_idx,
            Box::new(move |metadata| metadata.set_osrank(rank)),
        );
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol_traits::ledger::MockLedger;
    use crate::types::{
        AccountAttributes, Artifact, DampingFactors, Dependency, HyperParams, Network,
        ProjectAttributes, Weight,
    };
    use num_traits::Zero;

    #[test]
    fn everything_ok() {
        // build the example network
        let mut network = Network::default();
        let p1 = Artifact::Project(ProjectAttributes {
            id: "p1".to_string(),
            osrank: Zero::zero(),
        });
        let p2 = Artifact::Project(ProjectAttributes {
            id: "p2".to_string(),
            osrank: Zero::zero(),
        });
        let p3 = Artifact::Project(ProjectAttributes {
            id: "p3".to_string(),
            osrank: Zero::zero(),
        });
        let a1 = Artifact::Account(AccountAttributes {
            id: "a1".to_string(),
            osrank: Zero::zero(),
        });
        let a2 = Artifact::Account(AccountAttributes {
            id: "a2".to_string(),
            osrank: Zero::zero(),
        });
        let a3 = Artifact::Account(AccountAttributes {
            id: "a3".to_string(),
            osrank: Zero::zero(),
        });
        network.add_node(p1);
        network.add_node(p2);
        network.add_node(p3);
        network.add_node(a1);
        network.add_node(a2);
        network.add_node(a3);
        network.add_edge(
            0,
            3,
            Dependency::Influence(Weight::new(3, 7).as_f64().unwrap()),
        );
        network.add_edge(
            3,
            0,
            Dependency::Influence(Weight::new(1, 1).as_f64().unwrap()),
        );
        network.add_edge(
            0,
            1,
            Dependency::Influence(Weight::new(4, 7).as_f64().unwrap()),
        );
        network.add_edge(
            1,
            4,
            Dependency::Influence(Weight::new(1, 1).as_f64().unwrap()),
        );
        network.add_edge(
            4,
            1,
            Dependency::Influence(Weight::new(1, 3).as_f64().unwrap()),
        );
        network.add_edge(
            4,
            2,
            Dependency::Influence(Weight::new(2, 3).as_f64().unwrap()),
        );
        network.add_edge(
            2,
            4,
            Dependency::Influence(Weight::new(11, 28).as_f64().unwrap()),
        );
        network.add_edge(
            2,
            5,
            Dependency::Influence(Weight::new(1, 28).as_f64().unwrap()),
        );
        network.add_edge(
            2,
            0,
            Dependency::Influence(Weight::new(2, 7).as_f64().unwrap()),
        );
        network.add_edge(
            2,
            1,
            Dependency::Influence(Weight::new(2, 7).as_f64().unwrap()),
        );
        network.add_edge(
            5,
            2,
            Dependency::Influence(Weight::new(1, 1).as_f64().unwrap()),
        );

        let mock_ledger = MockLedger {
            params: HyperParams::default(),
            factors: DampingFactors::default(),
        };

        assert_eq!(network.edge_count(), 11);
        assert_eq!(
            osrank_naive(None, &mut network, &mock_ledger, 10).unwrap(),
            ()
        );
    }
}
