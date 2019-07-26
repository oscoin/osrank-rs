#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate ndarray;
extern crate petgraph;
extern crate rand;
extern crate sprs;

use crate::protocol_traits::graph::{Graph, GraphObject, Id, Metadata};
use crate::protocol_traits::ledger::LedgerView;
use crate::types::{Osrank, RandomWalk, RandomWalks, SeedSet};
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
pub fn random_walk<'a, L, G>(
    seed_set: Option<SeedSet>,
    network: &'a G,
    ledger_view: &L,
    iter: i32,
    get_weight: &Box<Fn(&<G::Edge as GraphObject>::Metadata) -> f64>,
) -> Result<WalkResult<'a, G, <G::Node as GraphObject>::Id>, OsrankError>
where
    L: LedgerView,
    G: Graph,
    Id<G::Node>: Clone + PartialEq,
{
    match seed_set {
        Some(_) => unimplemented!(),
        None => {
            let mut walks = RandomWalks::new();
            for i in network.nodes() {
                for _ in 0..iter {
                    let mut walk = RandomWalk::new();
                    walk.add_next(i.id().clone());
                    let mut current_node = i.id();
                    // TODO distinguish account/project
                    // TODO Should there be a safeguard so this doesn't run forever?
                    while rand::thread_rng().gen::<f64>()
                        < ledger_view.get_damping_factors().project
                    {
                        let neighbors = network.neighbours(&current_node);
                        match neighbors.choose_weighted(&mut rand::thread_rng(), |item| {
                            network
                                .lookup_edge_metadata(&item.id)
                                .and_then(|m| Some(get_weight(m)))
                                .unwrap()
                        }) {
                            Ok(next_edge) => {
                                walk.add_next(next_edge.target.clone());
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
    get_weight: Box<Fn(&<G::Edge as GraphObject>::Metadata) -> f64>,
    from_osrank: Box<(Fn(&G::Node, Osrank) -> Metadata<G::Node>)>,
) -> Result<(), OsrankError>
where
    L: LedgerView,
    G: Graph,
    Id<G::Node>: Clone + PartialEq,
{
    match seed_set {
        Some(_) => {
            // Phase1, rank the network and produce a NetworkView.
            let phase1 = random_walk(seed_set, &*network, ledger_view, iter, &get_weight)?;
            // Phase2, compute the osrank only on the NetworkView
            let phase2 = random_walk(None, &*phase1.network_view, ledger_view, iter, &get_weight)?;
            rank_network(&phase2.walks, &mut *network, ledger_view, &from_osrank)
        }
        None => {
            // Compute osrank on the full NetworkView
            let create_walks = random_walk(None, &*network, ledger_view, iter, &get_weight)?;
            rank_network(
                &create_walks.walks,
                &mut *network,
                ledger_view,
                &from_osrank,
            )
        }
    }
}

pub fn rank_network<'a, L, G: 'a>(
    random_walks: &RandomWalks<Id<G::Node>>,
    network_view: &'a mut G,
    ledger_view: &L,
    from_osrank: &Box<(Fn(&G::Node, Osrank) -> Metadata<G::Node>)>,
) -> Result<(), OsrankError>
where
    L: LedgerView,
    G: Graph,
    <G::Node as GraphObject>::Id: PartialEq + Clone,
{
    for node in network_view.nodes_mut() {
        let total_walks = random_walks.len();
        let node_visits = &random_walks.count_visits(node.id().clone());
        let rank = Fraction::from(1.0 - ledger_view.get_damping_factors().project)
            * Osrank::new(*node_visits as u32, total_walks as u32);

        node.set_metadata(from_osrank(&node, rank))
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol_traits::ledger::MockLedger;
    use crate::types::{Artifact, ArtifactType, DependencyType, Network, Weight};
    use num_traits::Zero;

    #[test]
    fn everything_ok() {
        // build the example network
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

        let mock_ledger = MockLedger::default();
        let get_weight = Box::new(|m: &DependencyType<f64>| *m.get_weight());
        let set_osrank = Box::new(|node: &Artifact<String>, rank| match node.get_metadata() {
            ArtifactType::Project { osrank: _ } => ArtifactType::Project { osrank: rank },
            ArtifactType::Account { osrank: _ } => ArtifactType::Account { osrank: rank },
        });

        assert_eq!(network.edge_count(), 11);
        assert_eq!(
            osrank_naive(None, &mut network, &mock_ledger, 10, get_weight, set_osrank).unwrap(),
            ()
        )
    }
}
