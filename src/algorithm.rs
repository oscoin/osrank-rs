#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate ndarray;
extern crate petgraph;
extern crate rand;
extern crate sprs;

use crate::protocol_traits::graph::{Graph, GraphObject, Id, Metadata};
use crate::protocol_traits::ledger::LedgerView;
use crate::types::walk::{RandomWalk, RandomWalks, SeedSet};
use crate::types::Osrank;
use core::iter::Iterator;
use fraction::Fraction;
use num_traits::{One, Zero};
use rand::distributions::WeightedError;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::hash::Hash;

#[derive(Debug)]
pub enum OsrankError {}

#[derive(Debug)]
pub struct WalkResult<G, I>
where
    I: Eq + Hash,
{
    network_view: G,
    pub walks: RandomWalks<I>,
}

fn walks<'a, L, G: 'a, RNG>(
    starting_nodes: impl Iterator<Item = &'a Id<G::Node>>,
    network: &G,
    ledger_view: &L,
    mut rng: RNG,
    get_weight: &dyn Fn(&<G::Edge as GraphObject>::Metadata) -> f64,
) -> RandomWalks<Id<G::Node>>
where
    L: LedgerView,
    G: Graph,
    Id<G::Node>: Clone + Eq + Hash,
    RNG: Rng + SeedableRng,
{
    let mut walks = RandomWalks::new();

    for i in starting_nodes {
        for _ in 0..(*ledger_view.get_random_walks_num()) {
            let mut walk = RandomWalk::new(i.clone());
            let mut current_node = i;
            // TODO distinguish account/project
            // TODO Should there be a safeguard so this doesn't run forever?
            while rng.gen::<f64>() < ledger_view.get_damping_factors().project {
                let neighbors = network.neighbours(&current_node);
                match neighbors.choose_weighted(&mut rng, |item| {
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
    walks
}

// FIXME(adn) It should be possible to make this code parametric over
// Dependency<W>, for I have ran into a cryptic error about the SampleBorrow
// trait not be implemented, and wasn't able to immediately make the code
// typecheck.
pub fn random_walk<L, G, RNG>(
    seed_set: Option<SeedSet<Id<G::Node>>>,
    network: &G,
    ledger_view: &L,
    rng: RNG,
    get_weight: &dyn Fn(&<G::Edge as GraphObject>::Metadata) -> f64,
) -> Result<WalkResult<G, <G::Node as GraphObject>::Id>, OsrankError>
where
    L: LedgerView,
    G: Graph + Clone,
    Id<G::Node>: Clone + Eq + Hash,
    RNG: Rng + SeedableRng,
{
    match seed_set {
        Some(seeds) => {
            let walks = walks(seeds.seedset_iter(), network, ledger_view, rng, get_weight);
            let mut trusted_node_ids: Vec<&Id<G::Node>> = Vec::new();
            for node in network.nodes() {
                if rank_node::<L, G>(&walks, node.id().clone(), ledger_view) > Osrank::zero() {
                    trusted_node_ids.push(&node.id());
                }
            }
            Ok(WalkResult {
                network_view: network.subgraph_by_nodes(trusted_node_ids),
                walks,
            })
        }
        None => {
            let whole_network = (*network).clone(); // FIXME, terrible.
            let all_node_ids = network.nodes().map(|n| n.id());
            let res = WalkResult {
                network_view: whole_network,
                walks: walks(all_node_ids, network, ledger_view, rng, get_weight),
            };

            Ok(res)
        }
    }
}

/// Naive version of the algorithm that given a full Network and a precomputed
/// set W of random walks, iterates over each edge of the Network and computes
/// the osrank.
pub fn osrank_naive<L, G, RNG>(
    seed_set: Option<SeedSet<Id<G::Node>>>,
    network: &mut G,
    ledger_view: &L,
    initial_seed: <RNG as SeedableRng>::Seed,
    get_weight: Box<dyn Fn(&<G::Edge as GraphObject>::Metadata) -> f64>,
    from_osrank: Box<dyn Fn(&G::Node, Osrank) -> Metadata<G::Node>>,
) -> Result<(), OsrankError>
where
    L: LedgerView,
    G: Graph + Clone,
    Id<G::Node>: Clone + Eq + Hash,
    RNG: Rng + SeedableRng,
    <RNG as SeedableRng>::Seed: Clone,
{
    //NOTE(adn) The fact we are creating a new RNG every time we call
    // `random_walk` is deliberate and something to think about. We probably
    // want to "restart" the randomness from the initial seed every call to
    // `random_walk`, which means this function has to consume the RNG.
    match seed_set {
        Some(_) => {
            // Phase1, rank the network and produce a NetworkView.
            let phase1 = random_walk(
                seed_set,
                &*network,
                ledger_view,
                RNG::from_seed(initial_seed.clone()),
                &get_weight,
            )?;
            // Phase2, compute the osrank only on the NetworkView
            let phase2 = random_walk(
                None,
                &phase1.network_view,
                ledger_view,
                RNG::from_seed(initial_seed.clone()),
                &get_weight,
            )?;
            rank_network(&phase2.walks, &mut *network, ledger_view, &from_osrank)
        }
        None => {
            // Compute osrank on the full NetworkView
            let create_walks = random_walk(
                None,
                &*network,
                ledger_view,
                RNG::from_seed(initial_seed.clone()),
                &get_weight,
            )?;
            rank_network(
                &create_walks.walks,
                &mut *network,
                ledger_view,
                &from_osrank,
            )
        }
    }
}

fn rank_node<L, G>(
    random_walks: &RandomWalks<Id<G::Node>>,
    node_id: Id<G::Node>,
    ledger_view: &L,
) -> Osrank
where
    L: LedgerView,
    G: Graph,
    <G::Node as GraphObject>::Id: Eq + Clone + Hash,
{
    let total_walks = random_walks.len();
    let node_visits = random_walks.count_visits(&node_id);

    // Avoids division by 0
    if total_walks == 0 {
        Osrank::zero()
    } else {
        // We don't use Fraction::from(f64), because that generates some
        // big numbers for the numer & denom, which eventually cause overflow.
        // What we do instead, is to exploit the fact we have a probability
        // distribution between 0.0 and 1.0 and we use a simple formula to
        // convert a percent into a fraction.
        let percent_f64 = (1.0 - ledger_view.get_damping_factors().project) * 100.0;
        let rank = Fraction::new(percent_f64.round() as u64, 100u64)
            * Osrank::new(node_visits as u32, total_walks as u32);

        if rank > Osrank::one() {
            Osrank::one()
        } else {
            rank
        }
    }
}

pub fn rank_network<'a, L, G: 'a>(
    random_walks: &RandomWalks<Id<G::Node>>,
    network_view: &'a mut G,
    ledger_view: &L,
    from_osrank: &dyn Fn(&G::Node, Osrank) -> Metadata<G::Node>,
) -> Result<(), OsrankError>
where
    L: LedgerView,
    G: Graph,
    <G::Node as GraphObject>::Id: Eq + Clone + Hash,
{
    for node in network_view.nodes_mut() {
        let rank = rank_node::<L, G>(&random_walks, node.id().clone(), ledger_view);

        node.set_metadata(from_osrank(&node, rank))
    }
    Ok(())
}

#[cfg(test)]
mod tests {

    extern crate quickcheck;
    extern crate rand;
    extern crate rand_xorshift;

    use super::*;
    use crate::protocol_traits::ledger::MockLedger;
    use crate::types::mock::MockNetwork;
    use crate::types::network::{Artifact, ArtifactType, DependencyType, Network};
    use crate::types::Weight;
    use fraction::ToPrimitive;
    use num_traits::Zero;
    use quickcheck::{quickcheck, TestResult};
    use rand_xorshift::XorShiftRng;

    // Test that our osrank algorithm yield a probability distribution,
    // i.e. the sum of all the ranks equals 1.0 (modulo some rounding error)

    fn prop_osrank_is_approx_probability_distribution(mut graph: MockNetwork) -> TestResult {
        if graph.is_empty() {
            return TestResult::discard();
        }

        let mock_ledger = MockLedger::default();
        let get_weight = Box::new(|m: &DependencyType<f64>| *m.get_weight());
        let set_osrank = Box::new(|node: &Artifact<String>, rank| match node.get_metadata() {
            ArtifactType::Project { osrank: _ } => ArtifactType::Project { osrank: rank },
            ArtifactType::Account { osrank: _ } => ArtifactType::Account { osrank: rank },
        });

        let initial_seed = [0; 16];

        osrank_naive::<MockLedger, MockNetwork, XorShiftRng>(
            None,
            &mut graph,
            &mock_ledger,
            initial_seed,
            get_weight,
            set_osrank,
        )
        .unwrap();

        let rank_f64 = graph
            .nodes()
            .fold(Osrank::zero(), |mut acc, node| {
                acc += node.get_metadata().get_osrank();
                acc
            })
            .to_f64()
            .unwrap();

        // FIXME(adn) This is a fairly weak check, but so far it's the best
        // we can shoot for.
        TestResult::from_bool(rank_f64 > 0.0 && rank_f64 < 1.01)
    }

    #[test]
    fn osrank_is_approx_probability_distribution() {
        quickcheck(prop_osrank_is_approx_probability_distribution as fn(MockNetwork) -> TestResult);
    }

    #[test]
    fn everything_ok() {
        // build the example network
        let mut network = Network::default();
        for node in &["p1", "p2", "p3"] {
            network.add_node(
                node.to_string(),
                ArtifactType::Project {
                    osrank: Zero::zero(),
                },
            )
        }

        // Create the seed set from all projects
        let seed_set = SeedSet::from(vec!["p1".to_string(), "p2".to_string(), "p3".to_string()]);

        for node in &["a1", "a2", "a3", "isle"] {
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
                &edge.0.to_string(),
                &edge.1.to_string(),
                2,
                DependencyType::Influence(edge.2.as_f64().unwrap()),
            )
        }

        let mock_ledger = MockLedger::default();
        let get_weight = Box::new(|m: &DependencyType<f64>| *m.get_weight());
        let set_osrank = Box::new(|node: &Artifact<String>, rank| match node.get_metadata() {
            ArtifactType::Project { osrank: _ } => ArtifactType::Project { osrank: rank },
            ArtifactType::Account { osrank: _ } => ArtifactType::Account { osrank: rank },
        });

        assert_eq!(network.edge_count(), 11);

        // This is the insertion point of the Graph API. If we had a GraphAPI
        // "handle" in scope here, we could extract the seed from some state
        // and use it in the algorithm. Faking it for now.

        let initial_seed = [0; 16];

        assert_eq!(
            osrank_naive::<MockLedger, MockNetwork, XorShiftRng>(
                Some(seed_set),
                &mut network,
                &mock_ledger,
                initial_seed,
                get_weight,
                set_osrank
            )
            .unwrap(),
            ()
        );
        assert_eq!(
            network.nodes().fold(Vec::new(), |mut ranks, node| {
                // let bla = *node.get_metadata();
                ranks.push(format!("{}", *node));
                ranks
            }),
            vec![
                "id: p1 osrank: 0.1425",
                "id: p2 osrank: 0.2225",
                "id: p3 osrank: 0.1575",
                "id: a1 osrank: 0.08",
                "id: a2 osrank: 0.23",
                "id: a3 osrank: 0.055",
                "id: isle osrank: 0"
            ]
        );
    }

}
