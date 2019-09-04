#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate ndarray;
extern crate oscoin_graph_api;
extern crate petgraph;
extern crate rand;
extern crate sprs;

use crate::protocol_traits::graph::GraphExtras;
use crate::protocol_traits::ledger::{LedgerView, MockLedger};
use crate::types::mock::{Mock, MockNetwork};
use crate::types::network::{Artifact, ArtifactType};
use crate::types::walk::{RandomWalk, RandomWalks, SeedSet};
use crate::types::Osrank;
use core::iter::Iterator;
use fraction::Fraction;
use num_traits::{One, Zero};
use oscoin_graph_api::{Direction, Edge, Graph, GraphAlgorithm, GraphObject, Id};
use rand::distributions::uniform::SampleUniform;
use rand::distributions::WeightedError;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::AddAssign;

#[derive(Debug, PartialEq, Eq)]
pub enum OsrankError {
    UknownError,
}

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
    rng: &mut RNG,
) -> RandomWalks<Id<G::Node>>
where
    L: LedgerView,
    G: GraphExtras,
    Id<G::Node>: Clone + Eq + Hash,
    RNG: Rng + SeedableRng,
    <G as Graph>::Weight:
        Default + Clone + PartialOrd + for<'x> AddAssign<&'x G::Weight> + SampleUniform,
{
    let mut walks = RandomWalks::new();

    for i in starting_nodes {
        for _ in 0..(*ledger_view.get_random_walks_num()) {
            let mut walk = RandomWalk::new(i.clone());
            let mut current_node = i;
            // TODO distinguish account/project
            // TODO Should there be a safeguard so this doesn't run forever?
            while rng.gen::<f64>() < ledger_view.get_damping_factors().project {
                let neighbors = network.edges_directed(&current_node, Direction::Outgoing);
                match neighbors.choose_weighted(rng, |item| {
                    network
                        .get_edge(&item.id)
                        .and_then(|m| Some(m.weight()))
                        .unwrap()
                }) {
                    Ok(next_edge) => {
                        walk.add_next(next_edge.to.clone());
                        current_node = next_edge.to;
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
    seed_set: Option<&SeedSet<Id<G::Node>>>,
    network: &G,
    ledger_view: &L,
    rng: &mut RNG,
) -> Result<WalkResult<G, <G::Node as GraphObject>::Id>, OsrankError>
where
    L: LedgerView,
    G: GraphExtras + Clone,
    Id<G::Node>: Clone + Eq + Hash,
    RNG: Rng + SeedableRng,
    <G as Graph>::Weight:
        Default + Clone + PartialOrd + for<'x> AddAssign<&'x G::Weight> + SampleUniform,
{
    match seed_set {
        Some(seeds) => {
            let walks = walks(seeds.seedset_iter(), network, ledger_view, rng);
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
                walks: walks(all_node_ids, network, ledger_view, rng),
            };

            Ok(res)
        }
    }
}

/// Naive version of the algorithm that given a full Network and a precomputed
/// set W of random walks, iterates over each edge of the Network and computes
/// the osrank.
pub fn osrank_naive<G>(
    seed_set: Option<&SeedSet<Id<G::Node>>>,
    network: &mut G,
    ledger_view: &impl LedgerView,
    rng: &mut (impl Rng + SeedableRng),
    set_osrank: &dyn Fn(&mut G::Node, Osrank) -> (),
) -> Result<(), OsrankError>
where
    G: GraphExtras + Clone,
    Id<G::Node>: Clone + Eq + Hash,
    <G as Graph>::Weight:
        Default + Clone + PartialOrd + for<'x> AddAssign<&'x G::Weight> + SampleUniform,
{
    match seed_set {
        Some(_) => {
            // Phase1, rank the network and produce a NetworkView.
            let phase1 = random_walk(seed_set, &*network, ledger_view, rng)?;
            // Phase2, compute the osrank only on the NetworkView
            let phase2 = random_walk(None, &phase1.network_view, ledger_view, rng)?;
            rank_network(&phase2.walks, &mut *network, ledger_view, set_osrank)
        }
        None => {
            // Compute osrank on the full NetworkView
            let create_walks = random_walk(None, &*network, ledger_view, rng)?;
            rank_network(&create_walks.walks, &mut *network, ledger_view, set_osrank)
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
    G: GraphExtras,
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
    set_osrank: &dyn Fn(&mut G::Node, Osrank) -> (),
) -> Result<(), OsrankError>
where
    L: LedgerView,
    G: GraphExtras,
    <G::Node as GraphObject>::Id: Eq + Clone + Hash,
{
    for node in network_view.nodes_mut() {
        set_osrank(
            node,
            rank_node::<L, G>(&random_walks, node.id().clone(), ledger_view),
        );
    }
    Ok(())
}

/// Marker type used to implement
pub struct OsrankNaiveAlgorithm<'a, G: 'a, L> {
    graph: PhantomData<G>,
    ledger: PhantomData<L>,
    ty: PhantomData<&'a ()>,
}

impl<'a, G, L> Default for OsrankNaiveAlgorithm<'a, G, L> {
    fn default() -> Self {
        OsrankNaiveAlgorithm {
            graph: PhantomData,
            ledger: PhantomData,
            ty: PhantomData,
        }
    }
}

pub struct OsrankNaiveMockContext<'a, G = MockNetwork>
where
    G: Graph,
{
    pub seed_set: Option<&'a SeedSet<Id<G::Node>>>,
    pub ledger_view: MockLedger,
    pub from_osrank: &'a (dyn Fn(&mut G::Node, Osrank) -> ()),
}

impl<'a> Default for OsrankNaiveMockContext<'a, MockNetwork> {
    fn default() -> Self {
        OsrankNaiveMockContext {
            seed_set: None,
            ledger_view: MockLedger::default(),
            from_osrank: &mock_network_set_osrank,
        }
    }
}

fn mock_network_set_osrank(node: &mut Artifact<String>, rank: Osrank) {
    let new_data = match node.data() {
        ArtifactType::Project { osrank: _ } => ArtifactType::Project { osrank: rank },
        ArtifactType::Account { osrank: _ } => ArtifactType::Account { osrank: rank },
    };

    *node.data_mut() = new_data;
}

impl<'a, G, L> GraphAlgorithm<G> for Mock<OsrankNaiveAlgorithm<'a, G, L>>
where
    G: GraphExtras + Clone,
    L: LedgerView,
    Id<G::Node>: Clone + Eq + Hash,
    <G as Graph>::Weight:
        Default + Clone + PartialOrd + for<'x> AddAssign<&'x G::Weight> + SampleUniform,
    OsrankNaiveMockContext<'a, G>: Default,
{
    type Output = ();
    type Context = OsrankNaiveMockContext<'a, G>;
    type Error = OsrankError;
    type RngSeed = [u8; 16];

    fn execute(
        &self,
        ctx: &mut Self::Context,
        graph: &mut G,
        initial_seed: <XorShiftRng as SeedableRng>::Seed,
    ) -> Result<Self::Output, Self::Error> {
        let mut rng = <XorShiftRng as SeedableRng>::from_seed(initial_seed);
        osrank_naive(
            ctx.seed_set,
            graph,
            &ctx.ledger_view,
            &mut rng,
            &ctx.from_osrank,
        )
    }
}

#[cfg(test)]
mod tests {

    extern crate oscoin_graph_api;
    extern crate quickcheck;
    extern crate rand;
    extern crate rand_xorshift;

    use super::*;
    use crate::protocol_traits::ledger::MockLedger;
    use crate::types::mock::{Mock, MockNetwork};
    use crate::types::network::{ArtifactType, DependencyType, Network};
    use crate::types::Weight;
    use fraction::ToPrimitive;
    use num_traits::Zero;
    use oscoin_graph_api::{Graph, GraphAlgorithm, GraphWriter};
    use quickcheck::{quickcheck, TestResult};

    // Test that our osrank algorithm yield a probability distribution,
    // i.e. the sum of all the ranks equals 1.0 (modulo some rounding error)
    fn prop_osrank_is_approx_probability_distribution(mut graph: MockNetwork) -> TestResult {
        if graph.is_empty() {
            return TestResult::discard();
        }

        let initial_seed = [0; 16];

        let algo: Mock<OsrankNaiveAlgorithm<MockNetwork, MockLedger>> = Mock {
            unmock: OsrankNaiveAlgorithm::default(),
        };
        let mut ctx = OsrankNaiveMockContext::default();

        assert_eq!(algo.execute(&mut ctx, &mut graph, initial_seed), Ok(()));

        let rank_f64 = graph
            .nodes()
            .fold(Osrank::zero(), |mut acc, node| {
                acc += node.data().get_osrank();
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
                2,
                &edge.0.to_string(),
                &edge.1.to_string(),
                edge.2.as_f64().unwrap(),
                DependencyType::Influence(edge.2.as_f64().unwrap()),
            )
        }

        assert_eq!(network.edge_count(), 11);

        // NOTE(adn) In the "real world" the initial_seed will be provided
        // by who calls `.execute`, probably the ledger/protocol.
        let initial_seed = [0; 16];

        let algo: Mock<OsrankNaiveAlgorithm<MockNetwork, MockLedger>> = Mock {
            unmock: OsrankNaiveAlgorithm::default(),
        };

        let mut ctx = OsrankNaiveMockContext::default();
        ctx.seed_set = Some(&seed_set);

        assert_eq!(algo.execute(&mut ctx, &mut network, initial_seed), Ok(()));

        assert_eq!(
            network.nodes().fold(Vec::new(), |mut ranks, node| {
                ranks.push(format!("{}", *node));
                ranks
            }),
            vec![
                "id: p1 osrank: 0.1075",
                "id: p2 osrank: 0.2325",
                "id: p3 osrank: 0.2125",
                "id: a1 osrank: 0.0575",
                "id: a2 osrank: 0.2675",
                "id: a3 osrank: 0.065",
                "id: isle osrank: 0"
            ]
        );
    }

}
