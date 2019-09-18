#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate ndarray;
extern crate oscoin_graph_api;
extern crate petgraph;
extern crate rand;
extern crate rayon;
extern crate sprs;

use crate::protocol_traits::graph::GraphExtras;
use crate::protocol_traits::ledger::{LedgerView, MockLedger};
use crate::types::mock::{Mock, MockAnnotator, MockNetwork};
use crate::types::network::Artifact;
use crate::types::walk::{RandomWalk, RandomWalks, SeedSet};
use crate::types::Osrank;
use core::iter::Iterator;
use fraction::Fraction;
use num_traits::{One, Zero};
use oscoin_graph_api::{Direction, Edge, Graph, GraphAlgorithm, GraphAnnotator, GraphObject, Id};
use rand::distributions::uniform::SampleUniform;
use rand::distributions::WeightedError;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;

use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::AddAssign;

#[derive(Debug, PartialEq, Eq)]
/// Errors that the `osrank` algorithm might throw.
pub enum OsrankError {
    /// Generic, catch-all error for things which can go wrong during the
    /// algorithm.
    UnknownError,
    RngFailedToSplit(String),
}

impl From<rand::Error> for OsrankError {
    fn from(err: rand::Error) -> OsrankError {
        OsrankError::RngFailedToSplit(format!("{}", err))
    }
}

#[derive(Debug)]
/// The output from a random walk.
pub struct WalkResult<G, I>
where
    I: Eq + Hash + Sync + Send,
{
    network_view: G,
    pub walks: RandomWalks<I>,
}

fn walks<'a, L, G: 'a, RNG>(
    starting_nodes: impl IntoParallelIterator<Item = &'a Id<G::Node>>,
    network: &G,
    ledger_view: &L,
    rng: &RNG,
) -> Result<RandomWalks<Id<G::Node>>, OsrankError>
where
    L: LedgerView + Send + Sync,
    G: GraphExtras + Send + Sync,
    Id<G::Node>: Clone + Eq + Hash + Send + Sync,
    RNG: Rng + SeedableRng + Clone + Send + Sync,
    <G as Graph>::Weight:
        Default + Clone + PartialOrd + for<'x> AddAssign<&'x G::Weight> + SampleUniform,
{
    let res = starting_nodes
        .into_par_iter()
        .map(move |i| {
            let mut thread_walks = RandomWalks::new();

            // ATTENTION: The accuracy (or lack thereof) of the final ranks
            // depends on the goodness of the input RNG. In particular, using
            // a poor RNG for testing might result in an osrank which is > 1.0.
            // An example of this is the `XorShiftRng`. Quoting the documentation
            // for `from_rng`:
            //
            // "The master PRNG should be at least as high quality as the child PRNGs.
            // When seeding non-cryptographic child PRNGs, we recommend using a
            // different algorithm for the master PRNG (ideally a CSPRNG) to avoid
            // correlations between the child PRNGs. If this is not possible (e.g. forking using
            // small non-crypto PRNGs) ensure that your PRNG has a good mixing function on the
            // output or consider use of a hash function with from_seed.
            // Note that seeding XorShiftRng from another XorShiftRng provides an extreme example
            // of what can go wrong: the new PRNG will be a clone of the parent."
            //
            let mut thread_rng: RNG = SeedableRng::from_rng(rng.clone())?;

            for _ in 0..(*ledger_view.get_random_walks_num()) {
                let mut walk = RandomWalk::new(i.clone());
                let mut current_node = i;
                // TODO distinguish account/project
                // TODO Should there be a safeguard so this doesn't run forever?
                while thread_rng.gen::<f64>() < ledger_view.get_damping_factors().project {
                    let neighbors = network.edges_directed(&current_node, Direction::Outgoing);
                    match neighbors.choose_weighted(&mut thread_rng, |item| {
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
                thread_walks.add_walk(walk);
            }

            Ok(thread_walks)
        })
        .reduce_with(|acc, w2| match (acc, w2) {
            (Err(e), _) => Err(e),
            (_, Err(e)) => Err(e),
            (Ok(mut w1), Ok(w2)) => {
                w1.append(w2);
                Ok(w1)
            }
        });

    match res {
        None => Err(OsrankError::RngFailedToSplit(
            "The starting_nodes were empty.".to_string(),
        )),
        Some(w) => w,
    }
}

/// Performs a random walk over the input network `G`.
///
/// If a `SeedSet` is provided, this function will produce a collection of
/// _trusted nodes_ to be used for subsequent walks, otherwise the entire
/// network is returned.
pub fn random_walk<L, G, RNG>(
    seed_set: Option<&SeedSet<Id<G::Node>>>,
    network: &G,
    ledger_view: &L,
    rng: &RNG,
) -> Result<WalkResult<G, <G::Node as GraphObject>::Id>, OsrankError>
where
    L: LedgerView + Send + Sync,
    G: GraphExtras + Clone + Send + Sync,
    Id<G::Node>: Clone + Eq + Hash + Send + Sync,
    RNG: Rng + SeedableRng + Clone + Send + Sync,
    <G as Graph>::Weight:
        Default + Clone + PartialOrd + for<'x> AddAssign<&'x G::Weight> + SampleUniform,
{
    match seed_set {
        Some(seeds) => {
            let walks = walks(seeds.seedset_iter().par_bridge(), network, ledger_view, rng)?;
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
            let all_node_ids: Vec<&Id<G::Node>> = network.nodes().map(|n| n.id()).collect();
            let walks = walks(
                all_node_ids.into_iter().par_bridge(),
                network,
                ledger_view,
                rng,
            )?;
            let res = WalkResult {
                network_view: whole_network,
                walks,
            };

            Ok(res)
        }
    }
}

/// Naive version of the `osrank` algorithm
///
/// Given a full network `G` and an optional `SeedSet`, iterates over each
/// edge of the network and computes the `Osrank`.
pub fn osrank_naive<G, A>(
    seed_set: Option<&SeedSet<Id<G::Node>>>,
    network: &G,
    annotator: &mut A,
    ledger_view: &(impl LedgerView + Send + Sync),
    rng: &(impl Rng + SeedableRng + Clone + Send + Sync),
    to_annotation: &dyn Fn(&G::Node, Osrank) -> A::Annotation,
) -> Result<(), OsrankError>
where
    G: GraphExtras + Clone + Send + Sync,
    A: GraphAnnotator,
    Id<G::Node>: Clone + Eq + Hash + Send + Sync,
    <G as Graph>::Weight:
        Default + Clone + PartialOrd + for<'x> AddAssign<&'x G::Weight> + SampleUniform,
{
    match seed_set {
        Some(_) => {
            // Phase1, rank the network and produce a NetworkView.
            let phase1 = random_walk(seed_set, &*network, ledger_view, rng)?;

            // Phase2, compute the osrank only on the NetworkView
            let phase2 = random_walk(None, &phase1.network_view, ledger_view, rng)?;

            rank_network(
                &phase2.walks,
                &*network,
                ledger_view,
                annotator,
                to_annotation,
            )
        }
        None => {
            // Compute osrank on the full NetworkView
            let create_walks = random_walk(None, &*network, ledger_view, rng)?;
            rank_network(
                &create_walks.walks,
                &*network,
                ledger_view,
                annotator,
                to_annotation,
            )
        }
    }
}

/// Assigns an `Osrank` to a `Node`.
fn rank_node<L, G>(
    random_walks: &RandomWalks<Id<G::Node>>,
    node_id: Id<G::Node>,
    ledger_view: &L,
) -> Osrank
where
    L: LedgerView,
    G: GraphExtras,
    <G::Node as GraphObject>::Id: Eq + Clone + Hash + Sync + Send,
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

/// Assigns an `Osrank` to a network `G`.
pub fn rank_network<'a, L, G: 'a, A>(
    random_walks: &RandomWalks<Id<G::Node>>,
    network_view: &'a G,
    ledger_view: &L,
    annotator: &mut A,
    to_annotation: &dyn Fn(&G::Node, Osrank) -> A::Annotation,
) -> Result<(), OsrankError>
where
    L: LedgerView,
    G: GraphExtras,
    A: GraphAnnotator,
    <G::Node as GraphObject>::Id: Eq + Clone + Hash + Sync + Send,
{
    for node in network_view.nodes() {
        let rank = rank_node::<L, G>(&random_walks, node.id().clone(), ledger_view);
        let ann = to_annotation(&node, rank);
        annotator.annotate_graph(ann);
    }
    Ok(())
}

/// This is a marker type used to implement a valid instance for `GraphAlgorithm`.
///
/// The term "marker" comes from the standard Rust [nomenclature](https://doc.rust-lang.org/std/marker/index.html)
/// to indicate types that have as main purpose the one of "carrying information"
/// around. In this case, we do need a `G` and a `L` in scope, as well as a valid
/// lifetime `'a` to exist in scope when we implement `GraphAlgorithm`, but we
/// might not necessary have one at hand. This is where the [PhantomData](https://doc.rust-lang.org/std/marker/struct.PhantomData.html)
/// marker type comes into play. It has a trivial type constructor but it allows
/// us to carry a "witness" that we have a `G`, `L` and `'a` in scope. For the
/// astute Haskeller, this is essentially the same as
///
/// ```haskell, no_run
/// newtype PhantomData a = PhantomData
/// ```
///
pub struct OsrankNaiveAlgorithm<'a, G: 'a, L, A: 'a> {
    graph: PhantomData<G>,
    ledger: PhantomData<L>,
    ty: PhantomData<&'a ()>,
    annotator: PhantomData<A>,
}

impl<'a, G, L, A> Default for OsrankNaiveAlgorithm<'a, G, L, A> {
    fn default() -> Self {
        OsrankNaiveAlgorithm {
            graph: PhantomData,
            ledger: PhantomData,
            ty: PhantomData,
            annotator: PhantomData,
        }
    }
}

/// The `Context` that the osrank naive _mock_ algorithm will need.
pub struct OsrankNaiveMockContext<'a, A, G = MockNetwork>
where
    G: Graph,
    A: GraphAnnotator,
{
    /// The optional `SeetSet` to use.
    pub seed_set: Option<&'a SeedSet<Id<G::Node>>>,
    /// The `LedgerView` for this context, i.e. a `MockLedger`.
    pub ledger_view: MockLedger,
    /// The `to_annotation` function is a "getter" function that given a
    /// generic `Node` "knows" how to extract an annotation out of an `Osrank`.
    /// This function is necessary to bridge the gap between the algorithm being
    /// written in a totally generic way and the need to convert from a
    /// fraction-like `Osrank` into an `A::Annotation`.
    pub to_annotation: &'a (dyn Fn(&G::Node, Osrank) -> A::Annotation),
}

impl<'a> Default for OsrankNaiveMockContext<'a, MockAnnotator<MockNetwork>, MockNetwork> {
    fn default() -> Self {
        OsrankNaiveMockContext {
            seed_set: None,
            ledger_view: MockLedger::default(),
            to_annotation: &mock_network_to_annotation,
        }
    }
}

fn mock_network_to_annotation(node: &Artifact<String>, rank: Osrank) -> (String, Osrank) {
    let artifact_id = node.id().clone();
    (artifact_id, rank)
}

/// A *mock* implementation for the `OsrankNaiveAlgorithm`, using the `Mock`
/// newtype wrapper.
///
/// Refer to the documentation for `osrank::types::mock::Mock` for the idea
/// behind the `Mock` type, but in brief we do not want to have an
/// implementation for the `OsrankNaiveAlgorithm` that depends on *mock* data,
/// yet it's useful to do so in tests. If we were to write:
///
/// ```ignore, no_run
/// impl<'a, G, L> GraphAlgorithm<G> OsrankNaiveAlgorithm<'a, G, L>
/// ...
/// ```
///
/// This was now preventing us from defining a trait implementation for the
/// *real* algorithm, using *real* data. This is why the `Mock` wrapper is so
/// handy: `Mock a` is still isomorphic to `a`, but allows us to
/// define otherwise-conflicting instances.
impl<'a, G, L, A> GraphAlgorithm<G, A> for Mock<OsrankNaiveAlgorithm<'a, G, L, A>>
where
    G: GraphExtras + Clone + Send + Sync,
    L: LedgerView + Send + Sync,
    Id<G::Node>: Clone + Eq + Hash + Send + Sync,
    <G as Graph>::Weight:
        Default + Clone + PartialOrd + for<'x> AddAssign<&'x G::Weight> + SampleUniform,
    OsrankNaiveMockContext<'a, A, G>: Default,
    A: GraphAnnotator,
{
    type Output = ();
    type Context = OsrankNaiveMockContext<'a, A, G>;
    type Error = OsrankError;
    type RngSeed = [u8; 32];
    type Annotation = <A as GraphAnnotator>::Annotation;

    fn execute(
        &self,
        ctx: &mut Self::Context,
        graph: &G,
        annotator: &mut A,
        initial_seed: <Xoshiro256StarStar as SeedableRng>::Seed,
    ) -> Result<Self::Output, Self::Error> {
        let rng = <Xoshiro256StarStar as SeedableRng>::from_seed(initial_seed);
        osrank_naive(
            ctx.seed_set,
            graph,
            annotator,
            &ctx.ledger_view,
            &rng,
            &ctx.to_annotation,
        )
    }
}

#[cfg(test)]
mod tests {

    extern crate arrayref;
    extern crate oscoin_graph_api;
    extern crate quickcheck;
    extern crate rand;
    extern crate rand_xorshift;

    use super::*;
    use crate::protocol_traits::ledger::MockLedger;
    use crate::types::mock::{Mock, MockAnnotator, MockNetwork};
    use crate::types::network::{ArtifactType, DependencyType, Network};
    use crate::types::Weight;
    use fraction::ToPrimitive;
    use num_traits::Zero;
    use oscoin_graph_api::{GraphAlgorithm, GraphWriter};
    use quickcheck::{quickcheck, TestResult};

    // Test that our osrank algorithm yield a probability distribution,
    // i.e. the sum of all the ranks equals 1.0 (modulo some rounding error)
    fn prop_osrank_is_approx_probability_distribution(graph: MockNetwork) -> TestResult {
        if graph.is_empty() {
            return TestResult::discard();
        }

        let initial_seed = [0; 32];

        let algo: Mock<OsrankNaiveAlgorithm<MockNetwork, MockLedger, MockAnnotator<MockNetwork>>> =
            Mock {
                unmock: OsrankNaiveAlgorithm::default(),
            };
        let mut ctx = OsrankNaiveMockContext::default();
        let mut annotator: MockAnnotator<MockNetwork> = Default::default();

        assert_eq!(
            algo.execute(&mut ctx, &graph, &mut annotator, initial_seed),
            Ok(())
        );

        let rank_f64 = annotator
            .annotator
            .values()
            .fold(Osrank::zero(), |mut acc, value| {
                acc += *value;
                acc
            })
            .to_f64()
            .unwrap();

        // FIXME(adn) This is a fairly weak check, but so far it's the best
        // we can shoot for.
        TestResult::from_bool(rank_f64 > 0.0 && rank_f64 <= 1.0)
    }

    #[test]
    fn osrank_is_approx_probability_distribution() {
        quickcheck(prop_osrank_is_approx_probability_distribution as fn(MockNetwork) -> TestResult);
    }

    // Test that given the same initial seed, two osrank algorithms yields
    // exactly the same result.
    fn prop_osrank_is_deterministic(graph: MockNetwork, entropy: Vec<u8>) -> TestResult {
        if graph.is_empty() || entropy.len() < 32 {
            return TestResult::discard();
        }

        let initial_seed: &[u8; 32] = array_ref!(entropy.as_slice(), 0, 32);

        let algo: Mock<OsrankNaiveAlgorithm<MockNetwork, MockLedger, MockAnnotator<MockNetwork>>> =
            Mock {
                unmock: OsrankNaiveAlgorithm::default(),
            };

        let mut ctx1 = OsrankNaiveMockContext::default();
        let mut ctx2 = OsrankNaiveMockContext::default();
        let mut annotator1: MockAnnotator<MockNetwork> = Default::default();
        let mut annotator2: MockAnnotator<MockNetwork> = Default::default();

        let first_run = algo.execute(&mut ctx1, &graph, &mut annotator1, *initial_seed);
        let second_run = algo.execute(&mut ctx2, &graph, &mut annotator2, *initial_seed);

        assert_eq!(first_run, second_run);

        let mut first_ranks = annotator1
            .annotator
            .iter()
            .map(|(nid, rank)| ((*nid).clone(), *rank))
            .collect::<Vec<(String, Osrank)>>();
        first_ranks.sort_by(|a, b| a.0.cmp(&b.0)); // Compare by Ids.

        let mut second_ranks = annotator2
            .annotator
            .iter()
            .map(|(nid, rank)| ((*nid).clone(), *rank))
            .collect::<Vec<(String, Osrank)>>();
        second_ranks.sort_by(|a, b| a.0.cmp(&b.0)); // Compare by Ids.

        TestResult::from_bool(first_ranks == second_ranks)
    }

    #[test]
    fn osrank_is_deterministic() {
        quickcheck(prop_osrank_is_deterministic as fn(MockNetwork, Vec<u8>) -> TestResult);
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
        let initial_seed = [0; 32];

        let algo: Mock<OsrankNaiveAlgorithm<MockNetwork, MockLedger, MockAnnotator<MockNetwork>>> =
            Mock {
                unmock: OsrankNaiveAlgorithm::default(),
            };

        let mut ctx = OsrankNaiveMockContext::default();
        ctx.seed_set = Some(&seed_set);

        let mut annotator: MockAnnotator<MockNetwork> = Default::default();

        assert_eq!(
            algo.execute(&mut ctx, &network, &mut annotator, initial_seed),
            Ok(())
        );

        // We need to sort the ranks because the order of the elements returned
        // by the `HashMap::iter` is not predictable.
        let mut expected = annotator
            .annotator
            .iter()
            .fold(Vec::new(), |mut ranks, info| {
                ranks.push(format!(
                    "id: {} osrank: {}",
                    info.0,
                    info.1.to_f64().unwrap()
                ));
                ranks
            });
        expected.sort();

        assert_eq!(
            expected,
            vec![
                "id: a1 osrank: 0.0725",
                "id: a2 osrank: 0.24",
                "id: a3 osrank: 0.05",
                "id: isle osrank: 0",
                "id: p1 osrank: 0.12",
                "id: p2 osrank: 0.205",
                "id: p3 osrank: 0.1675",
            ]
        );
    }

}
