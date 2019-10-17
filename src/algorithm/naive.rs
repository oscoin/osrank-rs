#![allow(unknown_lints)]
#![warn(clippy::all)]

/// Implementation of a naive (but correct) version of the Osrank algorithm,
/// which walks the entire graph every time.
extern crate ndarray;
extern crate oscoin_graph_api;
extern crate petgraph;
extern crate rand;
extern crate rayon;
extern crate sprs;

use crate::protocol_traits::graph::GraphExtras;
use crate::protocol_traits::ledger::{LedgerView, MockLedger};
use crate::types::dynamic_weight::DynamicWeights;
use crate::types::mock::{Mock, MockAnnotator, MockNetwork};
use crate::types::network::Artifact;
use crate::types::walk::{RandomWalk, RandomWalks, SeedSet};
use crate::types::Osrank;
use core::iter::Iterator;
use fraction::Fraction;
use num_traits::{One, Zero};
use oscoin_graph_api::{
    types, Direction, Edge, EdgeRef, Graph, GraphAlgorithm, GraphAnnotator, GraphObject, Id, Node,
};
use rand::distributions::WeightedError;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;

use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use super::OsrankError;

#[derive(Debug)]
/// The output from a random walk.
// TODO(adn) This type is internal as `random_walk` is the only place where
// we return it. However, we need to use `random_walk` in the benchmarks and
// thus this has to be `pub`, which is not very satisfactory.
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
    L: LedgerView<W = G::Weight>,
    <G as Graph>::Node: Node<types::NodeData<Osrank>>,
    <G as Graph>::Edge: Edge<
        <G as Graph>::Weight,
        <<G as Graph>::Node as GraphObject>::Id,
        types::EdgeData<<G as Graph>::Weight>,
    >,
    G: GraphExtras<
            NodeData = types::NodeData<Osrank>,
            EdgeData = types::EdgeData<<G as Graph>::Weight>,
        > + DynamicWeights
        + Send
        + Sync,
    Id<G::Node>: Clone + Eq + Hash + Send + Sync,
    RNG: Rng + SeedableRng + Clone + Send + Sync,
    <G as Graph>::Weight: Copy + Into<f64> + Send + Sync,
{
    let r_value = *ledger_view.get_random_walks_num();
    let prj_epsilon = ledger_view.get_damping_factors().project;
    let acc_epsilon = ledger_view.get_damping_factors().account;
    let hyperparams = ledger_view.get_hyperparams();

    let res = starting_nodes
        .into_par_iter()
        .map(move |i| {
            let mut thread_walks = RandomWalks::new();
            let mut hasher = DefaultHasher::new();

            // ATTENTION: The accuracy (or lack thereof) of the final ranks
            // depends on the goodness of the input RNG. In particular, using
            // a poor RNG for testing might result in an osrank which is > 1.0.

            // NOTE: This is *not* a cryptographic-safe implementation, as what
            // we really need here is a splittable, cryptographic RNG, so that
            // we can give each thread a different RNG and be sure that the
            // generate f64s are all different. What I have implemented below
            // is a sort of crappy version of `SplitMix`.

            i.hash(&mut hasher);
            let mut thread_rng: RNG = SeedableRng::from_rng(RNG::seed_from_u64(
                hasher.finish() ^ (rng.clone().gen::<u64>()),
            ))?;

            for _ in 0..r_value {
                let mut walk = RandomWalk::new(i.clone());
                let mut current_node_id = i;
                let mut damping_factor =
                    node_damping_factor(network, &current_node_id, prj_epsilon, acc_epsilon);

                // TODO Should there be a safeguard so this doesn't run forever?
                while thread_rng.gen::<f64>() < damping_factor {
                    // "blue phase", we select the edge type using the
                    // hyperparams as probability.

                    let mut possible_edge_types = BTreeSet::new();

                    for eref in network.edges_directed(&current_node_id, Direction::Outgoing) {
                        possible_edge_types.insert(eref.edge_type);
                    }

                    match possible_edge_types
                        .into_iter()
                        .collect::<Vec<_>>()
                        .as_slice()
                        .choose_weighted(&mut thread_rng, |edge_type| {
                            let hyper_value = hyperparams.get_param(&edge_type.to_tag());
                            let w: f64 = (*hyper_value).into();
                            w
                        }) {
                        Ok(selected_edge_type) => {
                            // "Red phase". At this point we have to compute the probability, based on the
                            // edge type. We first select all the outgoing edges of the same type,
                            // and repeat the process.

                            let edges_same_type = network
                                .edges_directed(current_node_id, Direction::Outgoing)
                                .into_iter()
                                .filter(|eref| eref.edge_type == *selected_edge_type)
                                .collect::<Vec<_>>();

                            match edges_same_type.as_slice().choose_weighted(
                                &mut thread_rng,
                                |item| match item.edge_type.to_tag() {
                                    types::EdgeTypeTag::ProjectToUserContribution => {
                                        weigh_contrib(network, item)
                                    }
                                    types::EdgeTypeTag::UserToProjectContribution => {
                                        weigh_contrib(network, item)
                                    }
                                    types::EdgeTypeTag::ProjectToUserMembership => {
                                        1.0 / edges_same_type.len() as f64
                                    }
                                    types::EdgeTypeTag::UserToProjectMembership => {
                                        1.0 / edges_same_type.len() as f64
                                    }
                                    types::EdgeTypeTag::Dependency => {
                                        1.0 / edges_same_type.len() as f64
                                    }
                                },
                            ) {
                                Ok(next_edge) => {
                                    walk.add_next(next_edge.to.clone());
                                    current_node_id = next_edge.to;
                                    // Update the damping factor for the next iteration of
                                    // the while loop.
                                    damping_factor = node_damping_factor(
                                        network,
                                        &current_node_id,
                                        prj_epsilon,
                                        acc_epsilon,
                                    );
                                }
                                Err(WeightedError::NoItem) => break,
                                Err(error) => panic!("Problem with the neighbors: {:?}", error),
                            }
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

/// Calculates the _damping factor_ for the input node.
fn node_damping_factor<G>(
    network: &G,
    current_node_id: &Id<G::Node>,
    prj_factor: f64,
    user_factor: f64,
) -> f64
where
    G: Graph,
{
    let current_node = network
        .get_node(current_node_id)
        .expect("Couldn't access not during random walk.");
    match &current_node.node_type() {
        types::NodeType::User { .. } => user_factor,
        types::NodeType::Project { .. } => prj_factor,
    }
}

fn weigh_contrib<G>(network: &G, item: &EdgeRef<Id<G::Node>, Id<G::Edge>>) -> f64
where
    <G as Graph>::Node: Node<types::NodeData<Osrank>>,
    <G as Graph>::Edge: Edge<
        <G as Graph>::Weight,
        <<G as Graph>::Node as GraphObject>::Id,
        types::EdgeData<<G as Graph>::Weight>,
    >,
    G: Graph<NodeData = types::NodeData<Osrank>, EdgeData = types::EdgeData<<G as Graph>::Weight>>,
{
    let selected_edge = network
        .get_edge(item.id)
        .expect("Couldn't access edge during random walk.");
    let source_node = network
        .get_node(selected_edge.source())
        .expect("walks: source node not found.");

    let p_x = selected_edge.edge_type().total_contributions();
    let n = source_node.node_type().total_contributions();

    if n == 0 {
        panic!("weigh_contrib: total contributions 'n' was 0. This would cause a division by 0.");
    }

    f64::from(p_x) / f64::from(n)
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
    L: LedgerView<W = G::Weight>,
    <G as Graph>::Node: Node<types::NodeData<Osrank>>,
    <G as Graph>::Edge: Edge<
        <G as Graph>::Weight,
        <<G as Graph>::Node as GraphObject>::Id,
        types::EdgeData<<G as Graph>::Weight>,
    >,
    G: GraphExtras<
            NodeData = types::NodeData<Osrank>,
            EdgeData = types::EdgeData<<G as Graph>::Weight>,
        > + Clone
        + DynamicWeights
        + Send
        + Sync,
    Id<G::Node>: Clone + Eq + Hash + Send + Sync,
    RNG: Rng + SeedableRng + Clone + Send + Sync,
    <G as Graph>::Weight: Copy + Into<f64> + Into<Osrank> + Send + Sync,
{
    match seed_set {
        Some(seeds) => {
            let walks = walks(seeds.seedset_iter().par_bridge(), network, ledger_view, rng)?;
            let mut trusted_node_ids: Vec<&Id<G::Node>> = Vec::new();
            for node in network.nodes() {
                if rank_node::<L, G>(&walks, &node.id(), &node.node_type(), ledger_view)
                    > (*ledger_view.get_tau()).into()
                {
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
    ledger_view: &(impl LedgerView<W = G::Weight>),
    rng: &(impl Rng + SeedableRng + Clone + Send + Sync),
    to_annotation: &dyn Fn(&G::Node, Osrank) -> A::Annotation,
) -> Result<(), OsrankError>
where
    <G as Graph>::Node: Node<types::NodeData<Osrank>>,
    <G as Graph>::Edge: Edge<
        <G as Graph>::Weight,
        <<G as Graph>::Node as GraphObject>::Id,
        types::EdgeData<<G as Graph>::Weight>,
    >,
    G: GraphExtras<
            NodeData = types::NodeData<Osrank>,
            EdgeData = types::EdgeData<<G as Graph>::Weight>,
        > + Clone
        + DynamicWeights
        + Send
        + Sync,
    A: GraphAnnotator,
    Id<G::Node>: Clone + Eq + Hash + Send + Sync,
    <G as Graph>::Weight: Copy + Into<f64> + Into<Osrank> + Send + Sync,
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
    node_id: &Id<G::Node>,
    node_type: &types::NodeType,
    ledger_view: &L,
) -> Osrank
where
    L: LedgerView<W = G::Weight>,
    G: GraphExtras,
    <G::Node as GraphObject>::Id: Eq + Clone + Hash + Sync + Send,
{
    // This is the total number of random walks, for *all* nodes, i.e. it's
    // `n` (the total number of nodes) multiplied by `R` (i.e. how many walks
    // we have to perform for each node). It correspond to `nR` in the formula.
    let total_walks = random_walks.len();

    let node_visits = random_walks.count_visits(&node_id);

    // Avoids division by 0
    if total_walks == 0 {
        Osrank::zero()
    } else {
        let damping_factors = ledger_view.get_damping_factors();
        let node_damping_factor = match node_type {
            types::NodeType::User { .. } => damping_factors.account,
            types::NodeType::Project { .. } => damping_factors.project,
        };

        // We don't use Fraction::from(f64), because that generates some
        // big numbers for the numer & denom, which eventually cause overflow.
        // What we do instead, is to exploit the fact we have a probability
        // distribution between 0.0 and 1.0 and we use a simple formula to
        // convert a percent into a fraction.
        let percent_f64 = (1.0 - node_damping_factor) * 100.0;

        let rank = Osrank(
            Fraction::new(percent_f64.round() as u64, 100u64)
                * Fraction::new(node_visits as u32, total_walks as u32),
        );

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
    L: LedgerView<W = G::Weight>,
    G: GraphExtras,
    A: GraphAnnotator,
    <G::Node as GraphObject>::Id: Eq + Clone + Hash + Sync + Send,
{
    for node in network_view.nodes() {
        annotator.annotate_graph(to_annotation(
            &node,
            rank_node::<L, G>(&random_walks, &node.id(), &node.node_type(), ledger_view),
        ))
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
pub struct OsrankNaiveMockContext<'a, W, A, G = MockNetwork<W>>
where
    G: Graph,
    A: GraphAnnotator,
{
    /// The optional `SeetSet` to use.
    pub seed_set: Option<&'a SeedSet<Id<G::Node>>>,
    /// The `LedgerView` for this context, i.e. a `MockLedger`.
    pub ledger_view: MockLedger<W>,
    /// The `to_annotation` function is a "getter" function that given a
    /// generic `Node` "knows" how to extract an annotation out of an `Osrank`.
    /// This function is necessary to bridge the gap between the algorithm being
    /// written in a totally generic way and the need to convert from a
    /// fraction-like `Osrank` into an `A::Annotation`.
    pub to_annotation: &'a (dyn Fn(&<G as Graph>::Node, Osrank) -> A::Annotation),
}

impl<'a, W> Default for OsrankNaiveMockContext<'a, W, MockAnnotator<MockNetwork<W>>, MockNetwork<W>>
where
    W: Clone,
    MockLedger<W>: Default,
{
    fn default() -> Self {
        OsrankNaiveMockContext {
            seed_set: None,
            ledger_view: MockLedger::default(),
            to_annotation: &mock_network_to_annotation,
        }
    }
}

fn mock_network_to_annotation(node: &Artifact<String, Osrank>, rank: Osrank) -> (String, Osrank) {
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
    <G as Graph>::Node: Node<types::NodeData<Osrank>>,
    <G as Graph>::Edge: Edge<
        <G as Graph>::Weight,
        <<G as Graph>::Node as GraphObject>::Id,
        types::EdgeData<<G as Graph>::Weight>,
    >,
    G: GraphExtras<
            NodeData = types::NodeData<Osrank>,
            EdgeData = types::EdgeData<<G as Graph>::Weight>,
        > + DynamicWeights
        + Clone
        + Send
        + Sync,
    L: LedgerView,
    Id<G::Node>: Clone + Eq + Hash + Send + Sync,
    <G as Graph>::Weight: Copy + Into<f64> + Into<Osrank> + Send + Sync,
    OsrankNaiveMockContext<'a, G::Weight, A, G>: Default,
    A: GraphAnnotator,
{
    type Output = ();
    type Context = OsrankNaiveMockContext<'a, G::Weight, A, G>;
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
    extern crate rand_xoshiro;

    use super::*;
    use crate::protocol_traits::ledger;
    use crate::types::mock;
    use crate::types::mock::Mock;
    use crate::types::network::Network;
    use crate::types::Weight;
    use crate::util::quickcheck::Vec32;
    use crate::util::{add_edges, add_projects, add_users, Pretty};
    use fraction::ToPrimitive;
    use num_traits::Zero;
    use oscoin_graph_api::{types, GraphAlgorithm};
    use quickcheck::{quickcheck, TestResult};
    use std::ops::Deref;

    type MockNetwork = mock::MockNetwork<Weight>;
    type MockAnnotator<N> = mock::MockAnnotator<N>;
    type MockLedger = ledger::MockLedger<Weight>;

    // Test that our osrank algorithm yield a probability distribution,
    // i.e. the sum of all the ranks equals 1.0 (modulo some rounding error)
    fn prop_osrank_is_approx_probability_distribution(_graph: Pretty<MockNetwork>) -> TestResult {
        let graph = _graph.unpretty;
        if graph.is_empty() {
            return TestResult::discard();
        }

        let initial_seed = [0; 32];

        let algo: Mock<OsrankNaiveAlgorithm<MockNetwork, MockLedger, MockAnnotator<MockNetwork>>> =
            Mock {
                unmock: OsrankNaiveAlgorithm::default(),
            };
        let mut ctx = OsrankNaiveMockContext::default();
        ctx.ledger_view.set_random_walks_num(1);
        let mut annotator: MockAnnotator<MockNetwork> = Default::default();

        assert_eq!(
            algo.execute(&mut ctx, &graph, &mut annotator, initial_seed),
            Ok(())
        );

        let rank = annotator
            .annotator
            .values()
            .fold(Fraction::zero(), |mut acc, value| {
                acc += *(value.deref());
                acc
            });

        let rank_f64 = Osrank(rank).to_f64().unwrap();

        let pred = rank_f64 > 0.0 && rank_f64 <= 1.0;

        assert_eq!(
            pred,
            true,
            "{}",
            format!(
                "Test failed! The total rank for this graph was > 1.2, specifically: {}",
                rank_f64
            )
        );

        TestResult::from_bool(pred)
    }

    // FIXME(adn) This test is currently ignored because we have no convergence
    // property to make sure the rank doesn't go > 1.0.
    fn _osrank_is_approx_probability_distribution() {
        quickcheck(
            prop_osrank_is_approx_probability_distribution as fn(Pretty<MockNetwork>) -> TestResult,
        );
    }

    // Test that our osrank algorithm ignores projects who do not meet tau threshold
    #[test]
    fn osrank_uses_tau() {
        // build the example network
        let network = build_spec_network();

        let initial_seed = [0; 32];
        let algo: Mock<OsrankNaiveAlgorithm<MockNetwork, MockLedger, MockAnnotator<MockNetwork>>> =
            Mock {
                unmock: OsrankNaiveAlgorithm::default(),
            };

        // Run Algo for Tau == 0
        let mut annotator1: MockAnnotator<MockNetwork> = Default::default();
        let mut ctx1 = OsrankNaiveMockContext::default();
        let seed_set1 = SeedSet::from(vec!["p1".to_string(), "p2".to_string(), "p3".to_string()]);
        ctx1.seed_set = Some(&seed_set1);
        ctx1.ledger_view.set_tau(Weight::zero());
        assert!(algo
            .execute(&mut ctx1, &network, &mut annotator1, initial_seed)
            .is_ok());

        // Run Algo for Non-Zero Tau
        let mut annotator2: MockAnnotator<MockNetwork> = Default::default();
        let mut ctx2 = OsrankNaiveMockContext::default();
        let seed_set2 = SeedSet::from(vec!["p1".to_string(), "p2".to_string(), "p3".to_string()]);
        ctx2.seed_set = Some(&seed_set2);
        ctx2.ledger_view.set_tau(Weight::new(1, 10));
        assert!(algo
            .execute(&mut ctx2, &network, &mut annotator2, initial_seed)
            .is_ok());

        let annotator1_ranks = annotator1
            .annotator
            .iter()
            .fold(Vec::new(), |mut ranks, info| {
                println!("id: {} osrank: {}", info.0, info.1.to_f64().unwrap());
                if info.1.to_f64().unwrap() > 0.0 {
                    ranks.push(format!(
                        "id: {} osrank: {}",
                        info.0,
                        info.1.to_f64().unwrap()
                    ));
                }
                ranks
            });

        let annotator2_ranks = annotator2
            .annotator
            .iter()
            .fold(Vec::new(), |mut ranks, info| {
                println!("id: {} osrank: {}", info.0, info.1.to_f64().unwrap());
                if info.1.to_f64().unwrap() > 0.0 {
                    ranks.push(format!(
                        "id: {} osrank: {}",
                        info.0,
                        info.1.to_f64().unwrap()
                    ));
                }
                ranks
            });

        // We should have more nodes in round one than round two!
        assert!(annotator1_ranks.len() > annotator2_ranks.len());
    }

    // Test that given the same initial seed, two osrank algorithms yields
    // exactly the same result.
    fn prop_osrank_is_deterministic(graph: MockNetwork, entropy: Vec32<u8>) -> TestResult {
        if graph.is_empty() {
            return TestResult::discard();
        }

        let initial_seed: &[u8; 32] = array_ref!(entropy.get_vec32.as_slice(), 0, 32);

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
        quickcheck(prop_osrank_is_deterministic as fn(MockNetwork, Vec32<u8>) -> TestResult);
    }

    /// Returns the network from the spec with an isolated node 'isle'.
    fn build_spec_network() -> MockNetwork {
        // build the example network
        let mut network = Network::default();

        add_projects(
            &mut network,
            [("p1", 100), ("p2", 30), ("p3", 80)].iter().cloned(),
        );
        add_users(
            &mut network,
            [("a1", 100), ("a2", 90), ("a3", 20), ("isle", 0)]
                .iter()
                .cloned(),
        );

        let edges = [
            (
                0,
                "p1",
                "a1",
                types::EdgeData {
                    weight: Weight::zero(),
                    edge_type: types::EdgeType::ProjectToUserContribution(100),
                },
            ),
            (
                1,
                "a1",
                "p1",
                types::EdgeData {
                    weight: Weight::zero(),
                    edge_type: types::EdgeType::UserToProjectContribution(100),
                },
            ),
            (
                2,
                "p1",
                "p2",
                types::EdgeData {
                    weight: Weight::zero(),
                    edge_type: types::EdgeType::Dependency,
                },
            ),
            (
                3,
                "p2",
                "a2",
                types::EdgeData {
                    weight: Weight::zero(),
                    edge_type: types::EdgeType::ProjectToUserContribution(30),
                },
            ),
            (
                4,
                "a2",
                "p2",
                types::EdgeData {
                    weight: Weight::zero(),
                    edge_type: types::EdgeType::UserToProjectContribution(30),
                },
            ),
            (
                5,
                "a2",
                "p3",
                types::EdgeData {
                    weight: Weight::zero(),
                    edge_type: types::EdgeType::UserToProjectContribution(60),
                },
            ),
            (
                6,
                "p3",
                "a2",
                types::EdgeData {
                    weight: Weight::zero(),
                    edge_type: types::EdgeType::ProjectToUserContribution(60),
                },
            ),
            (
                7,
                "p3",
                "a3",
                types::EdgeData {
                    weight: Weight::zero(),
                    edge_type: types::EdgeType::ProjectToUserContribution(20),
                },
            ),
            (
                8,
                "p3",
                "p1",
                types::EdgeData {
                    weight: Weight::zero(),
                    edge_type: types::EdgeType::Dependency,
                },
            ),
            (
                9,
                "p3",
                "p2",
                types::EdgeData {
                    weight: Weight::zero(),
                    edge_type: types::EdgeType::Dependency,
                },
            ),
            (
                10,
                "a3",
                "p3",
                types::EdgeData {
                    weight: Weight::zero(),
                    edge_type: types::EdgeType::UserToProjectContribution(20),
                },
            ),
        ];

        add_edges(&mut network, edges.iter().cloned());

        assert_eq!(network.edge_count(), 11);

        network
    }

    #[test]
    fn everything_ok() {
        let network = build_spec_network();

        // Create the seed set from all projects
        let seed_set = SeedSet::from(vec!["p1".to_string(), "p2".to_string(), "p3".to_string()]);

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

        // Assert that the dynamic weights are computed correctly.

        let dynamic_weights: Vec<Weight> = (0..11)
            .map(|x| {
                network.dynamic_weight(
                    network.get_edge(&x).unwrap(),
                    ctx.ledger_view.get_hyperparams(),
                )
            })
            .collect();

        assert_eq!(
            dynamic_weights,
            [
                Weight::new(1, 7),
                Weight::new(2, 5),
                Weight::new(4, 7),
                Weight::new(1, 7),
                Weight::new(2, 15),
                Weight::new(4, 15),
                Weight::new(3, 28),
                Weight::new(1, 28),
                Weight::new(2, 7),
                Weight::new(2, 7),
                Weight::new(2, 5),
            ]
        );

        assert_eq!(
            algo.execute(&mut ctx, &network, &mut annotator, initial_seed),
            Ok(())
        );

        // We need to sort the ranks because the order of the elements returned
        // by the `HashMap::iter` is not predictable.

        let total_rank: f64 = annotator
            .annotator
            .iter()
            .map(|info| info.1.to_f64().unwrap())
            .sum();

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

        //Assert that the total rank is <= 1.0
        assert_eq!(
            total_rank <= 1.0,
            true,
            "{}",
            format!(
                "Test failed! The total rank for this graph was > 1.0, specifically: {}",
                total_rank
            )
        );

        assert_eq!(
            expected,
            vec![
                "id: a1 osrank: 0.0575",
                "id: a2 osrank: 0.26",
                "id: a3 osrank: 0.0375",
                "id: isle osrank: 0",
                "id: p1 osrank: 0.125",
                "id: p2 osrank: 0.2425",
                "id: p3 osrank: 0.1525",
            ]
        );
    }

    #[test]
    fn mutual_dependency() {
        let mut network = Network::default();

        add_projects(&mut network, [("p1", 0), ("p2", 0)].iter().cloned());

        let edges = [
            (
                0,
                "p1",
                "p2",
                types::EdgeData {
                    weight: Weight::one(),
                    edge_type: types::EdgeType::Dependency,
                },
            ),
            (
                1,
                "p2",
                "p1",
                types::EdgeData {
                    weight: Weight::one(),
                    edge_type: types::EdgeType::Dependency,
                },
            ),
        ];

        add_edges(&mut network, edges.iter().cloned());

        let initial_seed = [0; 32];

        let algo: Mock<OsrankNaiveAlgorithm<MockNetwork, MockLedger, MockAnnotator<MockNetwork>>> =
            Mock {
                unmock: OsrankNaiveAlgorithm::default(),
            };

        let mut ctx = OsrankNaiveMockContext::default();
        ctx.ledger_view.set_random_walks_num(61);

        let mut annotator: MockAnnotator<MockNetwork> = Default::default();

        algo.execute(&mut ctx, &network, &mut annotator, initial_seed)
            .unwrap();

        let total_rank: Osrank = annotator.annotator.iter().map(|x| *x.1).sum();

        let mut expected = annotator
            .annotator
            .iter()
            .fold(Vec::new(), |mut ranks, info| {
                ranks.push(format!("id: {} osrank: {}", info.0, info.1));
                ranks
            });
        expected.sort();

        //Assert that the total rank is <= 1.0
        assert_eq!(
            total_rank <= Osrank::one(),
            true,
            "{}",
            format!(
                "Test failed! The total rank for this graph was > 1.0, specifically: {}",
                total_rank
            )
        );

        assert_eq!(
            expected,
            vec!["id: p1 osrank: 309/610", "id: p2 osrank: 30/61",]
        );
    }
}
