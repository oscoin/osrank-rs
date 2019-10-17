#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate oscoin_graph_api;
extern crate petgraph;
extern crate rand;
extern crate rayon;
extern crate sprs;

use crate::protocol_traits::graph::GraphExtras;
use crate::protocol_traits::ledger::{LedgerView, MockLedger};
use crate::types::mock::{Mock, MockAnnotator, MockNetwork};
use crate::types::network::Artifact;
use crate::types::walk::SeedSet;
use crate::types::Osrank;
use oscoin_graph_api::{Graph, GraphAlgorithm, GraphAnnotator, GraphObject, Id};
use rand::distributions::uniform::SampleUniform;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;

use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::AddAssign;

use super::OsrankError;

pub fn osrank_incremental<G, A>(
    _seed_set: Option<&SeedSet<Id<G::Node>>>,
    _network: &G,
    _annotator: &mut A,
    _ledger_view: &(impl LedgerView + Send + Sync),
    _rng: &(impl Rng + SeedableRng + Clone + Send + Sync),
    _to_annotation: &dyn Fn(&G::Node, Osrank) -> A::Annotation,
) -> Result<(), OsrankError>
where
    G: GraphExtras + Clone + Send + Sync,
    A: GraphAnnotator,
    Id<G::Node>: Clone + Eq + Hash + Send + Sync,
    <G as Graph>::Weight:
        Default + Clone + PartialOrd + for<'x> AddAssign<&'x G::Weight> + SampleUniform,
{
    unimplemented!("The incremental algorithm is not implemented yet.");
}

pub struct OsrankIncrementalAlgorithm<'a, G: 'a, L, A: 'a> {
    graph: PhantomData<G>,
    ledger: PhantomData<L>,
    ty: PhantomData<&'a ()>,
    annotator: PhantomData<A>,
}

impl<'a, G, L, A> Default for OsrankIncrementalAlgorithm<'a, G, L, A> {
    fn default() -> Self {
        OsrankIncrementalAlgorithm {
            graph: PhantomData,
            ledger: PhantomData,
            ty: PhantomData,
            annotator: PhantomData,
        }
    }
}

/// The `Context` that the osrank naive _mock_ algorithm will need.
pub struct OsrankIncrementalMockContext<'a, A, W>
where
    A: GraphAnnotator,
{
    /// The optional `SeetSet` to use.
    pub seed_set: Option<&'a SeedSet<Id<<MockNetwork<W> as Graph>::Node>>>,
    /// The `LedgerView` for this context, i.e. a `MockLedger`.
    pub ledger_view: MockLedger<W>,
    /// The `to_annotation` function is a "getter" function that given a
    /// generic `Node` "knows" how to extract an annotation out of an `Osrank`.
    /// This function is necessary to bridge the gap between the algorithm being
    /// written in a totally generic way and the need to convert from a
    /// fraction-like `Osrank` into an `A::Annotation`.
    pub to_annotation: &'a (dyn Fn(&<MockNetwork<W> as Graph>::Node, Osrank) -> A::Annotation),
}

impl<'a, W> Default for OsrankIncrementalMockContext<'a, MockAnnotator<MockNetwork<W>>, W>
where
    MockLedger<W>: Default,
    W: Clone,
{
    fn default() -> Self {
        OsrankIncrementalMockContext {
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

/// A *mock* implementation for the `OsrankIncrementalAlgorithm`, using the `Mock`
/// newtype wrapper.
///
/// Refer to the documentation for `osrank::types::mock::Mock` for the idea
/// behind the `Mock` type, but in brief we do not want to have an
/// implementation for the `OsrankIncrementalAlgorithm` that depends on *mock* data,
/// yet it's useful to do so in tests. If we were to write:
///
/// ```ignore, no_run
/// impl<'a, G, L> GraphAlgorithm<G> OsrankIncrementalAlgorithm<'a, G, L>
/// ...
/// ```
///
/// This was now preventing us from defining a trait implementation for the
/// *real* algorithm, using *real* data. This is why the `Mock` wrapper is so
/// handy: `Mock a` is still isomorphic to `a`, but allows us to
/// define otherwise-conflicting instances.
impl<'a, L, W, A> GraphAlgorithm<MockNetwork<W>, A>
    for Mock<OsrankIncrementalAlgorithm<'a, MockNetwork<W>, L, A>>
where
    L: LedgerView + Send + Sync,
    //Id<G::Node>: Clone + Eq + Hash + Send + Sync,
    W: Default + Clone + PartialOrd + for<'x> AddAssign<&'x W> + SampleUniform + Send + Sync,
    OsrankIncrementalMockContext<'a, A, W>: Default,
    A: GraphAnnotator,
{
    type Output = ();
    type Context = OsrankIncrementalMockContext<'a, A, W>;
    type Error = OsrankError;
    type RngSeed = [u8; 32];
    type Annotation = <A as GraphAnnotator>::Annotation;

    fn execute(
        &self,
        ctx: &mut Self::Context,
        graph: &MockNetwork<W>,
        annotator: &mut A,
        initial_seed: <Xoshiro256StarStar as SeedableRng>::Seed,
    ) -> Result<Self::Output, Self::Error> {
        let rng = <Xoshiro256StarStar as SeedableRng>::from_seed(initial_seed);
        osrank_incremental(
            ctx.seed_set,
            graph,
            annotator,
            &ctx.ledger_view,
            &rng,
            &ctx.to_annotation,
        )
    }
}
