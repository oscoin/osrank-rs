extern crate either;

use crate::types::{Network, RandomWalks};

pub enum OsrankError {}

type SeedSet = ();

/// A view over the whole network
pub struct NetworkView {}

impl NetworkView {
    pub fn from_network(network: &Network) -> Self {
        unimplemented!();
    }
}

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

/// Naive version of the algorithm that given a full Network and a precomputed
/// set W of random walks, iterates over each edge of the Network and computes
/// the osrank.
pub fn osrank_naive(seed_set: SeedSet, network: &mut Network) -> Result<(), OsrankError> {
    // Phase1, rank the network and produce a NetworkView.
    let mut phase1 = random_walk(Some(seed_set), &NetworkView::from_network(network))?;
    // Phase2, compute the osrank only on the NetworkView
    let mut phase2 = random_walk(None, &mut phase1.network_view)?;
    rank_network(&phase2.walks, &mut phase2.network_view)
}

pub fn rank_network(
    random_walks: &RandomWalks,
    network_view: &mut NetworkView,
) -> Result<(), OsrankError> {
    unimplemented!()
}
