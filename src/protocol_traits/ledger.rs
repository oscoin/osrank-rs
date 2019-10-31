#![allow(unknown_lints)]
#![warn(clippy::all)]

use crate::types::{Weight, R};
use num_traits::Zero;
use oscoin_graph_api::types;

/// An Osrank-specific _view_ of a more general _Ledger_.
///
/// This trait offers an integration shim between a more general _Ledger_
/// trait and _Osrank_. The idea is that the _Ledger_ interface might deal
/// directly with an authenticated data structure (like a Merkle Tree), but
/// all we need from the _Ledger_ on the _Osrank_ side is the ability to get
/// and set the hyperparameters and the damping factors, without the
/// additional complications of a general key-value store.
///
/// # Integration with a real Ledger
///
/// As part of this trait we also provide an example `MockLedger`, to be
/// used in tests and prototypes. When the _real_ _Ledger_ will be finally
/// implemented, it's conceivable it will also provide a _Ledger_ trait which
/// specifies its API. Given a contrete type which implements the full
/// _Ledger_ trait, it should always be possible to implement also a simplified
/// _LedgerView_ on it, which means _Osrank_ is not coupled with the Ledger
/// interface, but only to this trait.
pub trait LedgerView {
    /// The associated state for this view. A fully-fledger Ledger might use
    /// something like a Merkelised State Store here, for example.
    type State;
    type W;

    fn get_hyperparams(&self) -> &types::HyperParameters<Self::W>;
    fn set_hyperparams(&mut self, new: types::HyperParameters<Self::W>);

    /// Get the hyper value associated to the input `EdgeType`. It panics at
    /// runtime if the value cannot be found.
    fn get_param(&self, edge_type_tag: &types::EdgeTypeTag) -> &Self::W {
        self.get_hyperparams().get_param(edge_type_tag)
    }

    /// Returns the "R" parameter from the paper, as an unsigned 32bit integer.
    fn get_random_walks_num(&self) -> &R;
    fn set_random_walks_num(&mut self, new: R);

    /// Returns the "Tau" parameter for the phase 1 pruning, i.e.
    /// the "pruning threshold" for the initial phase of the Osrank computation.
    fn get_tau(&self) -> &Self::W;
    fn set_tau(&mut self, new: Self::W);

    fn get_damping_factors(&self) -> &types::DampingFactors;
    fn set_damping_factors(&mut self, new: types::DampingFactors);
}

/// A `MockLedger` implementation, suitable for tests.
pub struct MockLedger<W> {
    state: MockLedgerState<W>,
}

pub struct MockLedgerState<W> {
    params: types::HyperParameters<W>,
}

impl Default for MockLedger<Weight> {
    fn default() -> MockLedger<Weight> {
        MockLedger {
            state: MockLedgerState {
                params: types::HyperParameters {
                    pruning_threshold: Weight::zero(),
                    damping_factors: types::DampingFactors {
                        project: 0.85,
                        account: 0.85,
                    },
                    r_value: 10,
                    edge_weights: [
                        (
                            types::EdgeTypeTag::ProjectToUserContribution,
                            Weight::new(1, 7),
                        ),
                        (
                            types::EdgeTypeTag::UserToProjectContribution,
                            Weight::new(2, 5),
                        ),
                        (
                            types::EdgeTypeTag::ProjectToUserMembership,
                            Weight::new(2, 7),
                        ),
                        (
                            types::EdgeTypeTag::UserToProjectMembership,
                            Weight::new(3, 5),
                        ),
                        (types::EdgeTypeTag::Dependency, Weight::new(4, 7)),
                    ]
                    .iter()
                    .cloned()
                    .collect(),
                },
            },
        }
    }
}

impl Default for MockLedger<f64> {
    fn default() -> MockLedger<f64> {
        MockLedger {
            state: MockLedgerState {
                params: types::HyperParameters {
                    pruning_threshold: 0.0,
                    damping_factors: types::DampingFactors {
                        project: 0.85,
                        account: 0.85,
                    },
                    r_value: 10,
                    edge_weights: [
                        (
                            types::EdgeTypeTag::ProjectToUserContribution,
                            Weight::new(1, 7).as_f64().unwrap(),
                        ),
                        (
                            types::EdgeTypeTag::UserToProjectContribution,
                            Weight::new(2, 5).as_f64().unwrap(),
                        ),
                        (
                            types::EdgeTypeTag::ProjectToUserMembership,
                            Weight::new(2, 7).as_f64().unwrap(),
                        ),
                        (
                            types::EdgeTypeTag::UserToProjectMembership,
                            Weight::new(3, 5).as_f64().unwrap(),
                        ),
                        (
                            types::EdgeTypeTag::Dependency,
                            Weight::new(4, 7).as_f64().unwrap(),
                        ),
                    ]
                    .iter()
                    .cloned()
                    .collect(),
                },
            },
        }
    }
}

impl<Wght> LedgerView for MockLedger<Wght> {
    type State = MockLedgerState<Wght>;
    type W = Wght;

    fn get_hyperparams(&self) -> &types::HyperParameters<Self::W> {
        &self.state.params
    }

    fn set_hyperparams(&mut self, new: types::HyperParameters<Self::W>) {
        self.state.params = new
    }

    fn get_random_walks_num(&self) -> &R {
        &self.state.params.r_value
    }
    fn set_random_walks_num(&mut self, new: R) {
        self.state.params.r_value = new
    }

    fn get_tau(&self) -> &Self::W {
        &self.state.params.pruning_threshold
    }
    fn set_tau(&mut self, new: Self::W) {
        self.state.params.pruning_threshold = new
    }

    fn get_damping_factors(&self) -> &types::DampingFactors {
        &self.state.params.damping_factors
    }

    fn set_damping_factors(&mut self, new: types::DampingFactors) {
        self.state.params.damping_factors = new
    }
}
