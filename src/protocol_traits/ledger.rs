#![allow(unknown_lints)]
#![warn(clippy::all)]

use crate::types::{DampingFactors, HyperParams, Tau, R};

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

    fn get_hyperparams(&self) -> &HyperParams;
    fn set_hyperparams(&mut self, new: HyperParams);

    /// Returns the "R" parameter from the paper, as an unsigned 32bit integer.
    fn get_random_walks_num(&self) -> &R;
    fn set_random_walks_num(&mut self, new: R);

    /// Returns the "Tau" parameter for the phase 1 pruning, i.e.
    /// the "pruning threshold" for the initial phase of the Osrank computation.
    fn get_tau(&self) -> &Tau;
    fn set_tau(&mut self, new: Tau);

    fn get_damping_factors(&self) -> &DampingFactors;
    fn set_damping_factors(&mut self, new: DampingFactors);
}

/// A `MockLedger` implementation, suitable for tests.
#[derive(Default)]
pub struct MockLedger {
    state: MockLedgerState,
}

pub struct MockLedgerState {
    params: HyperParams,
    factors: DampingFactors,
    r: R,
    tau: Tau,
}

impl Default for MockLedgerState {
    fn default() -> MockLedgerState {
        MockLedgerState {
            params: HyperParams::default(),
            factors: DampingFactors::default(),
            r: 10,
            tau: 0.0,
        }
    }
}

impl LedgerView for MockLedger {
    type State = MockLedgerState;

    fn get_hyperparams(&self) -> &HyperParams {
        &self.state.params
    }

    fn set_hyperparams(&mut self, new: HyperParams) {
        self.state.params = new
    }

    fn get_random_walks_num(&self) -> &R {
        &self.state.r
    }
    fn set_random_walks_num(&mut self, new: R) {
        self.state.r = new
    }

    fn get_tau(&self) -> &Tau {
        &self.state.tau
    }
    fn set_tau(&mut self, new: Tau) {
        self.state.tau = new
    }

    fn get_damping_factors(&self) -> &DampingFactors {
        &self.state.factors
    }

    fn set_damping_factors(&mut self, new: DampingFactors) {
        self.state.factors = new
    }
}
