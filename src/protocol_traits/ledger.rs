#![allow(unknown_lints)]
#![warn(clippy::all)]

use crate::types::{DampingFactors, HyperParams};

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

    fn get_damping_factors(&self) -> &DampingFactors;
    fn set_damping_factors(&mut self, new: DampingFactors);
}

/// A `MockLedger` implementation, suitable for tests.
#[derive(Default)]
pub struct MockLedger {
    pub params: HyperParams,
    pub factors: DampingFactors,
}

impl LedgerView for MockLedger {
    type State = HyperParams;

    fn get_hyperparams(&self) -> &HyperParams {
        &self.params
    }

    fn set_hyperparams(&mut self, new: HyperParams) {
        self.params = new
    }

    fn get_damping_factors(&self) -> &DampingFactors {
        &self.factors
    }

    fn set_damping_factors(&mut self, new: DampingFactors) {
        self.factors = new
    }
}
