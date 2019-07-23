#![allow(unknown_lints)]
#![warn(clippy::all)]

use crate::protocol_traits::graph::Graph;
use crate::protocol_traits::ledger::LedgerView;
use std::path::Path;

pub enum CsvImporterError {}

pub fn from_csv_files<G, L>(
    deps_csv_file: &Path,
    deps_meta_csv_file: &Path,
    contrib_csv_file: &Path,
    ledger_view: &L,
) -> Result<G, CsvImporterError>
where
    G: Graph,
    L: LedgerView,
{
    unimplemented!()
}
