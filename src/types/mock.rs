#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate csv as csv_lib;
extern crate oscoin_graph_api;

use crate::exporters::csv::{export_rank_to_csv, CsvExporterError};
use crate::exporters::Exporter;
use crate::importers::csv;
use crate::protocol_traits::ledger::{LedgerView, MockLedger};
use crate::types::network::Network;
use crate::types::Osrank;
use crate::util::quickcheck::{Alphanumeric, NonEmpty};
use fraction::ToPrimitive;
use num_traits::{Num, Signed};
use oscoin_graph_api::{Graph, GraphAnnotator, GraphObject};
use quickcheck::{Arbitrary, Gen};
use rand::Rng;
use std::collections::HashMap;
use std::hash::Hash;

/// A mock network is a network which uses `W` as edge weights and proper
/// fractions for the Osrank.
pub type MockNetwork<W> = Network<W, Osrank>;

/// Equivalent to `newtype Mock a = Mock a` in Haskell.
///
/// Useful for defining some trait which operates over mocks implementation only.
pub struct Mock<A> {
    pub unmock: A,
}

impl<W> Arbitrary for MockNetwork<W>
where
    MockLedger<W>: Default,
    W: Default + Arbitrary + Num + Copy + Clone + From<u32> + PartialOrd + Signed,
{
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let mock_ledger = MockLedger::default();
        let mut deps: String = String::from("FROM_ID,TO_ID\n");
        let mut deps_meta: String = String::from("ID,NAME,PLATFORM\n");
        let mut contrib: String = String::from("ID,MAINTAINER,REPO,CONTRIBUTIONS,NAME\n");

        // Max 5 projects.
        let prjs: NonEmpty<Alphanumeric> = Arbitrary::arbitrary(g);
        let random_projects: Vec<&Alphanumeric> =
            prjs.get_nonempty.iter().by_ref().take(5).collect();

        let random_dependencies: Vec<(usize, usize)> = arbitrary_dependencies(g, &random_projects);

        let random_contributors_names: NonEmpty<Alphanumeric> = Arbitrary::arbitrary(g);
        // Max 5 contributors
        let random_contributors: Vec<String> = random_contributors_names
            .get_nonempty
            .iter()
            .cloned()
            .take(5)
            .map(|c: Alphanumeric| {
                let mut s = c.get_alphanumeric;
                s.insert_str(0, "github@");
                s
            })
            .collect();

        let random_contributions: Vec<(usize, String, u32)> =
            arbitrary_contributions(g, &random_projects, random_contributors);

        for (dep_from, dep_to) in random_dependencies {
            deps.push_str(format!("{},{}\n", dep_from, dep_to).as_str());
        }

        for (prj_id, prj_name) in random_projects.iter().cloned().enumerate() {
            deps_meta.push_str(format!("{},{},Foo\n", prj_id, prj_name.get_alphanumeric).as_str());
        }

        for (prj_id, contrib_name, contributions) in random_contributions.iter() {
            contrib.push_str(
                format!(
                    "{},{},http://github.com/foo,{},{}\n",
                    prj_id, contrib_name, contributions, random_projects[*prj_id].get_alphanumeric
                )
                .as_str(),
            );
        }

        csv::import_network(
            csv_lib::ReaderBuilder::new()
                .flexible(true)
                .from_reader(deps.as_bytes()),
            csv_lib::ReaderBuilder::new()
                .flexible(true)
                .from_reader(deps_meta.as_bytes()),
            csv_lib::ReaderBuilder::new()
                .flexible(true)
                .from_reader(contrib.as_bytes()),
            None,
            &mock_ledger.get_hyperparams(),
        )
        .unwrap_or_else(|e| {
            panic!(
                "returned unexpected error when generating arbitrary MockNetwork: {}",
                e
            )
        })
    }
}

fn arbitrary_dependencies<G: Gen>(g: &mut G, xs: &[&Alphanumeric]) -> Vec<(usize, usize)> {
    let mut add_more_deps;
    let mut current_deps;
    let mut res = Vec::new();

    for (project_id, _) in xs.iter().enumerate() {
        add_more_deps = Arbitrary::arbitrary(g);
        current_deps = 0;

        // Limit the number of dependencies to 3 per project.
        while add_more_deps && current_deps <= 3 {
            let target_id = g.gen_range(0, xs.len());

            // avoid self-loops
            if target_id != project_id {
                res.push((project_id, target_id));
            }

            current_deps += 1;
        }
    }

    res
}

fn arbitrary_contributions<G: Gen>(
    g: &mut G,
    all_projects: &[&Alphanumeric],
    all_users: Vec<String>,
) -> Vec<(usize, String, u32)> {
    let mut add_more_contribs;
    let mut current_contribs;
    let mut res = Vec::new();

    for user_id in all_users.iter() {
        add_more_contribs = Arbitrary::arbitrary(g);
        current_contribs = 0;

        // Limit the number of contributions to 3 per user.
        while add_more_contribs && current_contribs <= 3 {
            let target_project_id = g.gen_range(0, all_projects.len());
            let user_contrib = g.gen_range(1, 100);

            res.push((target_project_id, user_id.clone(), user_contrib));
            current_contribs += 1;
        }
    }

    res
}

/// A mock `GraphAnnotator` that stores the state into a dictionary
/// (typically, an `HashMap`).
pub struct KeyValueAnnotator<K, V> {
    pub annotator: HashMap<K, V>,
}

impl<K, V> GraphAnnotator for KeyValueAnnotator<K, V>
where
    K: Eq + Hash,
{
    type Annotation = (K, V);
    fn annotate_graph(&mut self, note: Self::Annotation) {
        self.annotator.insert(note.0, note.1);
    }
}

/// A `MockAnnotator` monomorphic over a graph `G`.
pub type MockAnnotator<G> = KeyValueAnnotator<<<G as Graph>::Node as GraphObject>::Id, Osrank>;

impl Default for MockAnnotator<MockNetwork<f64>> {
    fn default() -> Self {
        KeyValueAnnotator {
            annotator: Default::default(),
        }
    }
}

pub struct MockAnnotatorCsvExporter<'a, W> {
    pub annotator: MockAnnotator<MockNetwork<W>>,
    pub out_path: &'a str,
}

impl<'a, W> MockAnnotatorCsvExporter<'a, W> {
    pub fn new(annotator: MockAnnotator<MockNetwork<W>>, out_path: &'a str) -> Self {
        MockAnnotatorCsvExporter {
            annotator,
            out_path,
        }
    }
}

impl<'a, W> Exporter for MockAnnotatorCsvExporter<'a, W> {
    type ExporterOutput = ();
    type ExporterError = CsvExporterError;
    fn export(self) -> Result<Self::ExporterOutput, Self::ExporterError> {
        export_rank_to_csv(
            self.annotator.annotator.into_iter(),
            Box::new(|v: Osrank| v.to_f64().unwrap_or(0.0)),
            self.out_path,
        )
    }
}
