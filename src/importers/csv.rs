#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate csv;
extern crate log;
extern crate num_traits;
extern crate oscoin_graph_api;
extern crate serde;
extern crate sprs;

use crate::adjacency::new_network_matrix;
use crate::linalg::{DenseMatrix, SparseMatrix};
use crate::protocol_traits::graph::GraphExtras;
use crate::types::network::{Artifact, Dependency};
use crate::types::{Osrank, Weight};
use core::fmt;
use num_traits::{Num, One, Signed, Zero};
use oscoin_graph_api::{types, Graph, GraphDataReader, GraphDataWriter, GraphObject, GraphWriter};
use serde::Deserialize;
use sprs::{CsMat, TriMat, TriMatBase};
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::rc::Rc;

//
// Types
//

type ProjectId = u32;
type ProjectName = String;
type Contributor = String;
type Contributions = HashMap<Rc<Contributor>, HashMap<ProjectName, u32>>;
// We need a way to map external indexes with an incremental index suitable
// for matrix manipulation, so we define this LocalMatrixIndex exactly for
// this purpose.
type LocalMatrixIndex = usize;

/// The `SparseMatrix` of the dependencies.
pub type DependencyMatrix<N> = SparseMatrix<N>;
/// The `SparseMatrix` of the contributions.
pub type ContributionMatrix<N> = SparseMatrix<N>;
/// The `SparseMatrix` of the maintainers.
pub type MaintenanceMatrix<N> = SparseMatrix<N>;

/// A single, deserialised row of the `{platform}_dependencies_meta.csv` file.
#[derive(Debug, Deserialize)]
pub struct DepMetaRow {
    pub id: u32,
    pub name: String,
    pub platform: String,
}

pub struct DependenciesMetadata {
    pub ids: HashSet<ProjectId>,
    pub labels: Vec<ProjectName>,
    pub project2index: HashMap<ProjectId, LocalMatrixIndex>,
}

impl DependenciesMetadata {
    pub fn new() -> Self {
        DependenciesMetadata {
            ids: Default::default(),
            labels: Default::default(),
            project2index: Default::default(),
        }
    }
}

impl Default for DependenciesMetadata {
    fn default() -> Self {
        DependenciesMetadata::new()
    }
}

pub struct ContributionsMetadata {
    pub rows: Vec<ContribRow>,
    pub contributors: HashSet<Rc<Contributor>>,
    pub contributor2index: HashMap<Rc<Contributor>, LocalMatrixIndex>,
}

impl ContributionsMetadata {
    pub fn new() -> Self {
        ContributionsMetadata {
            rows: Vec::default(),
            contributors: Default::default(),
            contributor2index: Default::default(),
        }
    }
}

impl Default for ContributionsMetadata {
    fn default() -> Self {
        ContributionsMetadata::new()
    }
}

/// A single, deserialised row of the `{platform}_contributors.csv` file.
#[derive(Debug, Deserialize)]
pub struct ContribRow {
    pub project_id: ProjectId,
    pub contributor: Contributor,
    pub repo: String,
    pub contributions: u32,
    pub project_name: ProjectName,
}

/// A single, deserialised row of the `{platform}_dependencies.csv` file.
#[derive(Debug, Deserialize)]
pub struct DepRow {
    pub from: ProjectId,
    pub to: ProjectId,
}

//
// Errors
//

/// Errors arising during the import process.
#[derive(Debug)]
pub enum CsvImportError {
    /// Returned in case of generic I/O error.
    IOError(std::io::Error),

    /// Returned when the CSV deserialisation failed.
    CsvDeserialisationError(csv::Error),
}

impl fmt::Display for CsvImportError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CsvImportError::IOError(e) => {
                write!(f, "i/o error when reading/writing on the CSV file {}", e)
            }
            CsvImportError::CsvDeserialisationError(e) => {
                write!(f, "Deserialisation failed on a CSV row {}", e)
            }
        }
    }
}

impl From<std::io::Error> for CsvImportError {
    fn from(err: std::io::Error) -> CsvImportError {
        CsvImportError::IOError(err)
    }
}

impl From<csv::Error> for CsvImportError {
    fn from(err: csv::Error) -> CsvImportError {
        CsvImportError::CsvDeserialisationError(err)
    }
}

//
// Utility Traits
//

pub trait DisplayAsF64 {
    fn to_f64(self) -> f64;
}

impl DisplayAsF64 for f64 {
    fn to_f64(self: f64) -> f64 {
        self
    }
}

impl DisplayAsF64 for Weight {
    fn to_f64(self: Weight) -> f64 {
        self.as_f64().unwrap()
    }
}

pub fn debug_sparse_matrix_to_csv<N>(
    matrix: &CsMat<N>,
    out_path: &str,
) -> Result<(), CsvImportError>
where
    N: DisplayAsF64 + Zero + Clone + Copy,
{
    debug_dense_matrix_to_csv(&matrix.to_dense(), out_path)
}

pub fn debug_dense_matrix_to_csv<N>(
    matrix: &DenseMatrix<N>,
    out_path: &str,
) -> Result<(), CsvImportError>
where
    N: DisplayAsF64 + Zero + Clone + Copy,
{
    let mut output_csv = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out_path)?;

    for row in matrix.genrows() {
        if let Some((last, els)) = row.as_slice().and_then(|e| e.split_last()) {
            for cell in els {
                output_csv.write_all(format!("{},", cell.to_f64()).as_str().as_bytes())?;
            }
            output_csv.write_all(format!("{}", last.to_f64()).as_str().as_bytes())?;
        }
        output_csv.write_all(b"\n")?;
    }

    Ok(())
}

/// The import context.
struct ImportCtx<G> {
    deps_meta: DependenciesMetadata,
    contribs_meta: ContributionsMetadata,
    graph: G,
    // Stores the global mapping between matrix indexes and node names,
    // which includes projects & accounts
    index2id: HashMap<usize, String>,
    // Stores the mapping between users and their total contributions to
    // *all* the projects.
    user2contributions: Contributions,
}

impl<G> ImportCtx<G>
where
    G: GraphDataReader<
            Node = Artifact<String, Osrank>,
            Edge = Dependency<usize, String, <G as Graph>::Weight>,
            NodeData = types::NodeData<Osrank>,
            EdgeData = types::EdgeData<<G as Graph>::Weight>,
        > + GraphDataWriter,
{
    fn new() -> ImportCtx<G> {
        ImportCtx {
            deps_meta: DependenciesMetadata::new(),
            contribs_meta: ContributionsMetadata::new(),
            graph: G::default(),
            index2id: HashMap::default(),
            user2contributions: HashMap::default(),
        }
    }

    /// Determines which kind of edge type this is by taking an holistic view
    /// over the entire graph.
    fn edge_type(
        &self,
        source: &<G::Node as GraphObject>::Id,
        target: &<G::Node as GraphObject>::Id,
    ) -> types::EdgeType {
        let source_type = self
            .graph
            .node_data(source)
            .expect("node_data not found for csv::edge_type.");
        let target_type = self
            .graph
            .node_data(target)
            .expect("node_data not found for csv::edge_type.");

        match (&source_type.node_type, &target_type.node_type) {
            (types::NodeType::User { .. }, types::NodeType::User { .. }) => {
                panic!("Impossible: account -> account connection found.")
            }
            (types::NodeType::User { .. }, types::NodeType::Project { .. }) => {
                // NOTE(adn) By the virtue of the fact we ignore maintenanceship
                // for now, this is always `contribᵒ` for now.
                let contribs = self.edge_contributions(source, target);
                types::EdgeType::UserToProjectContribution(contribs)
            }
            (types::NodeType::Project { .. }, types::NodeType::User { .. }) => {
                // NOTE(adn) Same considerations as per above..
                let contribs = self.edge_contributions(target, source);
                types::EdgeType::ProjectToUserContribution(contribs)
            }
            (types::NodeType::Project { .. }, types::NodeType::Project { .. }) => {
                types::EdgeType::Dependency
            }
        }
    }

    fn edge_contributions(
        &self,
        source: &<G::Node as GraphObject>::Id,
        target: &<G::Node as GraphObject>::Id,
    ) -> u32 {
        self.user2contributions
            .get(source)
            .and_then(|p| p.get(target))
            .map(|v| *v)
            .unwrap_or(0)
    }

    fn set_user_total_contribs(&mut self, user: &Contributor) {
        match self.user2contributions.get(user) {
            None => (),
            Some(prjs) => {
                let total: u32 = prjs.values().sum();

                if total > 0 {
                    match self.graph.node_data_mut(user).iter_mut().next() {
                        None => (),
                        Some(d) => d.node_type.set_contributions(total),
                    }
                }
            }
        }
    }

    fn update_projects_contributions(&mut self, user: &Contributor) {
        match self.user2contributions.get(user) {
            None => {}
            Some(contributed_projects) => {
                for (project_id, contribs) in contributed_projects.iter() {
                    match self.graph.node_data_mut(project_id) {
                        None => {}
                        Some(ref mut node_data) => node_data.node_type.add_contributions(*contribs),
                    }
                }
            }
        }
    }

    fn set_user_contribs_for_project(
        &mut self,
        contributor: &Rc<Contributor>,
        prj: &ProjectName,
        contrib_num: u32,
    ) {
        let prjs = self
            .user2contributions
            .entry(Rc::clone(contributor))
            .or_insert_with(HashMap::new);

        prjs.insert(prj.clone(), contrib_num);
    }
}

/// Constructs a `Network` graph from a list of CSVs. The structure of each
/// csv is important, and is documented below.
///
/// #deps_csv_file
///
/// This must be a csv file in this format:
///
/// ```ignore,no_run
/// FROM_ID,TO_ID
/// 30742,31187
/// 30742,31296
/// [..]
/// ```
///
/// #deps_meta_csv_file
/// This must be a csv file in this format:
///
/// ```ignore,no_run
/// ID,NAME,PLATFORM
/// 30742,acacia,Cargo
/// 30745,aio,Cargo
/// 30746,advapi32-sys,Cargo
/// [..]
/// ```
///
/// #contrib_csv_file
/// This must be a csv file in this format:
///
/// ```ignore,no_run
/// ID,MAINTAINER,REPO,CONTRIBUTIONS,NAME
/// 30742,github@aepsil0n,https://github.com/aepsil0n/acacia,118,acacia
/// 30743,github@emk,https://github.com/emk/abort_on_panic-rs,32,abort_on_panic
/// 30745,github@reem,https://github.com/reem/rust-aio,35,aio
/// [..]
/// ```
///
/// # Using the import_network to load the Cargo ecosystem
///
/// To use the `import_network` to import a `Network` relative to the *whole*
/// Rust ecosystem, simply run
///
/// ```rust, no_run
/// use osrank::protocol_traits::ledger::{MockLedger, LedgerView};
/// use osrank::importers::csv::import_network;
/// use osrank::types::mock::MockNetwork;
/// use std::fs::File;
///
/// let mut deps_csv_file = File::open("data/cargo_dependencies.csv").unwrap();
/// let mut deps_meta_csv_file = File::open("data/cargo_dependencies_meta.csv").unwrap();
/// let mut contribs_csv_file = File::open("data/cargo_contributions.csv").unwrap();
/// let mock_ledger = MockLedger::default();
/// let network = import_network::<MockNetwork<f64>, File>( csv::Reader::from_reader(deps_csv_file)
///                             , csv::Reader::from_reader(deps_meta_csv_file)
///                             , csv::Reader::from_reader(contribs_csv_file)
///                             , None
///                             , &mock_ledger.get_hyperparams());
/// assert_eq!(network.is_ok(), true);
/// ```
///
pub fn import_network<G, R>(
    deps_csv: csv::Reader<R>,
    deps_meta_csv: csv::Reader<R>,
    mut contribs_csv: csv::Reader<R>,
    //TODO(and) We want to consider maintainers at some point.
    _maintainers_csv_file: Option<csv::Reader<R>>,
    hyperparams: &types::HyperParameters<<G as Graph>::Weight>,
) -> Result<G, CsvImportError>
where
    R: Read,
    <G as Graph>::Weight: Default + Num + Copy + Clone + From<u32> + PartialOrd + Signed,
    G: GraphExtras<
            Node = Artifact<String, Osrank>,
            Edge = Dependency<usize, String, <G as Graph>::Weight>,
            NodeData = types::NodeData<Osrank>,
            EdgeData = types::EdgeData<<G as Graph>::Weight>,
        > + GraphWriter
        + GraphDataReader,
{
    debug!("Starting to import a Graph from the CSV files...");

    let mut ctx: ImportCtx<G> = ImportCtx::new();

    // Iterate once over the dependencies metadata and store the name and id.
    // We need to maintain some sort of mapping between the order of visit
    // (which will be used as the index in the matrix) and the project id.
    for result in deps_meta_csv.into_records().filter_map(|e| e.ok()) {
        let row: DepMetaRow = result.deserialize(None)?;
        let prj_id = row.name.clone();
        ctx.deps_meta.ids.insert(row.id);
        ctx.deps_meta.labels.push(row.name);
        ctx.deps_meta
            .project2index
            .insert(row.id, ctx.deps_meta.ids.len() - 1);
        ctx.index2id.insert(ctx.index2id.len(), prj_id.clone());

        // Add the projects as nodes in the graph.
        ctx.graph.add_node(
            prj_id,
            types::NodeData {
                node_type: types::NodeType::Project {
                    contributions_from_all_users: 0,
                },
                rank: Zero::zero(),
            },
        );
    }

    debug!(
        "Added all {} projects as nodes to the graph..",
        ctx.graph.node_count()
    );

    // Iterate once over the contributions and build a matrix where
    // rows are the project names and columns the (unique) contributors.
    for result in contribs_csv.records().filter_map(|e| e.ok()) {
        let row: ContribRow = result.deserialize(None)?;
        let contributor = Rc::new(row.contributor.clone());

        if ctx
            .contribs_meta
            .contributors
            .get(&row.contributor)
            .is_none()
        {
            ctx.contribs_meta
                .contributors
                .insert(Rc::clone(&contributor));
            ctx.contribs_meta.contributor2index.insert(
                Rc::clone(&contributor),
                ctx.contribs_meta.contributors.len() - 1,
            );

            ctx.index2id
                .insert(ctx.index2id.len(), Rc::clone(&contributor).to_string());

            ctx.graph.add_node(
                contributor.to_string(),
                types::NodeData {
                    node_type: types::NodeType::User {
                        contributions_to_all_projects: 0,
                    },
                    rank: Zero::zero(),
                },
            );
        }

        ctx.set_user_contribs_for_project(&contributor, &row.project_name, row.contributions);
        ctx.contribs_meta.rows.push(row);
    }

    // At this point we added to the graph all the contributor nodes, and we
    // have build a mapping between each user and a map of <project_id, u32>
    // contributions. At this point we need to iterate over the contribution_id
    // set and update the *projects* total_contributions correctly.

    let contributors = ctx.contribs_meta.contributors.clone();
    for contributor in &contributors {
        ctx.set_user_total_contribs(&contributor);
        ctx.update_projects_contributions(&contributor);
    }

    debug!(
        "Added all the {} contributions as nodes to the graph..",
        contributors.len()
    );

    let dep_adj_matrix = new_dependency_adjacency_matrix(&ctx.deps_meta, deps_csv)?;

    debug!("Generated dep_adj_matrix...");

    let con_adj_matrix = new_contribution_adjacency_matrix(&ctx.deps_meta, &ctx.contribs_meta)?;

    debug!("Generated con_adj_matrix...");

    //FIXME(adn) For now the maintenance matrix is empty.
    let maintainers_matrix = CsMat::zero((dep_adj_matrix.rows(), con_adj_matrix.cols()));

    let network_matrix = new_network_matrix(
        &dep_adj_matrix,
        &con_adj_matrix,
        &maintainers_matrix,
        hyperparams,
    );

    debug!("Generated the full graph adjacency matrix...");

    let mut invalid_connections = 0;

    for (current_edge_id, (&weight, (source, target))) in network_matrix.iter().enumerate() {
        // Here we still exploit the adjacency information, but we use it
        // to construct parallel edges.

        let source_id = &ctx.index2id.get(&source).unwrap();
        let target_id = &ctx.index2id.get(&target).unwrap();

        let edge_type = ctx.edge_type(source_id, target_id);

        let valid_connection = match edge_type {
            types::EdgeType::Dependency => true,
            _ => edge_type.total_contributions() > 0,
        };

        if valid_connection {
            ctx.graph.add_edge(
                current_edge_id,
                source_id,
                target_id,
                types::EdgeData {
                    edge_type,
                    // It doesn't matter which value we assign here, because we are
                    // not going to look at it during the algorithms (i.e. we will
                    // use the dynamic weight calculation).
                    weight,
                },
            );
        } else {
            invalid_connections += 1;
        }
    }

    debug!(
        "Generated a graph with {} nodes and {} edges, while filtering out {} invalid connections.",
        ctx.graph.node_count(),
        ctx.graph.edge_count(),
        invalid_connections
    );

    // Build a graph out of the matrix.
    Ok(ctx.graph)
}

/// Creates a (sparse) adjacency matrix for the dependencies.
/// Corresponds to the "cargo-dep-adj.csv" from the Python scripts.
pub fn new_dependency_adjacency_matrix<N, R>(
    deps_meta: &DependenciesMetadata,
    deps_csv: csv::Reader<R>,
) -> Result<DependencyMatrix<N>, CsvImportError>
where
    N: Num + Clone,
    R: Read,
{
    let mut dep_adj: TriMat<N> = TriMatBase::new((deps_meta.ids.len(), deps_meta.ids.len()));

    // Iterate through the dependencies, populating the matrix.
    for result in deps_csv.into_records().filter_map(|e| e.ok()) {
        let row: DepRow = result.deserialize(None)?;
        if let (Some(from_index), Some(to_index)) = (
            deps_meta.project2index.get(&row.from),
            deps_meta.project2index.get(&row.to),
        ) {
            dep_adj.add_triplet(*from_index as usize, *to_index as usize, One::one());
        }
    }

    Ok(dep_adj.to_csr())
}

/// Creates a (sparse) adjacency matrix for the contributions.
/// Corresponds to the "cargo-contrib-adj.csv" from the Python scripts.
pub fn new_contribution_adjacency_matrix<N>(
    deps_meta: &DependenciesMetadata,
    contribs_meta: &ContributionsMetadata,
) -> Result<ContributionMatrix<N>, CsvImportError>
where
    N: Num + Clone + From<u32>,
{
    // Creates a sparse matrix of projects x contributors
    let mut contrib_adj: TriMat<N> =
        TriMatBase::new((deps_meta.ids.len(), contribs_meta.contributors.len()));

    for row in contribs_meta.rows.iter() {
        if let Some((row_ix, col_ix)) =
            deps_meta
                .project2index
                .get(&row.project_id)
                .and_then(|row_ix| {
                    contribs_meta
                        .contributor2index
                        .get(&row.contributor)
                        .map(|col| (*row_ix, *col))
                })
        {
            contrib_adj.add_triplet(row_ix, col_ix, N::from(row.contributions))
        }
    }

    Ok(contrib_adj.to_csr())
}

#[cfg(test)]
mod tests {
    extern crate num_traits;
    extern crate tempfile;

    use crate::protocol_traits::ledger::{LedgerView, MockLedger};
    use crate::types::mock::MockNetwork;
    use num_traits::Zero;
    use oscoin_graph_api::{types, GraphDataReader};
    use std::io::{Seek, Write};
    use tempfile::tempfile;

    #[test]
    /// This test setups the same example graph in the basic model PDF
    /// and check the result is what we expect.
    fn csv_network_import_works() {
        let deps_csv = String::from(
            r###"FROM_ID,TO_ID
0,1
2,0
2,1
"###,
        );

        let mut deps_file = tempfile().unwrap();
        deps_file.write_all(deps_csv.as_bytes()).unwrap();
        deps_file.seek(std::io::SeekFrom::Start(0)).unwrap();

        let deps_csv_meta = String::from(
            r###"ID,NAME,PLATFORM
0,foo,Cargo
1,bar,Cargo
2,baz,Cargo
"###,
        );

        let mut meta_file = tempfile().unwrap();
        meta_file.write_all(deps_csv_meta.as_bytes()).unwrap();
        meta_file.seek(std::io::SeekFrom::Start(0)).unwrap();

        let contribs_csv = String::from(
            r###"ID,MAINTAINER,REPO,CONTRIBUTIONS,NAME
0,github@john,https://github.com/foo/foo-rs,100,foo
1,github@tom,https://github.com/bar/bar-rs,30,bar
2,github@tom,https://github.com/baz/baz-rs,60,baz
2,github@alice,https://github.com/baz/baz-rs,20,baz
"###,
        );

        let mut contrib_file = tempfile().unwrap();
        contrib_file.write_all(contribs_csv.as_bytes()).unwrap();
        contrib_file.seek(std::io::SeekFrom::Start(0)).unwrap();

        let mock_ledger = MockLedger::default();

        let network: MockNetwork<f64> = super::import_network(
            csv::ReaderBuilder::new()
                .flexible(true)
                .from_reader(deps_file),
            csv::ReaderBuilder::new()
                .flexible(true)
                .from_reader(meta_file),
            csv::ReaderBuilder::new()
                .flexible(true)
                .from_reader(contrib_file),
            None,
            &mock_ledger.get_hyperparams(),
        )
        .unwrap_or_else(|e| panic!("returned unexpected error: {}", e));

        assert_eq!(
            network.node_data(&String::from("foo")),
            Some(&types::NodeData {
                node_type: types::NodeType::Project {
                    contributions_from_all_users: 100
                },
                rank: Zero::zero()
            }),
        );

        assert_eq!(
            network.edge_data(&0),
            Some(&types::EdgeData {
                edge_type: types::EdgeType::Dependency,
                weight: 0.8,
            }),
        );

        assert_eq!(
            network.edge_data(&7),
            Some(&types::EdgeData {
                edge_type: types::EdgeType::UserToProjectContribution(100),
                weight: 1.0,
            }),
        )
    }
}
