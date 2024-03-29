#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate fnv;
extern crate fraction;
extern crate num_traits;
extern crate petgraph;
extern crate rayon;

use fnv::FnvHashMap;

use rayon::prelude::*;
use std::hash::Hash;

#[derive(Debug, Default)]
pub struct RandomWalks<Id>
where
    Id: Hash + Eq + Sync + Send,
{
    /// A collection of random walks.
    random_walks: Vec<RandomWalk<Id>>,
}

impl<Id> RandomWalks<Id>
where
    Id: Clone + Eq + Hash + Sync + Send,
{
    pub fn new() -> Self {
        RandomWalks {
            random_walks: Vec::new(),
        }
    }

    pub fn add_walk(&mut self, walk: RandomWalk<Id>) {
        self.random_walks.push(walk);
    }

    pub fn len(&self) -> usize {
        self.random_walks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.random_walks.len() == 0
    }

    /// Counts the number of visits for the given element *in all walks*.
    ///
    /// ```
    /// use osrank::types::walk::*;
    ///
    /// let ids = vec!["a", "b", "c", "c", "a", "a"];
    /// let mut walks: RandomWalks<String> = RandomWalks::new();
    ///
    /// for i in ids {
    ///     let mut walk = RandomWalk::new(String::from(i));
    ///     walk.add_next(String::from(i));
    ///     walks.add_walk(walk);
    /// }
    ///
    /// assert_eq!(walks.count_visits(&String::from("c")), 4);
    /// ```
    pub fn count_visits(&self, idx: &Id) -> Count {
        self.random_walks
            .par_iter()
            .map(|rw| rw.count_visits(idx))
            .sum()
    }

    /// Counts the number of walks that originates from the input element.
    ///
    /// ```
    /// use osrank::types::walk::*;
    ///
    /// let ids = vec!["a", "b", "c", "c", "a", "a"];
    /// let mut walks: RandomWalks<String> = RandomWalks::new();
    ///
    /// for i in ids {
    ///     let mut walk = RandomWalk::new(String::from(i));
    ///     walk.add_next(String::from(i));
    ///     walks.add_walk(walk);
    /// }
    ///
    /// assert_eq!(walks.count_walks_from(&String::from("c")), 2);
    /// ```
    pub fn count_walks_from(&self, source: &Id) -> Count {
        self.random_walks
            .iter()
            .filter(|rw| rw.random_walk_source == *source)
            .count()
    }

    pub fn append(&mut self, mut rhs: Self) {
        self.random_walks.append(&mut rhs.random_walks)
    }
}

type Count = usize;

#[derive(Debug)]
/// A random walk over a `Graph`. Each walk stores the source of the walk as
/// well as a mapping between a certain node `Id` and the number of visits on
/// that element.
pub struct RandomWalk<Id>
where
    Id: Hash + Eq + Sync + Send,
{
    random_walk_source: Id,
    random_walk_visits: FnvHashMap<Id, Count>,
}

impl<Id> RandomWalk<Id>
where
    Id: Clone + Eq + Hash + Sync + Send,
{
    /// Creates a new `RandomWalk` by passing the source (i.e. beginning)
    /// of the walk. Note that this also counts as a visit, i.e. it's not
    /// necessary to call `add_next` after calling `new`.
    pub fn new(source: Id) -> Self {
        let mut m = FnvHashMap::default();
        m.insert(source.clone(), 1);
        RandomWalk {
            random_walk_source: source,
            random_walk_visits: m,
        }
    }

    /// Adds a segment (typically a graph's node) to the walk.
    pub fn add_next(&mut self, idx: Id) {
        if let Some(visits) = self.random_walk_visits.get_mut(&idx) {
            *visits += 1;
        } else {
            self.random_walk_visits.insert(idx, 1);
        }
    }

    /// Returns the number of visits of the given segment in the walk.
    pub fn count_visits(&self, idx: &Id) -> Count {
        *self.random_walk_visits.get(idx).unwrap_or(&0)
    }

    /// Given the `Id` of a segment within the walk, returns the `Id` of the
    /// source of the walk, if the input `Id` belongs to the walk.
    pub fn source_from(&self, segment_id: &Id) -> Option<&Id> {
        if self.random_walk_visits.contains_key(segment_id) {
            Some(&self.random_walk_source)
        } else {
            None
        }
    }
}

/// A set of trusted nodes, used to perform walks over the graph. This set has
/// two purposes:
///
/// 1. It allows pruning of the input `Graph`, so that nodes falling below
///    a certain threshold are discarded and not used in the actual osrank
///    calculation;
/// 2. Ensure that the entire `Graph` is explored and that random walks
///    eventually "explore" all the nodes.
#[derive(Debug)]
pub struct SeedSet<Id> {
    trusted_nodes: Vec<Id>,
}

impl<I> SeedSet<I> {
    /// Creates a new empty `SeedSet` collection.
    pub fn new() -> Self {
        SeedSet {
            trusted_nodes: Vec::new(),
        }
    }

    /// Creates a new `SeedSet` collection from a vector of identifiers.
    pub fn from(nodes: Vec<I>) -> Self {
        SeedSet {
            trusted_nodes: nodes,
        }
    }

    pub fn len(&self) -> usize {
        self.trusted_nodes.len()
    }

    pub fn add_node(&mut self, node: I) {
        self.trusted_nodes.push(node)
    }

    pub fn is_empty(&self) -> bool {
        self.trusted_nodes.is_empty()
    }

    pub fn seedset_iter(&self) -> SeedSetIter<I> {
        SeedSetIter {
            range: 0..self.trusted_nodes.len(),
            inner: &self.trusted_nodes,
        }
    }
}

pub struct SeedSetIter<'a, I> {
    range: std::ops::Range<usize>,
    inner: &'a Vec<I>,
}

impl<'a, I> Iterator for SeedSetIter<'a, I> {
    type Item = &'a I;
    fn next(&mut self) -> Option<Self::Item> {
        match self.range.next() {
            None => None,
            Some(ix) => Some(&self.inner[ix]),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn seed_set_from() {
        let nodes = vec![1, 2, 3];
        assert_eq!(SeedSet::from(nodes).is_empty(), false);
    }

    #[quickcheck]
    fn seed_set_seedset_iter(nodes: Vec<u32>) {
        assert_eq!(
            SeedSet::from(nodes.clone())
                .seedset_iter()
                .cloned()
                .collect::<Vec<u32>>(),
            nodes
        );
    }

    #[test]
    fn random_walk_count_visits_non_existent() {
        let source = "foo";
        let w = RandomWalk::new(source);
        assert_eq!(w.count_visits(&"bar"), 0);
    }

    #[test]
    fn random_walk_count_visits_existent() {
        let source = "foo";
        let w = RandomWalk::new(source);
        assert_eq!(w.count_visits(&"foo"), 1);
    }

    #[test]
    fn random_walk_add_next() {
        let source = "foo";
        let n1 = "bar";
        let mut w = RandomWalk::new(source);
        w.add_next(n1);
        assert_eq!(w.count_visits(&"bar"), 1);
    }

    #[test]
    fn random_walk_add_next_twice() {
        let source = "foo";
        let n1 = "foo";
        let mut w = RandomWalk::new(source);
        w.add_next(n1);
        assert_eq!(w.count_visits(&"foo"), 2);
    }

    #[test]
    fn random_walk_source_from() {
        let source = "foo";
        let n1 = "bar";
        let mut w = RandomWalk::new(source);
        w.add_next(n1);
        assert_eq!(w.source_from(&"bar"), Some(&"foo"));
        assert_eq!(w.source_from(&"baz"), None);
    }

    #[test]
    fn random_walks_count_walks_from() {
        let ids = vec!["a", "b", "c", "a", "a"];
        let mut walks: RandomWalks<String> = RandomWalks::new();

        for i in ids {
            let walk = RandomWalk::new(String::from(i));
            walks.add_walk(walk);
        }

        assert_eq!(walks.count_walks_from(&String::from("a")), 3);
    }
}
