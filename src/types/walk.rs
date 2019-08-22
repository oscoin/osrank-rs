#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate fnv;
extern crate fraction;
extern crate num_traits;
extern crate petgraph;

use fnv::FnvHashMap;
use std::hash::Hash;
use std::rc::Rc;

#[derive(Debug, Default)]
pub struct RandomWalks<Id>
where
    Id: Hash + Eq,
{
    /// A `HashMap` between the source of the walk and the walk itself.
    random_walks: Vec<RandomWalk<Id>>,
}

impl<Id> RandomWalks<Id>
where
    Id: Eq + Hash,
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

    pub fn count_visits(&self, idx: Id) -> NumVisits {
        self.random_walks
            .iter()
            .fold(0, |acc, rw| acc + rw.count_visits(&idx))
    }
}

type NumVisits = usize;

#[derive(Debug)]
pub struct RandomWalk<Id>
where
    Id: Hash + Eq,
{
    random_walk_source: Rc<Id>,
    random_walk_visits: FnvHashMap<Rc<Id>, NumVisits>,
}

impl<Id> RandomWalk<Id>
where
    Id: Eq + Hash,
{
    /// Creates a new `RandomWalk` by passing the source (i.e. beginning)
    /// of the walk. Note that this also counts as a visit, i.e. it's not
    /// necessary to call `add_next` after calling `new`.
    pub fn new(source: Id) -> Self {
        let ix = Rc::new(source);
        let mut m = FnvHashMap::default();
        m.insert(Rc::clone(&ix), 1);
        RandomWalk {
            random_walk_source: ix,
            random_walk_visits: m,
        }
    }

    /// Adds a segment (typically a graph's node) to the walk.
    pub fn add_next(&mut self, idx: Id) {
        let ix = Rc::new(idx);
        if let Some(visits) = self.random_walk_visits.get_mut(&ix) {
            *visits += 1;
        } else {
            self.random_walk_visits.insert(ix, 1);
        }
    }

    /// Returns the number of visits of the given segment in the walk.
    pub fn count_visits(&self, idx: &Id) -> NumVisits {
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

// Just an alias for now.
pub type SeedSet = ();

#[cfg(test)]
mod tests {

    use super::*;

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

}
