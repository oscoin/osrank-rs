#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate fraction;
extern crate num_traits;
extern crate petgraph;

use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug, Default)]
pub struct RandomWalks<Id>
where
    Id: Hash + Eq,
{
    /// A HashMap between the source of the walk and the walk itself.
    random_walks: HashMap<Id, RandomWalk<Id>>,
}

impl<Id> RandomWalks<Id>
where
    Id: Clone + Eq + Hash,
{
    pub fn new() -> Self {
        RandomWalks {
            random_walks: HashMap::new(),
        }
    }

    pub fn add_walk(&mut self, walk: RandomWalk<Id>) {
        self.random_walks
            .insert(walk.random_walk_source.clone(), walk);
    }

    pub fn len(&self) -> usize {
        self.random_walks.len()
    }

    pub fn count_visits(&self, idx: Id) -> usize {
        self.random_walks
            .iter()
            .map(|(_, rw)| rw.count_visits(&idx))
            .sum()
    }
}

type NumVisits = usize;

#[derive(Debug)]
pub struct RandomWalk<Id>
where
    Id: Hash + Eq,
{
    random_walk_source: Id,
    random_walk_visits: HashMap<Id, NumVisits>,
}

impl<Id> RandomWalk<Id>
where
    Id: Clone + Eq + Hash,
{
    pub fn new(source: Id) -> Self {
        let mut m = HashMap::new();
        m.insert(source.clone(), 1);
        RandomWalk {
            random_walk_source: source,
            random_walk_visits: m,
        }
    }

    pub fn add_next(&mut self, idx: Id) {
        if let Some(visits) = self.random_walk_visits.get_mut(&idx) {
            *visits += 1;
        } else {
            self.random_walk_visits.insert(idx, 1);
        }
    }

    pub fn count_visits(&self, idx: &Id) -> NumVisits {
        *self.random_walk_visits.get(idx).unwrap_or(&0)
    }
}

// Just an alias for now.
pub type SeedSet = ();
