#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate fraction;
extern crate num_traits;
extern crate petgraph;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct RandomWalks<Id> {
    random_walks_internal: Vec<RandomWalk<Id>>,
}

impl<Id> RandomWalks<Id>
where
    Id: PartialEq,
{
    pub fn new() -> Self {
        RandomWalks {
            random_walks_internal: Vec::new(),
        }
    }

    pub fn add_walk(&mut self, walk: RandomWalk<Id>) {
        self.random_walks_internal.push(walk);
    }

    pub fn len(&self) -> usize {
        self.random_walks_internal.len()
    }

    pub fn count_visits(&self, idx: Id) -> usize {
        self.random_walks_internal
            .iter()
            .map(|rw| rw.count_visits(&idx))
            .sum()
    }
}

#[derive(Debug, Default, PartialEq, Eq, Hash)]
pub struct RandomWalk<Id> {
    random_walk_internal: Vec<Id>,
}

impl<Id> RandomWalk<Id>
where
    Id: PartialEq,
{
    pub fn new() -> Self {
        RandomWalk {
            random_walk_internal: Vec::new(),
        }
    }

    pub fn add_next(&mut self, idx: Id) {
        self.random_walk_internal.push(idx);
    }

    pub fn count_visits(&self, idx: &Id) -> usize {
        self.random_walk_internal
            .iter()
            .filter(|i| i == &idx)
            .count()
    }
}

// Just an alias for now.
pub type SeedSet = ();
