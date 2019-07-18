#![allow(unknown_lints)]
#![warn(clippy)]
use ndarray::{Array2, ArrayBase, ArrayView1, ArrayViewMut1, ViewRepr};
use sprs::CsMat;

use core::iter::IntoIterator;
use std::collections::hash_map::Values;
use std::collections::HashMap;

/// A wrapper that allows labeling data structures (in particular, array and
/// matrixes).
pub struct Labeled<T> {
    labels: Labels,
    internal: T,
}

type DimLabels<'a> = &'a [String];

#[derive(Debug)]
struct Labels {
    row_labels: HashMap<u32, String>,
    col_labels: HashMap<u32, String>,
}

pub trait WithLabels<'a>
where
    Self: Sized,
{
    fn labeled(self, labels: (DimLabels<'a>, DimLabels<'a>)) -> Labeled<Self>;
}

trait MatrixLike {
    fn get_rows(&self) -> usize;
    fn get_cols(&self) -> usize;
}

impl<N> MatrixLike for CsMat<N> {
    fn get_rows(&self) -> usize {
        self.rows()
    }
    fn get_cols(&self) -> usize {
        self.cols()
    }
}

impl<N> MatrixLike for Array2<N> {
    fn get_rows(&self) -> usize {
        self.dim().0
    }
    fn get_cols(&self) -> usize {
        self.dim().1
    }
}

impl<'a, N> WithLabels<'a> for CsMat<N> {
    fn labeled(self: CsMat<N>, labels: (DimLabels<'a>, DimLabels<'a>)) -> Labeled<CsMat<N>> {
        let mut row_labels = HashMap::default();
        let mut col_labels = HashMap::default();

        if self.is_csr() {
            for ((ix, _vec), label) in self.outer_iterator().enumerate().zip(labels.0.iter()) {
                row_labels.insert(ix as u32, label.clone());
            }
        } else {
            for ((ix, _vec), label) in self.outer_iterator().enumerate().zip(labels.1.iter()) {
                col_labels.insert(ix as u32, label.clone());
            }
        }

        Labeled {
            internal: self,
            labels: Labels {
                row_labels,
                col_labels,
            },
        }
    }
}

impl<'a, N> WithLabels<'a> for Array2<N> {
    fn labeled(self: Array2<N>, labels: (DimLabels, DimLabels)) -> Labeled<Array2<N>> {
        let mut row_labels = HashMap::default();
        let mut col_labels = HashMap::default();

        for (row_ix, label) in (0..self.get_rows()).zip(labels.0.iter()) {
            row_labels.insert(row_ix as u32, label.clone());
        }

        for (col_ix, label) in (0..self.get_cols()).zip(labels.1.iter()) {
            col_labels.insert(col_ix as u32, label.clone());
        }

        Labeled {
            internal: self,
            labels: Labels {
                row_labels,
                col_labels,
            },
        }
    }
}

/// A Rank is a Nx1 labeled matrix which can be sorted.
pub struct Rank<N> {
    internal: Labeled<Array2<N>>,
}

#[derive(Debug)]
pub enum RankError {
    ShapeMismatch(usize, usize),
    SortFailedCannotDeconstructArray,
    SortFailedCannotConstructArray,
}

impl<N> Rank<N> {
    pub fn from(labeled: Labeled<Array2<N>>) -> Result<Self, RankError> {
        let (rows, cols) = (labeled.internal.get_rows(), labeled.internal.get_cols());
        if cols > 1 {
            Err(RankError::ShapeMismatch(rows, cols))
        } else {
            Ok(Rank { internal: labeled })
        }
    }

    fn iter(&self) -> RankIter<N> {
        RankIter {
            current_ix: 0,
            limit: self.internal.internal.get_rows() as u32,
            labeled: self,
        }
    }
}

pub struct RankIter<'a, N> {
    current_ix: u32,
    limit: u32,
    labeled: &'a Rank<N>,
}

impl<'a, N: 'a> Iterator for RankIter<'a, N> {
    type Item = (&'a str, &'a N);
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_ix >= self.limit {
            None
        } else {
            match self
                .labeled
                .internal
                .labels
                .row_labels
                .get(&self.current_ix)
            {
                Some(l) => {
                    let v = Some((
                        (*l).as_str(),
                        &self.labeled.internal.internal[[self.current_ix as usize, 0]],
                    ));
                    self.current_ix += 1;
                    v
                }
                None => None,
            }
        }
    }
}

impl<'a, N: 'a> IntoIterator for &'a Rank<N> {
    type Item = (&'a str, &'a N);
    //type IntoIter = <ArrayView1<'a, N> as IntoIterator>::IntoIter;
    type IntoIter = RankIter<'a, N>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
    use ndarray::arr2;
    use pretty_assertions::assert_eq;
    use sprs::CsMat;

    #[test]
    fn labeled_construction_csmat() {
        let mtrx = CsMat::csr_from_dense(arr2(&[[1, 0, 0], [1, 0, 0], [1, 0, 1]]).view(), 0);

        let rows = vec!["foo", "bar", "baz"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<String>>();
        let no_labels = Vec::default();
        let labels = (rows.as_slice(), no_labels.as_slice());

        let actual = mtrx.clone();
        let lbld = mtrx.labeled(labels);

        assert_eq!(lbld.internal, actual);
    }

    #[test]
    fn labeled_construction_array2() {
        let mtrx = arr2(&[[1, 0, 0], [1, 0, 0], [1, 0, 1]]);

        let rows = vec!["foo", "bar", "baz"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<String>>();
        let cols = vec!["quux", "lorem", "ipsum"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<String>>();
        let labels = (rows.as_slice(), cols.as_slice());

        let actual = mtrx.clone();
        let lbld = mtrx.labeled(labels);

        assert_eq!(lbld.internal, actual);
    }

    #[test]
    fn test_rank_into_iter() {
        let mtrx = arr2(&[[1], [9], [7]]);

        let rows = vec!["foo", "bar", "baz"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<String>>();
        let cols = vec!["quux", "lorem", "ipsum"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<String>>();
        let labels = (rows.as_slice(), cols.as_slice());

        let lbld = mtrx.labeled(labels);

        let rank = Rank::from(lbld).unwrap_or_else(|e| panic!(e));

        assert_eq!(
            rank.into_iter()
                .sorted_by(|(_, v1), (_, v2)| v2.cmp(v1))
                .take(2)
                .collect::<Vec<(&str, &i32)>>(),
            vec![("bar", &9), ("baz", &7)]
        );
    }
}
