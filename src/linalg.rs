extern crate ndarray;
extern crate sprs;

use ndarray::Array2;
use num_traits::{Num, Signed, Zero};
use sprs::CsMat;

pub type SparseMatrix<N> = CsMat<N>;
pub type DenseMatrix<N> = Array2<N>;

pub fn transpose_storage<N>(matrix: &SparseMatrix<N>) -> SparseMatrix<N>
where
    N: Num + Copy + Signed + PartialOrd + Default,
{
    CsMat::csr_from_dense(matrix.to_dense().t().view(), N::zero())
}

/// Inefficient implementation of the Hadamard multiplication that operates
/// on the dense representation.
pub fn hadamard_mul<N>(lhs: SparseMatrix<N>, rhs: &SparseMatrix<N>) -> SparseMatrix<N>
where
    N: Zero + PartialOrd + Signed + Clone,
{
    CsMat::csr_from_dense((lhs.to_dense() * rhs.to_dense()).view(), Zero::zero())
}

/// Normalises the rows of a Matrix.
/// N.B. It returns a brand new Matrix, therefore it performs a copy.
/// FIXME(adn) Is there a way to yield only a (partial) view copying only
/// the values?
pub fn normalise_rows<N>(matrix: &SparseMatrix<N>) -> SparseMatrix<N>
where
    N: Num + Copy,
{
    let mut cloned = matrix.clone();
    normalise_rows_mut(&mut cloned);
    cloned
}

/// Normalises the rows for the input matrix.
pub fn normalise_rows_mut<N>(matrix: &mut CsMat<N>)
where
    N: Num + Copy,
{
    for mut row_vec in matrix.outer_iterator_mut() {
        let mut ixs = Vec::new();
        let norm = row_vec.iter().fold(N::zero(), |acc, v| {
            ixs.push(v.0);
            acc + *(v.1)
        });
        if norm != N::zero() {
            for ix in ixs {
                row_vec[ix] = row_vec[ix] / norm;
            }
        }
    }
}
