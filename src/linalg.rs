#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate ndarray;
extern crate sprs;

use ndarray::Array2;
use num_traits::{Num, Signed, Zero};
use sprs::prod::*;
use sprs::{CsMat, CsMatI, CsMatViewI, SpIndex, TriMatI};

pub type SparseMatrix<N> = CsMat<N>;
pub type DenseMatrix<N> = Array2<N>;

/// Inefficient implementation of matrix transposition that preserve the sparse
/// storage layout (i.e. a CSR matrix stays as such instead of being converted
/// into a CSC).
pub fn transpose_storage_naive<N>(matrix: &SparseMatrix<N>) -> SparseMatrix<N>
where
    N: Num + Copy + Signed + PartialOrd + Default,
{
    CsMat::csr_from_dense(matrix.to_dense().t().view(), N::zero())
}

/// Transpose a CSR matrix. Panics if a CSC matrix is given.
pub fn transpose_storage_csr<N>(matrix: &SparseMatrix<N>) -> SparseMatrix<N>
where
    N: Num + Copy + Signed + PartialOrd + Default,
{
    csr_transpose_impl(&matrix.view())
}

pub fn csr_transpose_impl<N, I>(m: &CsMatViewI<N, I>) -> CsMatI<N, I>
where
    N: Num + Copy + Zero,
    I: SpIndex,
{
    let res_rows = m.cols(); // flipped, as we are transposing.
    let res_cols = m.rows();

    let mut res = TriMatI::new((res_rows, res_cols));

    for (&val, (r, c)) in m.iter() {
        res.add_triplet(c.index(), r.index(), val)
    }
    res.to_csr()
}

/// Inefficient implementation of the Hadamard multiplication that (internally)
/// operates on the dense representation.
pub fn hadamard_mul_naive<N>(lhs: SparseMatrix<N>, rhs: &SparseMatrix<N>) -> SparseMatrix<N>
where
    N: Zero + PartialOrd + Signed + Clone,
{
    CsMat::csr_from_dense((lhs.to_dense() * rhs.to_dense()).view(), Zero::zero())
}

pub fn hadamard_mul<N>(lhs: &SparseMatrix<N>, rhs: &SparseMatrix<N>) -> SparseMatrix<N>
where
    N: Zero + PartialOrd + Signed + Copy,
{
    let mut ws = workspace_csr(lhs, rhs);
    csr_hadamard_mul_csr_impl(lhs.view(), rhs.view(), &mut ws)
}

/// Actual implementation of CSR-CSR Hadamard product.
pub fn csr_hadamard_mul_csr_impl<N, I>(
    lhs: CsMatViewI<N, I>,
    rhs: CsMatViewI<N, I>,
    workspace: &mut [N],
) -> CsMatI<N, I>
where
    N: Num + Copy + Zero,
    I: SpIndex,
{
    let res_rows = lhs.rows();
    let res_cols = rhs.cols();
    if lhs.cols() != rhs.cols() {
        panic!("Dimension mismatch (cols)");
    }
    if lhs.rows() != rhs.rows() {
        panic!("Dimension mismatch (rows)");
    }
    if res_cols != workspace.len() {
        panic!("Bad storage dimension");
    }
    if lhs.storage() != rhs.storage() {
        panic!(
            "Storage mismatch: lhs is {:#?}, rhs is {:#?}",
            lhs.storage(),
            rhs.storage()
        );
    }
    if !rhs.is_csr() {
        panic!("rhs is not a CSR matrix.");
    }

    let mut res = CsMatI::empty(lhs.storage(), res_cols);
    res.reserve_nnz_exact(lhs.nnz() + rhs.nnz());

    for (lrow_ix, lvec) in lhs.outer_iterator().enumerate() {
        // reset the accumulators
        for wval in workspace.iter_mut() {
            *wval = N::zero();
        }

        // accumulate the resulting row
        for (lcol_ix, &lval) in lvec.iter() {
            // we can't be out of bounds thanks to the checks of dimension
            // compatibility and the structure check of CsMat. Therefore it
            // should be safe to call into an unsafe version of outer_view

            let rvec = rhs.outer_view(lrow_ix).unwrap();
            let rval = match rvec.get(lcol_ix) {
                None => Zero::zero(),
                Some(v) => *v,
            };

            let wval = &mut workspace[lcol_ix];
            let prod = lval * rval;
            *wval = prod;
        }

        // compress the row into the resulting matrix
        res = res.append_outer(&workspace);
    }
    // TODO: shrink res storage? would need methods on CsMat
    assert_eq!(res_rows, res.rows());
    res
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
pub fn normalise_rows_mut<N>(matrix: &mut SparseMatrix<N>)
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
