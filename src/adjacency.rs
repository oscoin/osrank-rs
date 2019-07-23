#![allow(unknown_lints)]
#![warn(clippy::all)]

extern crate num_traits;
extern crate serde;
extern crate sprs;

use crate::linalg::{hadamard_mul, normalise_rows, transpose_storage, SparseMatrix};
use crate::types::{HyperParams, Weight};
use num_traits::{Num, Signed};
use sprs::binop::scalar_mul_mat;
use sprs::{hstack, vstack, CsMat};

/// Builds a new (normalised) network graph adjacency matrix.
pub fn new_network_matrix<N>(
    dep_matrix: &SparseMatrix<N>,
    contrib_matrix: &SparseMatrix<N>,
    maintainer_matrix: &SparseMatrix<N>,
    hyperparams: HyperParams,
) -> SparseMatrix<N>
where
    N: Num + Copy + Default + From<Weight> + PartialOrd + Signed,
{
    let contrib_t = transpose_storage(&contrib_matrix);
    let contrib_t_norm = normalise_rows(&contrib_t);

    let maintainer_t = maintainer_matrix.clone().transpose_into();
    let maintainer_norm = normalise_rows(&maintainer_matrix);

    let project_to_project = scalar_mul_mat(
        &normalise_rows(&dep_matrix),
        hyperparams.depend_factor.into(),
    );
    let project_to_account = &scalar_mul_mat(&maintainer_norm, hyperparams.maintain_factor.into())
        + &scalar_mul_mat(
            &normalise_rows(&contrib_matrix),
            hyperparams.contrib_factor.into(),
        );
    let account_to_project =
        &hadamard_mul(
            scalar_mul_mat(&maintainer_t, hyperparams.maintain_prime_factor.into()),
            &contrib_t_norm,
        ) + &scalar_mul_mat(&contrib_t_norm, hyperparams.contrib_prime_factor.into());

    let account_to_account: SparseMatrix<N> =
        CsMat::zero((contrib_matrix.cols(), contrib_matrix.cols()));

    // Join the matrixes together
    let q1_q2 = hstack(&vec![project_to_project.view(), project_to_account.view()]);
    let q3_q4 = hstack(&vec![account_to_project.view(), account_to_account.view()]);

    normalise_rows(&vstack(&vec![q1_q2.view(), q3_q4.view()]))
}
