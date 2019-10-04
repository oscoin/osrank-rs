/// Incremental Osrank algorithm.
pub mod incremental;
/// Naive Osrank algorithm.
pub mod naive;

/// Shared types between algorithms.

#[derive(Debug, PartialEq, Eq)]
/// Errors that the `osrank` algorithm might throw.
pub enum OsrankError {
    /// Generic, catch-all error for things which can go wrong during the
    /// algorithm.
    UnknownError,
    RngFailedToSplit(String),
}

impl From<rand::Error> for OsrankError {
    fn from(err: rand::Error) -> OsrankError {
        OsrankError::RngFailedToSplit(format!("{}", err))
    }
}
