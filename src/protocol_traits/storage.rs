#![allow(unknown_lints)]
#![warn(clippy::all)]
pub trait KeyValueStorage {
    type Key;
    type Value;
    fn get(&self, key: &Self::Key) -> Option<&Self::Value>;
}
