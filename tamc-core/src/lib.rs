pub mod traits;
pub mod metropolis;
pub mod ensembles;
pub mod pt;
pub mod sa;

#[cfg(feature = "rayon")]
pub mod parallel;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
