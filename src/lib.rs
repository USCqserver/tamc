//!
//! Trait descriptions:
//!    An *instance* is a specification of a computational problem, including an
//!    energy (i.e. cost function), which may be parametric,
//!    to optimize or sample defined over a sample space.
//!    It must be immutable and thread-safe. It has associated types
//!
//!    A *state* is a mutable point in the sample space of an instance
//!
//!    An *entropy source* is a generalization of an Rng
//!
//!    A *sampler* advances a state using immutable data, an instance and parameters,
//!    along with an entropy source
//!
//!
use num_traits::{Num, Zero};
use num_traits::real::Real;
use rand::Rng;
use rand::distributions::{Distribution, Standard};
use rand::distributions::uniform::{SampleUniform, Uniform};
use std::marker::PhantomData;
pub use tamc_core::metropolis;
pub use tamc_core::traits::*;
pub mod ising;



pub trait GenericSampler<I>{

}
struct ConcreteSampler<I, S>{
    sub_sampler: S,
    instance: I

}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
