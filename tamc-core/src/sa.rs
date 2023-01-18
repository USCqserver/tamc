//! Simulated annealing of a Metropolis sampler
use std::cell::Cell;
use num_traits::{FromPrimitive, Num, PrimInt};
use crate::traits::{State, Instance, Sampler, Macrostate, MacroSampler};
use crate::ensembles::EnsembleSampler;
use crate::metropolis::MetropolisSampler;
use rand::prelude::*;
use rand::distributions::Standard;
use num_traits::real::Real;
use rand::distributions::uniform::SampleUniform;


pub fn geometric_beta_schedule(beta0: f64, betaf: f64, num_sweeps: usize) -> Vec<f64>{
    let mut beta_schedule = Vec::with_capacity(num_sweeps);
    let log_beta0 = beta0.ln();
    let log_betaf = betaf.ln();
    let dlt_log_beta = (log_betaf - log_beta0)/((num_sweeps-1) as f64);
    for i in 0..num_sweeps{
        beta_schedule.push(f64::exp(log_beta0+dlt_log_beta*(i as f64)))
    }
    return beta_schedule;
}

/// Reference implementation of simulated annealing
/// with Metropolis MC
/// using a single thread and Rng
pub fn simulated_annealing<R, N, St, I, D: Distribution<N>, Rn: Rng+?Sized, F: FnMut(usize, &Vec<St>)>(
    sampler : MetropolisSampler<R, N, St, I, D>,
    states: &mut Vec<St>,
    beta_schedule : &[R],
    rng: &mut Rn,
    mut measure: F
)
where
    I: Instance<N, St, Energy=R>,
    St: State<N>,
    Standard: Distribution<R>,
    R: Real,
    N: Num + FromPrimitive
{
    let num_beta = beta_schedule.len();
    let mut ensemble_sampler = EnsembleSampler::new(sampler);
    for i in 0..num_beta{
        ensemble_sampler.sub_sampler.beta = beta_schedule[i];
        ensemble_sampler.sweep(states,  rng);
        measure(i, &states);
    }
}

