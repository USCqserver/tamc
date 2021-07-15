//! Simulated annealing of a Metropolis sampler
use crate::traits::{State, Instance, Sampler};
use crate::ensembles::EnsembleSampler;
use crate::metropolis::MetropolisSampler;
use rand::prelude::*;
use rand::distributions::Standard;
use num_traits::real::Real;


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

/// Reference implemenation of simulated annealing
/// with Metropolis MC
/// using a single thread and Rng
pub fn simulated_annealing<St, I: Instance<usize, St>, Rn: Rng+?Sized, F: FnMut(usize, &Vec<St>)>
(
    instance: &I,
    init_states: Vec<St>,
    beta_schedule : &[I::Energy],
    rng: &mut Rn,
    mut measure: F
) -> Vec<St>
    where St: State<usize>, Standard: Distribution<I::Energy>, I::Energy: Copy+Real
{
    let mut states = init_states;
    let n = instance.size();
    let num_beta = beta_schedule.len();
    if num_beta == 0 {
        return states;
    }
    let init_beta = beta_schedule[0];
    let sampler = MetropolisSampler::new_uniform(instance, init_beta, n);
    let mut ensemble_sampler = EnsembleSampler::new(sampler);
    for i in 0..num_beta{
        ensemble_sampler.sub_sampler.beta = beta_schedule[i];
        ensemble_sampler.sweep(&mut states,  rng);
        measure(i, &states);
    }

    return states;
}
