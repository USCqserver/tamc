use crate::traits::*;
use rand::Rng;
use rand::distributions::{Standard, Distribution};
use ndarray::prelude::*;
use num_traits::real::Real;
use num_traits::Num;

pub use crate::pt::{PTState, PTRoundTrip};

use rayon::prelude::*;

/// Implementes the Parallel Tempering algorithm on a vector of macrocanonical samplers
pub struct ThreadedPTSampler<S, R>{
    tempering_chain: Vec<S>,
    delta_beta: Vec<R>
}

pub fn parallel_tempering_sampler<R, S>(sampler_chain: Vec<S> )
                                        -> ThreadedPTSampler<S, R>
    where
        S: Macrostate<R>, R: Real
{
    let betas : Vec<R> = sampler_chain.iter().map(|s|s.beta()).collect();
    let delta_beta : Vec<R> = betas.iter().skip(1)
        .zip(betas.iter())
        .map(|(&b1, &b2)| b1 - b2).collect();
    return ThreadedPTSampler{tempering_chain: sampler_chain, delta_beta};
}


impl<R, Rn: Rng+Send, S>
Sampler<Vec<Rn>>
for ThreadedPTSampler<S, R>
    where   R: Real, Standard: Distribution<R>,
            S: MacroSampler<R, Rn> + Sync,
            S::SampleType: Send
{
    type SampleType = PTState<S::SampleType>;
    //type ParamType = S::ParamType;

    fn advance(&self, state: &mut PTState<S::SampleType>,  rng_vec: &mut Vec<Rn>) {
        let n = state.states.len();
        if self.tempering_chain.len() != n{
            panic!("ParallelTemperingSampler: Expected a chain of {} states but got {}", n, self.tempering_chain.len());
        }
        // Apply replica exchange moves
        let energies: Vec<R> = self.tempering_chain.iter()
            .zip(state.states_mut().iter())
            .map(|(s,x)| s.energy( x))
            .collect();
        let delta_es: Vec<R> = energies.iter().skip(1).zip(energies.iter())
            .map(|(&e1, &e2)| e1 - e2)
            .collect();
        let rng = &mut rng_vec[0];
        for j in 0..n-1{
            let dlt: R = self.delta_beta[j]*delta_es[j];
            if dlt >= R::zero() || (rng.sample::<R, _>(Standard) < R::exp(dlt)){
                state.swap_states(j);
            }
        }
        state.update_round_trips();
        // Sweep samples
        self.tempering_chain.par_iter()
                .zip(state.states_mut().par_iter_mut())
                .zip_eq(rng_vec.par_iter_mut())
                .for_each(
            |((sampler, xi), rng)| sampler.sweep(xi,  rng)
        )
    }

    fn sweep(&self, state: &mut PTState<S::SampleType>, rng: &mut Vec<Rn>){
        self.advance(state, rng);
    }
}
