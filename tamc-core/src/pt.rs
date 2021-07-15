use crate::traits::*;
use rand::Rng;
use rand::distributions::{Standard, Distribution};
use num_traits::real::Real;
use num_traits::Num;

pub struct PTState<St>{
    states: Vec<St>,
    num_acceptances: Vec<usize>
}

impl<St> PTState<St>{
    pub fn new(states: Vec<St>) -> Self{
        let mut num_acceptances = Vec::with_capacity(states.len());
        num_acceptances.resize( states.len(), 0);
        return Self{states, num_acceptances}
    }
    pub fn into_data(self) -> (Vec<St>, Vec<usize>){
        return (self.states, self.num_acceptances)
    }
}
/// Implementes the Parallel Tempering algorithm on a vector of macrocanonical samplers
pub struct ParallelTemperingSampler<S, R>{
    tempering_chain: Vec<S>,
    delta_beta: Vec<R>
}

pub fn parallel_tempering_sampler<R, S>(sampler_chain: Vec<S> )
        -> ParallelTemperingSampler<S, R>
where
        S: Macrostate<R>, R: Real
{
    let betas : Vec<R> = sampler_chain.iter().map(|s|s.beta()).collect();
    let delta_beta : Vec<R> = betas.iter().skip(1)
            .zip(betas.iter())
            .map(|(&b1, &b2)| b1 - b2).collect();
    return ParallelTemperingSampler{tempering_chain: sampler_chain, delta_beta};
}


impl<R, Rn: Rng+?Sized, S: MacroSampler<R, Rn>>
Sampler<Rn>
for ParallelTemperingSampler<S, R>
where R: Real, Standard: Distribution<R>
{
    type SampleType = PTState<S::SampleType>;
    //type ParamType = S::ParamType;

    fn advance(&self, state: &mut PTState<S::SampleType>,  rng: &mut Rn) {
        let n = state.states.len();
        // Apply replica exchange moves
        let energies: Vec<R> = self.tempering_chain.iter()
            .zip(state.states.iter())
            .map(|(s,x)| s.energy( x))
            .collect();
        let delta_es: Vec<R> = energies.iter().skip(1).zip(energies.iter())
            .map(|(&e1, &e2)| e1 - e2)
            .collect();
        for j in 0..n-1{
            let dlt: R = self.delta_beta[j]*delta_es[j];
            if dlt >= R::zero() || (rng.sample::<R, _>(Standard) < R::exp(dlt)){
                state.states.swap(j, j+1);
                state.num_acceptances[j] += 1;
            }
        }
        // Sweep samples
        for (sampler, xi) in self.tempering_chain.iter()
                .zip(state.states.iter_mut()){
            sampler.sweep(xi,  rng);
        }
    }

    fn sweep(&self, state: &mut PTState<S::SampleType>, rng: &mut Rn){
        self.advance(state, rng);
    }
}
