use crate::traits::*;
use rand::Rng;
use rand::distributions::{Standard, Distribution};
use num_traits::real::Real;

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

pub fn parallel_tempering_sampler<I: Instance, Rn: ?Sized, S: MacroSampler<I, Rn>>(sampler_chain: Vec<S>)
        -> ParallelTemperingSampler<S, I::Energy>
{
    let betas : Vec<I::Energy> = sampler_chain.iter().map(|s|s.beta()).collect();
    let delta_beta : Vec<I::Energy> = betas.iter().skip(1)
            .zip(betas.iter())
            .map(|(&b1, &b2)| b1 - b2).collect();
    return ParallelTemperingSampler{tempering_chain: sampler_chain, delta_beta};
}


impl<I: Instance, R, Rn: Rng+?Sized, S: MacroSampler<I, Rn>> Sampler<I, Rn>
for ParallelTemperingSampler<S, R>
where R: Real, I:Instance<Energy=R>, Standard: Distribution<R>
{
    type SampleType = PTState<S::SampleType>;
    //type ParamType = S::ParamType;

    fn advance(&self, state: &mut PTState<S::SampleType>, instance: &I, rng: &mut Rn) {
        let n = state.states.len();
        // Apply replica exchange moves
        let energies: Vec<R> = self.tempering_chain.iter()
            .zip(state.states.iter())
            .map(|(s,x)| s.energy(instance, x))
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
            sampler.sweep(xi, instance,  rng);
        }
    }

    fn sweep(&self, state: &mut PTState<S::SampleType>, instance: &I,  rng: &mut Rn){
        self.advance(state, instance, rng);
    }
}
