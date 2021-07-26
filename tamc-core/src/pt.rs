use crate::traits::*;
use rand::Rng;
use rand::distributions::{Standard, Distribution};
use num_traits::real::Real;
use num_traits::Num;

#[derive(Copy, Clone)]
pub enum PTRoundTrip{
    None,
    MinBeta,
    MaxBeta
}

#[derive(Clone)]
pub struct PTState<St>{
    pub states: Vec<St>,
    pub num_acceptances: Vec<u32>,
    pub round_trips: u32,
    pub round_trip_tags: Vec<PTRoundTrip>,
    pub diffusion_hist: Vec<(u32, u32)>
}

impl<St> PTState<St>{
    pub fn new(states: Vec<St>) -> Self{
        let n = states.len();
        let mut num_acceptances = Vec::with_capacity(n);
        let mut round_trip_tags = Vec::with_capacity(n);
        let mut diffusion_hist = Vec::with_capacity(n);
        num_acceptances.resize( n, 0);
        round_trip_tags.resize(n, PTRoundTrip::None);
        diffusion_hist.resize(n, (0, 0));
        round_trip_tags[0] = PTRoundTrip::MinBeta;
        *round_trip_tags.last_mut().unwrap() = PTRoundTrip::MaxBeta;
        return Self{states, num_acceptances, round_trips: 0, round_trip_tags, diffusion_hist}
    }
    pub fn reset_tags(&mut self){
        let n = self.states.len();
        self.num_acceptances.clear();
        self.round_trip_tags.clear();
        self.diffusion_hist.clear();
        self.num_acceptances.resize( n, 0);
        self.round_trip_tags.resize(n, PTRoundTrip::None);
        self.diffusion_hist.resize(n, (0, 0));
        self.round_trip_tags[0] = PTRoundTrip::MinBeta;
        *self.round_trip_tags.last_mut().unwrap() = PTRoundTrip::MaxBeta;
    }

    pub fn states_ref(&self) -> &[St]{
        return &self.states;
    }
    pub fn states_mut(&mut self)-> &mut [St]{
        return &mut self.states;
    }
    pub fn num_acceptances_ref(&self) -> &[u32]{
        return &self.num_acceptances;
    }
    pub fn num_acceptances_mut(&mut self) -> &mut [u32]{
        return &mut self.num_acceptances;
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
                state.round_trip_tags.swap(j, j+1);
                state.num_acceptances[j] += 1;
            }
        }
        // Check for a round trip
        if let PTRoundTrip::MaxBeta = state.round_trip_tags[0]{
            state.round_trips += 1;
        }
        // Apply round-trip tags
        state.round_trip_tags[0] = PTRoundTrip::MinBeta;
        *state.round_trip_tags.last_mut().unwrap() = PTRoundTrip::MaxBeta;
        // Increment round-trip histogram
        for (h, &t) in state.diffusion_hist.iter_mut().zip(state.round_trip_tags.iter()){
            match t{
                PTRoundTrip::None => {},
                PTRoundTrip::MinBeta => {
                    h.0 += 1;
                }
                PTRoundTrip::MaxBeta => {
                    h.1 += 1;
                }
            };
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
