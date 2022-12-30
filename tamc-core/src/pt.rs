use crate::traits::*;
use ndarray::prelude::*;
use num_traits::real::Real;
use num_traits::Num;
use rand::Rng;
use rand::distributions::{Standard, Distribution};
use serde::{Serialize, Deserialize};

#[derive(Copy, Clone, Serialize, Deserialize)]
pub enum PTRoundTrip{
    None, // untagged
    MinBeta, // origin at minimum beta
    MaxBeta, // origin at maximum beta
    MinBetaReflected, // reflected from minimum beta, origin from maximum beta
    MaxBetaReflected, // reflected from maximum beta, origin from minimum beta
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PTState<St>{
    pub states: Vec<St>,
    pub num_acceptances: Array1<u32>,
    pub round_trips: u32,
    pub round_trip_tags: Vec<PTRoundTrip>,
    pub diffusion_hist: Array2<u32>
}

impl<St> PTState<St>{
    pub fn new(states: Vec<St>) -> Self{
        let n = states.len();
        let num_acceptances = Array1::zeros(n);
        let diffusion_hist = Array2::zeros((n, 2));

        let mut round_trip_tags = Vec::with_capacity(n);
        round_trip_tags.resize(n, PTRoundTrip::None);
        round_trip_tags[0] = PTRoundTrip::MinBeta;
        *round_trip_tags.last_mut().unwrap() = PTRoundTrip::MaxBeta;
        return Self{states, num_acceptances, round_trips: 0, round_trip_tags, diffusion_hist}
    }
    pub fn reset_tags(&mut self){
        let n = self.states.len();
        self.num_acceptances = Array1::zeros(n);
        self.diffusion_hist = Array2::zeros((n, 2));
        self.round_trips = 0;
        self.round_trip_tags.clear();
        self.round_trip_tags.resize(n, PTRoundTrip::None);
        self.round_trip_tags[0] = PTRoundTrip::MinBeta;
        *self.round_trip_tags.last_mut().unwrap() = PTRoundTrip::MaxBeta;
    }
    pub(crate) fn swap_states(&mut self, j: usize){
        self.states.swap(j, j+1);
        self.round_trip_tags.swap(j, j+1);
        self.num_acceptances[j] += 1;
    }
    pub(crate) fn update_round_trips(&mut self){
        let lowest_tag = &mut self.round_trip_tags[0];
        match *lowest_tag{
            PTRoundTrip::None => {*lowest_tag = PTRoundTrip::MinBeta},
            PTRoundTrip::MinBeta => {},
            PTRoundTrip::MinBetaReflected => {},
            PTRoundTrip::MaxBetaReflected => {self.round_trips += 1; *lowest_tag=PTRoundTrip::MinBeta},
            PTRoundTrip::MaxBeta => {*lowest_tag=PTRoundTrip::MinBetaReflected}
        };
        let highest_tag = self.round_trip_tags.last_mut().unwrap();
        match *highest_tag{
            PTRoundTrip::None => {*highest_tag = PTRoundTrip::MaxBeta},
            PTRoundTrip::MaxBeta => {},
            PTRoundTrip::MaxBetaReflected => {},
            PTRoundTrip::MinBetaReflected => {self.round_trips += 1; *highest_tag=PTRoundTrip::MaxBeta},
            PTRoundTrip::MinBeta => {*highest_tag=PTRoundTrip::MaxBetaReflected}
        };
        // Increment diffusion histogram
        for (mut h, &t) in self.diffusion_hist.axis_iter_mut(Axis(0))
            .zip(self.round_trip_tags.iter()){
            match t{
                PTRoundTrip::MinBeta | PTRoundTrip::MinBetaReflected  => {
                    h[0] += 1;
                }
                PTRoundTrip::MaxBeta | PTRoundTrip::MaxBetaReflected => {
                    h[1] += 1;
                }
                _ => {},
            };
        }
    }

    pub fn states_ref(&self) -> &[St]{
        return &self.states;
    }
    pub fn states_mut(&mut self)-> &mut [St]{
        return &mut self.states;
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
        if self.tempering_chain.len() != n{
            panic!("ParallelTemperingSampler: Expected a chain of {} states but got {}", n, self.tempering_chain.len());
        }
        // Apply replica exchange moves
        let energies: Vec<R> = self.tempering_chain.iter()
            .zip(state.states.iter_mut())
            .map(|(s,x)| s.energy( x))
            .collect();
        // E_{i+1} - E_i array
        let mut delta_es: Vec<R> = energies.iter().skip(1).zip(energies.iter())
            .map(|(&e1, &e2)| e1 - e2)
            .collect();
        for j in 0..n-1{
            let dlt: R = self.delta_beta[j]*delta_es[j];
            if dlt >= R::zero() || (rng.sample::<R, _>(Standard) < R::exp(dlt)){
                state.swap_states(j);
                if j < n-2 {
                    delta_es[j + 1] = delta_es[j + 1] + delta_es[j];
                }
                    delta_es[j] = -delta_es[j];
            }
        }
        state.update_round_trips();
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
