use crate::traits::*;
use rayon::prelude::*;

/// Samples an ensemble of states using the same sampler S
/// Advancing this sampler performs a sweep of the sub-sampler
pub struct ThreadedEnsembleSampler<S>{
    pub sub_sampler: S,
}
impl<S> ThreadedEnsembleSampler<S>{
    pub fn new(sub_sampler: S) -> Self{
        return Self{sub_sampler};
    }
}


impl<Rn: Send, S >
Sampler<Vec<Rn>> for ThreadedEnsembleSampler<S>
where   S: Sampler<Rn> + Sync,
        S::SampleType: Send
{
    type SampleType=Vec<S::SampleType>;

    fn advance(&self, state: &mut Vec<S::SampleType>, rng_vec: &mut Vec<Rn>) {
        state.par_iter_mut().zip_eq(rng_vec.par_iter_mut())
            .for_each(|(xi, rng)| self.sub_sampler.sweep(xi, rng));
    }

    fn sweep(&self, state: &mut Vec<S::SampleType>, rng_vec: &mut Vec<Rn>){
        self.advance(state, rng_vec);
    }
}