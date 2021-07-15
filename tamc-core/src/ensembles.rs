use crate::traits::*;

/// Samples an ensemble of states using the same sampler S
/// Advancing this sampler performs a sweep of the sub-sampler
pub struct EnsembleSampler<S>{
    pub sub_sampler: S,
}
impl<S> EnsembleSampler<S>{
    pub fn new(sub_sampler: S) -> Self{
        return Self{sub_sampler};
    }
}


impl<Rn: ?Sized, S: Sampler<Rn> >
Sampler<Rn> for EnsembleSampler<S>
{
    type SampleType=Vec<S::SampleType>;
    //type ParamType=S::ParamType;
    fn advance(&self, state: &mut Vec<S::SampleType>, rng: &mut Rn) {
        for xi in state.iter_mut(){
            self.sub_sampler.sweep(xi,  rng);
        }
    }

    fn sweep(&self, state: &mut Vec<S::SampleType>, rng: &mut Rn){
        self.advance(state, rng);
    }
}