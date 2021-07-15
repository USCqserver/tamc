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


impl<I: Instance,Rn: ?Sized, S: Sampler<I, Rn> > Sampler<I, Rn> for EnsembleSampler<S>
{
    type SampleType=Vec<S::SampleType>;
    //type ParamType=S::ParamType;
    fn advance(&self, state: &mut Vec<S::SampleType>, instance: &I, rng: &mut Rn) {
        for xi in state.iter_mut(){
            self.sub_sampler.sweep(xi, instance,  rng);
        }
    }

    fn sweep(&self, state: &mut Vec<S::SampleType>, instance: &I, rng: &mut Rn){
        self.advance(state,instance, rng);
    }
}