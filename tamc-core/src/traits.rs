use num_traits::real::Real;

/// A Markov Chain state is capable of independently proposing moves with an RNG and accepting moves by
/// mutating itself
pub trait State{
    /// While a move may be small and copyable, it can also be large or an entirely new state in itself
    /// so it should be referred to by ref until it is passed to accept_move
    type Move;
    //fn propose_move<R: Rng+?Sized>(&self, rng: &mut R ) -> Self::Move;
    fn accept_move(&mut self, mv: Self::Move);
}

/// An instance specifies the computational problem
pub trait Instance{
    type St: State;
    //type Param: Copy;
    type Energy: Real;
    /// Evaluate the energy
    fn energy(&self, state: &Self::St) -> Self::Energy;
    unsafe fn delta_energy(&self, state: &Self::St, mv: &<Self::St as State>::Move) -> Self::Energy;
    fn size(&self) -> usize;
}

/// A Markov Chain specifies move proposals and acceptances for a state over an instance
/// Using an entropy type Rn
pub trait MarkovChain<I: Instance, Rn: ?Sized>{
    fn propose_move(&self, instance: &I, state: &I::St, rng: &mut Rn) -> <I::St as State>::Move;
    fn accept_move(&self, state: &mut I::St, mv: <I::St as State>::Move){
        state.accept_move(mv);
    }
}

/// General sampler type
pub trait Sampler<I: Instance, Rn: ?Sized>{
    type SampleType;
    //type ParamType;
    fn advance(&self, state: &mut Self::SampleType, instance: &I, rng: &mut Rn );
    /// A sweep performs a number of advances that scales linearly with the system size.
    fn sweep(&self, state: &mut Self::SampleType, instance: &I, rng: &mut Rn ){
        let n = instance.size();
        for _ in 0..n{
            self.advance(state, instance,  rng);
        }
    }
    fn sweep_n(&self, n: u64, state: &mut Self::SampleType, instance: &I, rng: &mut Rn ){
        for _ in 0..n{
            self.sweep(state, instance,  rng);
        }
    }
}

/// A macrocanonical sampler always has a well-defined temperature (i.e. inverse temperature beta),
/// in the units of energy of the instance, as well as a thermodynamic energy at that temperature.
/// This may be an average energy or simply I.energy(state)
pub trait MacroSampler<I: Instance, Rn: ?Sized>: Sampler<I, Rn>{
    fn beta(&self) -> I::Energy;
    fn energy(&self, instance: &I, st: &Self::SampleType) -> I::Energy;
}
