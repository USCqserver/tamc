use num_traits::real::Real;

/// A Markov Chain state is capable of independently proposing moves with an RNG and accepting moves by
/// mutating itself
pub trait State<Mv>{
    /// While a move may be small and copyable, it can also be large or an entirely new state in itself
    /// so it should be referred to by ref until it is passed to accept_move
    //type Move;
    //fn propose_move<R: Rng+?Sized>(&self, rng: &mut R ) -> Self::Move;
    fn accept_move(&mut self, mv: Mv);
}

/// An instance specifies the computational problem
pub trait Instance<Mv, St: State<Mv>>{
    //type St: State;
    //type Param: Copy;
    type Energy: Real;
    /// Evaluate the energy
    fn energy(&self, state: &mut St) -> Self::Energy;
    unsafe fn delta_energy(&self, state: &mut St, mv: &Mv) -> Self::Energy;
    fn size(&self) -> usize;
}

/// General sampler type
pub trait Sampler<Rn: ?Sized>{
    type SampleType;
    //type ParamType;
    fn advance(&self, state: &mut Self::SampleType, rng: &mut Rn );
    /// A sweep performs a number of advances that scales linearly with the system size.
    fn sweep(&self, state: &mut Self::SampleType, rng: &mut Rn );

    fn sweep_n(&self, n: u64, state: &mut Self::SampleType,  rng: &mut Rn ){
        for _ in 0..n{
            self.sweep(state,   rng);
        }
    }
}

/// A macrocanonical sampler always has a well-defined temperature (i.e. inverse temperature beta),
/// in the units of energy of the instance, as well as a thermodynamic energy at that temperature.
/// This may be an average energy or simply I.energy(state)
pub trait Macrostate<R>{
    type Microstate;
    fn beta(&self) -> R;
    fn energy(&self, st: &mut Self::Microstate) -> R;
}

pub trait MacroSampler<R,  Rn: ?Sized>: Sampler<Rn> + Macrostate<R, Microstate=<Self as Sampler<Rn>>::SampleType>{
}

impl<R,  Rn: ?Sized, T> MacroSampler<R, Rn> for T
    where T: Sampler<Rn> + Macrostate<R, Microstate=<T as Sampler<Rn>>::SampleType>
{}