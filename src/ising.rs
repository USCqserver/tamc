use std::cmp::min;
use std::fmt::Formatter;
use std::fs::File;
use std::ops::{AddAssign, Index, IndexMut};
use std::time;

use fixedbitset::FixedBitSet;
use log::{info, debug, warn};
use ndarray::AssignElem;
use ndarray::prelude::*;
use num_traits::{FromPrimitive, Num, ToPrimitive};
use num_traits::NumAssignOps;
use num_traits::real::Real;
use petgraph::csr::Csr;
use rand::distributions::{Standard, Uniform};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sprs::{CsMat, DenseVector, TriMat};
use itertools::Itertools;

use tamc_core::ensembles as ens;
use tamc_core::metropolis::MetropolisSampler;
use tamc_core::parallel::ensembles as pens;
use tamc_core::parallel::pt as ppt;
use tamc_core::pt as pt;
use tamc_core::sa::geometric_beta_schedule;
use tamc_core::traits::*;

use crate::{Instance, State};
use crate::util::{read_adjacency_list_from_file, read_txt_vec};
use tamc_core::pt::PTState;
use crate::pt::BetaOptions::Arr;
use tamc_core::util::monotonic_divisions;

pub type Spin=i8;

#[derive(Debug, Clone, Serialize)]
#[derive()]
pub struct IsingState{
    pub arr: Vec<Spin>,
    #[serde(skip)]
    pub energy: f32,
    pub energy_init: bool
}

impl IsingState{
    /// Fast access and convert to f64 spin by simply checking the sign
    #[inline]
    pub unsafe fn uget_f64(&self, index: u32) -> f64{
        let &si = self.arr.get_unchecked(index as usize);
        if si > 0{
            1.0
        } else {
            -1.0
        }
    }
    /// Fast access and convert to f64 spin by simply checking the sign
    #[inline]
    pub unsafe fn uget(&self, index: u32) -> i8{
        return *self.arr.get_unchecked(index as usize);
    }
    /// Access and flip the sign of the spin
    #[inline]
    pub unsafe fn uset_neg(&mut self, index: u32){
        *self.arr.get_unchecked_mut(index as usize) *= -1;
    }

    pub fn mag(&self) -> i64{
        let mut m : i64 = 0;
        for &s in self.arr.iter(){
            m += s as i64;
        }
        return m;
    }

    pub fn as_binary_string(&self) -> String{
        let mut s = std::string::String::with_capacity(self.arr.len());
        for &si in self.arr.iter(){
            if si > 0 { // match +1 to 0
                s.push('0');
            } else {  // match -1 to 1
                s.push('1');
            }
        }
        return s;
    }

    /// Number of bytes needed to represent this state, using one bit per spin
    pub fn num_bytes(&self) -> usize{
        let n = self.arr.len();
        let num_bytes = n/8 + (if n%8 == 0{ 0 } else { 1 });
        return num_bytes;
    }
    
    pub fn write_to_bytes(&self, bytes: &mut [u8]) -> Result<(),()>{
        if bytes.len() < self.num_bytes(){
            return Err(())
        }
        for (i, &si) in self.arr.iter().enumerate(){
            let bi = i / 8;
            let k = i % 8;
            unsafe {
                let b = bytes.get_unchecked_mut(bi);
                *b |= if si > 0 { 0 } else {1 << k};
            }
        };
        return Ok(())
    }
    
    pub fn as_bytes(&self) -> Vec<u8>{
        let mut bytes_vec: Vec<u8> =(&[0]).repeat(self.num_bytes());
        self.write_to_bytes(bytes_vec.as_mut_slice()).unwrap_or(());
        return bytes_vec;
    }

    pub fn as_u64_vec(&self) -> Vec<u64>{
        let n = self.arr.len();
        let num_bytes = n/64 + (if n%64 == 0{ 0 } else { 1 });
        let mut bytes_vec: Vec<u64> =(&[0]).repeat(num_bytes);
        for (i, &si) in self.arr.iter().enumerate(){
            let bi = i / 64;
            let k = i % 64;
            unsafe {
                let b = bytes_vec.get_unchecked_mut(bi);
                *b |= ( if si > 0 { 0 } else {1 << k});
            }
        };

        return bytes_vec;
    }

    pub fn overlap(&self, other: &IsingState) -> i64 {
        let mut q : i64 = 0;
        for (&si, &sj) in self.arr.iter().zip_eq(other.arr.iter()){
            q +=  (si * sj) as i64;
        }
        return q;
    }
}

pub fn rand_ising_state<I: Instance<u32, IsingState>, Rn: Rng+?Sized>(n: u32, instance: &I, rng: &mut Rn) -> IsingState{
    let mut arr = Vec::new();
    arr.reserve(n as usize);
    for _ in 0..n{
        arr.push( 2*rng.sample(Uniform::new_inclusive(0, 1)) - 1);
    }
    let mut ising_state = IsingState{arr, energy: 0.0, energy_init: false};
    instance.energy(&mut ising_state);
    return ising_state;
}

impl Index<usize> for IsingState{
    type Output = Spin;

    fn index(&self, index: usize) -> &Spin {
        return &self.arr[index];
    }
}

impl IndexMut<usize> for IsingState{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        return &mut self.arr[index]
    }
}

impl State<u32> for IsingState{
    fn accept_move(&mut self, mv: u32) {
        unsafe{
            self.uset_neg(mv);
        }
    }
}

pub struct IsingSampler<'a>{
    pub samp: MetropolisSampler<'a, f32, u32, IsingState, BqmIsingInstance, Uniform<u32>>
}

impl<'a> IsingSampler<'a>{
    pub fn new(instance: &'a BqmIsingInstance, beta: f32, n: u32) -> Self{
        let samp = MetropolisSampler::new_uniform(instance, beta, n);
        return Self{samp};
    }
}

impl<'a, Rn: Rng+?Sized> Sampler<Rn>
for IsingSampler<'a>
    where
{
    type SampleType = IsingState;
    //type ParamType = I::Param;

    fn advance(&self, state: &mut IsingState, rng: &mut Rn) {
        let mv = rng.sample(&self.samp.rand_distr);
        let de = self.samp.advance_impl(mv, state, rng);
        state.energy += de.unwrap_or(0.0);
    }

    fn sweep(&self, state: &mut IsingState, rng: &mut Rn){
        let mut de = 0.0;
        let n = state.arr.len() as u32;
        for i in 0..n{
            let dei = self.samp.advance_impl(i, state, rng);
            de += dei.unwrap_or(0.0);
        }
        state.energy += de;
    }
}


impl<'a> Macrostate<f32>
for IsingSampler<'a>{
    type Microstate = IsingState;

    fn beta(&self) -> f32 {
        return self.samp.beta();
    }

    fn energy(&self, st: &mut IsingState) -> f32 {
        return self.samp.energy(st);
    }
}


/// An Ising instance specified by an arbitrary binary quadratic model
/// in sparse matrix form.
/// The energy function is the Hamiltonian
///     $$ H = \sum_i h_i s_i + \sum_{i<j} J_{ij} s_i s_j $$
///
/// where $h_i$ are the biases and $J_{ij}$ are the couplings
pub struct BqmIsingInstance{
    pub offset: f32,
    pub bias: Vec<f32>,
    pub coupling: CsMat<f32>,
    pub coupling_vecs: Vec<Vec<(u32, f32)>>,
    pub suscept_coefs: Vec<Vec<f64>>
}

impl BqmIsingInstance{
    pub fn new_zero_bias(coupling: CsMat<f32>) -> Self{
        let (n1, n2) = coupling.shape();
        if n1 != n2{
            panic!("couplings matrix must be square, but has shape {}, {}",n1, n2);
        }
        let mut coupling_vecs = Vec::new();
        coupling_vecs.resize(n1, Vec::new());
        for (i, row)in coupling.outer_iterator().enumerate(){
            for (j, &K) in row.iter() {
                if i == j{
                    panic!("Expected a zero-bias Csr instance");
                }
                coupling_vecs[i].push((j.to_u32().unwrap(), K as f32));
            }
        }
        let mut bias = Vec::new();
        bias.resize(n1, 0.0);

        return Self{offset: 0.0, bias, coupling, coupling_vecs, suscept_coefs: Vec::new()};
    }
    pub fn from_instance_file(file: &str, qubo: bool) -> Self{
        let adj_list = read_adjacency_list_from_file(file)
            .expect("Unable to read adjancency from instance file");
        let n = adj_list.len();
        let mut offset = 0.0;
        let mut tri_mat = TriMat::new((n, n));
        let mut coupling_vecs = Vec::with_capacity(n);
        coupling_vecs.resize(n, Vec::new());
        let mut bias = Vec::new();
        bias.resize(n, 0.0);

        for i in 0..n{
            let neighborhood = &adj_list[i];
            coupling_vecs[i].reserve(neighborhood.len());
            for (&j, &K) in neighborhood.iter(){
                if qubo{
                    if i != j {
                        offset += K / 8.0;
                        tri_mat.add_triplet(i, j, K/4.0);
                        coupling_vecs[i].push((j.to_u32().unwrap(), (K/4.0)));
                        bias[i] += K / 4.0;
                    } else {
                        offset += K / 2.0;
                        bias[i] += K / 2.0;
                        coupling_vecs[i].push((j.to_u32().unwrap(), (K/2.0)));
                    }
                } else {
                    if i != j {
                        tri_mat.add_triplet(i, j, K);
                        coupling_vecs[i].push((j.to_u32().unwrap(), K));
                    } else {
                        bias[i] = K;
                    }
                }
            }
        }
        let coupling = tri_mat.to_csr();
        return Self{offset, bias, coupling, coupling_vecs, suscept_coefs: Vec::new() };
    }
    pub fn with_suscept(self, suscept_files: &Vec<String>) -> Self{
        let mut me = self;
        for file in suscept_files.iter(){
            let f = File::open(file).expect("Unable to open susceptibility file");
            let dvec = read_txt_vec(f)
                .expect("Unable to read susceptibility coefficients from file");
            let n1 = dvec.len();
            let n2 = me.bias.len();
            if n1 != n2{
                println!("WARNING: Ignoring suscept file {} - Expected {} coefficients, but got {}",
                         file, n2, n1)
            }
            me.suscept_coefs.push(dvec)
        }
        return me;
    }

    pub fn to_csr_graph(&self) -> Csr<(), ()>{
        // Construct csr graph
        let edges: Vec<_> = self.coupling.iter()
            .map(|(_, (i,j))| (i as u32,j as u32)).collect();
        let g: Csr<(), ()> = Csr::from_sorted_edges(&edges).unwrap();

        return g;
    }

    pub fn suscept(&self, overlap: &[Spin], i: usize) -> f64{
        let mut chi = 0.0;
        for (&w, &si) in self.suscept_coefs[i].iter().zip_eq(overlap.iter()){
            chi += w * (si as f64);
        }
        return chi;
    }
}
impl Instance<u32, IsingState> for BqmIsingInstance {
    type Energy = f32;

    fn energy_ref(&self, state: & IsingState) -> Self::Energy {
        let mut total_energy = self.offset;
        for ( row, i)in self.coupling_vecs.iter().zip(0..){
            unsafe {
                let mut h = 0.0;
                let si = state.uget(i) as Self::Energy;
                h += *self.bias.get_unchecked(i as usize) ;
                for &(j, K) in row.iter() {
                    let sj = state.uget(j) as Self::Energy;
                    h += (K * sj) / 2.0;
                }
                total_energy += h * si;
            }
        }

        return total_energy;
    }
    fn energy(&self, state: &mut IsingState) -> Self::Energy {
        if state.energy_init{
            return state.energy;
        }
        let total_energy = self.energy_ref(state);
        state.energy = total_energy;
        state.energy_init = true;
        return total_energy;
    }

    /// The \Delta E of a move proposal to flip spin i is
    ///   H(-s_i) - H(s_i) = -2 h_i s_i - 2 \sum_j J_{ij} s_i s_j
    /// Safe only if the move is within the size of the state
    unsafe fn delta_energy(&self, state: &mut IsingState, mv: &u32) -> Self::Energy {
        let mut delta_e = 0.0;
        let i = *mv;
        delta_e += *self.bias.get_unchecked(i as usize);
        let row = self.coupling_vecs.get_unchecked(i as usize);
        for &(j, K) in row.iter(){
            delta_e  += K * (state.uget(j) as Self::Energy);
        }
        let si = state.uget(i) as Self::Energy;
        delta_e *= -2.0 * si;

        return delta_e;
    }

    fn size(&self) -> usize {
        return self.bias.len();
    }
}



#[cfg(test)]
pub(crate) mod tests {
    use ndarray::prelude::Array1;
    use rand::prelude::*;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use sprs::TriMat;

    use tamc_core::metropolis::MetropolisSampler;
    use tamc_core::pt::{parallel_tempering_sampler, PTState};
    use tamc_core::sa::{geometric_beta_schedule, simulated_annealing};
    use tamc_core::traits::*;

    use crate::ising::{BqmIsingInstance,  rand_ising_state};

    pub fn make_ising_2d_instance(l: usize) -> BqmIsingInstance{
        let n = l*l;
        let mut tri_mat = TriMat::new((n, n));
        for i in 0..l{
            for j in 0..l{
                let q0 = i*l + j;
                let q1 = ((i+1)%l)*l + j;
                let q2 = i*l + (j+1)%l;
                tri_mat.add_triplet(q0, q1, -1.0);
                tri_mat.add_triplet(q1, q0, -1.0);
                tri_mat.add_triplet(q0, q2, -1.0);
                tri_mat.add_triplet(q2, q0, -1.0);
            }
        }

        let instance = BqmIsingInstance::new_zero_bias(tri_mat.to_csr());
        return instance;
    }
    #[test]
    fn test_ising_2d_sa() {
        let l = 8;
        let n: u32 = l*l;
        let ensemble_size= 16;
        let beta0 :f64 = 0.02;
        let betaf :f64 = 2.0;
        let num_sweeps = 200;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1234);
        let instance = make_ising_2d_instance(l as usize);

        let mut init_states = Vec::with_capacity(ensemble_size);
        for _ in 0..ensemble_size{
            init_states.push(rand_ising_state(n, &instance, &mut rng));
        }

        let beta_schedule = geometric_beta_schedule(beta0, betaf, num_sweeps);

        //sampler.advance();
        let mut states = simulated_annealing(&instance, init_states, &beta_schedule, &mut rng, |_i, _|{} );
        for st in states.iter_mut(){
            let mz = st.mag();
            let e = instance.energy(st);
            println!("mz = {}", mz);
            println!("e = {}", e)
        }

        println!("Done.")
    }

}
