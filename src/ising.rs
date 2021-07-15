
use ndarray::prelude::Array1;
use num_traits::Num;
use num_traits::NumAssignOps;
use num_traits::real::Real;
use sprs::{CsMat, CsMatBase};
use crate::{State, Instance};
use std::ops::{AddAssign, Index, IndexMut};
use sprs::CompressedStorage::CSR;
use rand::prelude::*;
use rand::distributions::Uniform;
use ndarray::AssignElem;

pub type Spin=i8;

pub struct IsingState{
    pub arr: Array1<Spin>
}
impl IsingState{
    /// Fast access and convert to f64 spin by simply checking the sign
    #[inline]
    pub unsafe fn uget_f64(&self, index: usize) -> f64{
        let &si = self.arr.uget(index);
        if si > 0{
            1.0
        } else {
            -1.0
        }
    }
    /// Access and flip the sign of the spin
    #[inline]
    pub unsafe fn uset_neg(&mut self, index: usize){
        *self.arr.uget_mut(index) *= -1;
    }

    pub fn mag(&self) -> i64{
        let mut m : i64 = 0;
        for &s in self.arr.iter(){
            m += s as i64;
        }
        return m;
    }
}

pub fn rand_ising_state<Rn: Rng+?Sized>(n: usize, rng: &mut Rn) -> IsingState{
    let mut arr = Array1::uninit(n);
    for s in arr.iter_mut(){
        s.assign_elem( 2*rng.sample(Uniform::new_inclusive(0, 1)) - 1);
    }
    let arr = unsafe { arr.assume_init() };
    return IsingState{arr};
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

impl State<usize> for IsingState{
    fn accept_move(&mut self, mv: usize) {
        self.arr[mv] *= -1;
    }
}

/// An Ising instance specified by an arbitrary binary quadratic model
/// in sparse matrix form.
/// The energy function is the Hamiltonian
///     $$ H = \sum_i h_i s_i + \sum_{i<j} J_{ij} s_i s_j $$
///
/// where $h_i$ are the biases and $J_{ij}$ are the couplings
pub struct BqmIsingInstance{
    bias: Array1<f64>,
    coupling: CsMat<f64>
}
impl BqmIsingInstance{
    fn new_zero_bias(coupling: CsMat<f64>) -> Self{
        let (n1, n2) = coupling.shape();
        if n1 != n2{
            panic!("couplings matrix must be square, but has shape {}, {}",n1, n2);
        }
        let bias = Array1::zeros(n1);
        return Self{bias, coupling};
    }
}
impl Instance<usize, IsingState> for BqmIsingInstance {
    type Energy = f64;

    fn energy(&self, state: &IsingState) -> f64 {
        let mut total_energy = 0.0;
        for (i, row)in self.coupling.outer_iterator().enumerate(){
            let mut h :f64 = 0.0;
            h += self.bias[i] * (state.arr[i] as f64);
            for (j, &K) in row.iter(){
                h += (K * (state.arr[j] as f64))/2.0
            }
            total_energy += h;
        }
        return total_energy;
    }

    /// The \Delta E of a move proposal to flip spin i is
    ///   H(-s_i) - H(s_i) = -2 h_i s_i - \sum_j J_{ij} s_i s_j
    /// Safe only if the move is within the size of the state
    unsafe fn delta_energy(&self, state: &IsingState, mv: &usize) -> f64 {
        let mut delta_e = 0.0;
        let &i = mv;
        let si = state.uget_f64(i);
        delta_e += -2.0 * si * self.bias[i];
        let row = self.coupling.outer_view(i).unwrap();
        for (j, &K) in row.iter(){
            delta_e  += - K * si * state.uget_f64(j)
        }
        return delta_e;
    }

    fn size(&self) -> usize {
        return self.bias.len();
    }
}



#[cfg(test)]
mod tests {
    use sprs::TriMat;
    use crate::ising::{BqmIsingInstance, rand_ising_state};
    use rand_xoshiro::Xoshiro256PlusPlus;
    use tamc_core::traits::*;
    use tamc_core::metropolis::MetropolisSampler;
    use tamc_core::sa::{simulated_annealing, geometric_beta_schedule};
    use tamc_core::pt::{parallel_tempering_sampler, PTState};
    use rand::prelude::*;

    fn make_ising_2d_instance(l: usize) -> BqmIsingInstance{
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
        let n = l*l;
        let ensemble_size= 16;
        let beta0 :f64 = 0.02;
        let betaf :f64 = 2.0;
        let num_sweeps = 200;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1234);
        let instance = make_ising_2d_instance(l);

        let mut init_states = Vec::with_capacity(ensemble_size);
        for _ in 0..ensemble_size{
            init_states.push(rand_ising_state(n, &mut rng));
        }

        let beta_schedule = geometric_beta_schedule(beta0, betaf, num_sweeps);

        //sampler.advance();
        let states = simulated_annealing(&instance, init_states, &beta_schedule, &mut rng, |_i, _|{} );
        for st in states.iter(){
            let mz = st.mag();
            let e = instance.energy(st);
            println!("mz = {}", mz);
            println!("e = {}", e)
        }

        println!("Done.")
    }

    #[test]
    fn test_ising_2d_pt(){
        let l = 8;
        let n = l*l;
        let beta0 :f64 = 0.02;
        let betaf :f64 = 2.0;
        let num_betas = 16;
        let num_sweeps = 200;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1234);
        let instance = make_ising_2d_instance(l);
        let mut init_states = Vec::with_capacity(num_betas);
        for _ in 0..num_betas{
            init_states.push(rand_ising_state(n, &mut rng));
        }
        let betas = geometric_beta_schedule(beta0, betaf, num_betas);
        let samplers: Vec<_> = betas.iter()
                .map(|&b |MetropolisSampler::new_uniform(&instance,b, n))
                .collect();
        let pt_sampler = parallel_tempering_sampler(samplers);
        let mut init_state = PTState::new(init_states);
        pt_sampler.sweep(&mut init_state,  &mut rng);
    }
}
