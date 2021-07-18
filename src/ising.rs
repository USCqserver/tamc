
use ndarray::prelude::Array1;
use num_traits::Num;
use num_traits::NumAssignOps;
use num_traits::real::Real;
use serde::{Serialize, Deserialize};
use sprs::{CsMat, TriMat};
use crate::{State, Instance};
use std::ops::{AddAssign, Index, IndexMut};
use rand::prelude::*;
use rand::distributions::Uniform;
use ndarray::AssignElem;
use crate::util::read_adjacency_list_from_file;
use petgraph::csr::Csr;
use fixedbitset::FixedBitSet;
use std::fmt::Formatter;
use tamc_core::sa::geometric_beta_schedule;

pub type Spin=i8;

#[derive(Debug, Clone)]
#[derive()]
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
    pub fn new_zero_bias(coupling: CsMat<f64>) -> Self{
        let (n1, n2) = coupling.shape();
        if n1 != n2{
            panic!("couplings matrix must be square, but has shape {}, {}",n1, n2);
        }
        let bias = Array1::zeros(n1);
        return Self{bias, coupling};
    }
    pub fn from_instance_file(file: &str) -> Self{
        let adj_list = read_adjacency_list_from_file(file)
            .expect("Unable to read adjancency from instance file");
        let n = adj_list.len();
        let mut tri_mat = TriMat::new((n, n));
        let mut bias = Array1::zeros(n);
        for i in 0..n{
            let neighborhood = &adj_list[i];
            for (&j, &K) in neighborhood.iter(){
                if i != j{
                    tri_mat.add_triplet(i, j, K);
                } else {
                    bias[i] = K;
                }
            }
        }
        let coupling = tri_mat.to_csr();
        return Self{bias, coupling };
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


fn houdayer_cluster_move<R: Rng+?Sized>(replica1: &mut IsingState, replica2: &mut IsingState,
                                        graph: &Csr<(), ()>, rng: &mut R) -> Option<FixedBitSet>{
    use rand::seq::SliceRandom;
    use petgraph::visit::Bfs;
    use petgraph::visit::NodeFiltered;
    //let n = instance.size();
    let n = graph.node_count();
    if n  > u32::MAX as usize{
        panic!("houdayer_cluster_move: instance size must fit in u32")
    }
    let overlap: Array1<i8> = &replica1.arr * &replica2.arr;
    // Select a random spin with q=-1
    // Rather than random sampling until we find q=-1, we manually apply the shuffling algorithm
    // and terminate once the spin being swapped has q=-1
    let mut idxs : Vec<usize> = (0..n).collect();
    let mut init_spin = None;
    idxs.shuffle(rng);
    for i in (1..n).rev(){
        let j = rng.gen_range(0..(i+1) as u32) as usize;
        let k = idxs[j];
        if overlap[k] < 0 {
            init_spin = Some(k);
            break;
        }
        idxs.swap(i, j);
    }

    let init_spin = match init_spin{
        None => return None, // No spin has q=-1
        Some(k) => k as u32
    };

    let filtered_graph = NodeFiltered::from_fn(graph, |n|overlap[n as usize] < 0);
    let mut bfs = Bfs::new(&filtered_graph, init_spin);

    while let Some(x) = bfs.next(&filtered_graph){
        ()
    }
    let nodes = bfs.discovered;
    let cluster_size = nodes.count_ones(..);
    //println!("cluster size = {}", cluster_size);
    // Finally, swap all
    for i in nodes.ones(){
        unsafe { std::mem::swap(&mut replica1.arr.uget_mut(i), &mut replica2.arr.uget_mut(i)); }
    }
    return Some(nodes);
}

#[derive(Debug, Clone)]
pub struct PtError{
    msg: String
}

impl PtError{
    pub fn new(msg: &str) -> Self{
        Self{msg: msg.to_string()}
    }
}
impl std::fmt::Display for PtError{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.msg)
    }
}
impl std::error::Error for PtError { }

#[derive(Clone, Serialize, Deserialize)]
pub struct PtIcmMinResults{
    pub gs_states: Vec<Vec<Spin>>,
    pub gs_energies: Vec<f64>,
    pub gs_time_steps: Vec<u32>,
    pub num_measurements: u32,
    pub acceptance_counts: Vec<u32>
}

impl PtIcmMinResults{
    fn new(num_betas: u32) -> Self{
        let acceptance_counts = Array1::zeros(num_betas as usize).into_raw_vec();
        return Self{
            gs_states: Vec::new(),
            gs_energies: Vec::new(),
            gs_time_steps: Vec::new(),
            num_measurements: 0,
            acceptance_counts
        };
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct BetaSpec{
    pub beta_min: f64,
    pub beta_max: f64,
    pub num_beta: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum BetaOptions{
    Geometric(BetaSpec),
    Arr(Vec<f64>)
}

impl BetaOptions{
    pub fn new_geometric(beta_min: f64, beta_max: f64, num_beta: u32) -> Self{
        return BetaOptions::Geometric(BetaSpec{beta_min, beta_max, num_beta});
    }
    pub fn get_beta_arr(&self) -> Vec<f64>{
        return match &self {
            BetaOptions::Geometric(b) => {
                geometric_beta_schedule(b.beta_min, b.beta_max, b.num_beta as usize)
            }
            BetaOptions::Arr(v) => {
                v.to_owned()
            }
        };
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PtIcmParams {
    pub num_sweeps: u32,
    pub warmup_fraction: f64,
    pub beta: BetaOptions,
    pub num_replica_chains: u32
}

impl Default for PtIcmParams{
    fn default() -> Self {
        Self{
            num_sweeps: 256,
            warmup_fraction: 0.5,
            beta: BetaOptions::Geometric(BetaSpec{beta_min:0.1, beta_max:10.0, num_beta: 8}),
            num_replica_chains: 2
        }
    }
}

impl PtIcmParams {
    // pub fn check_options(&self) -> Result<(), PtError>{
    //
    //     return Ok(())
    // }
}
pub fn pt_icm_minimize(instance: &BqmIsingInstance,
                       params: &PtIcmParams)
                       -> PtIcmMinResults
{
    use rand_xoshiro::Xoshiro256PlusPlus;
    use tamc_core::traits::*;
    use tamc_core::metropolis::MetropolisSampler;
    use tamc_core::pt::*;
    use tamc_core::ensembles::EnsembleSampler;

    let n = instance.size();
    let num_sweeps = params.num_sweeps;
    let beta_vec = params.beta.get_beta_arr();
    let num_betas = beta_vec.len();
    // seed and create random number generator
    let mut rngt = thread_rng();
    let mut seed_seq = [0u8; 32];
    rngt.fill_bytes(&mut seed_seq);
    let mut rng = Xoshiro256PlusPlus::from_seed(seed_seq);
    // Construct csr graph
    let edges: Vec<_> = instance.coupling.iter().map(|(_, (i,j))| (i as u32,j as u32)).collect();
    let csr_graph: Csr<(), ()> = Csr::from_sorted_edges(&edges).unwrap();

    // randomly generate initial states
    let mut init_states1 = Vec::with_capacity(num_betas);
    let mut init_states2 = Vec::with_capacity(num_betas);
    for _ in 0..num_betas {
        init_states1.push(rand_ising_state(n, &mut rng));
        init_states2.push(rand_ising_state(n, &mut rng));
    }
    let init_state1 = PTState::new(init_states1);
    let init_state2 = PTState::new(init_states2);
    let mut pt_state = vec![init_state1, init_state2];
    // Initialize samplers
    let samplers: Vec<_> = beta_vec.iter()
        .map(|&b | MetropolisSampler::new_uniform(instance,b, n))
        .collect();

    let pt_sampler = parallel_tempering_sampler(samplers);
    let mut pt_results = PtIcmMinResults::new(num_betas as u32);
    let mut pt_chains_sampler = EnsembleSampler::new(pt_sampler);
    //let mut minimum_state = None;
    let measurement_start = (params.warmup_fraction * (num_sweeps as f64)) as u32;
    let mut minimum_e = None;
    for i in 0..num_sweeps{
        // Apply ICM move
        for pt_pairs in pt_state.chunks_exact_mut(2){
            let (pt0, pt1) = pt_pairs.split_at_mut(1);
            let icm_cluster = houdayer_cluster_move(
                &mut pt0[0].states_mut()[0],
                &mut pt1[0].states_mut()[0],
                &csr_graph, &mut rng);
        }

        // Sweep the PT chains
        pt_chains_sampler.sweep(&mut pt_state, &mut rng);
        // Measure statistics/lowest energy state so far
        if i >= measurement_start {
            let mut min_energies = Vec::with_capacity(pt_state.len());
            for pts in pt_state.iter() {
                let (i1, e1) = pts.states_ref().iter()
                    .map(|st| instance.energy(st)).enumerate()
                    .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap()).unwrap();
                min_energies.push((i1, e1))
            }

            let (min_e_ch, &(min_idx, min_e)) = min_energies.iter().enumerate()
                .min_by(|&x, &y| x.1.1.partial_cmp(&y.1.1).unwrap())
                .unwrap();
            let chain = pt_state[min_e_ch].states_ref();
            let min_state = &chain[min_idx];

            if minimum_e.map_or(true, |x| min_e < x) {
                minimum_e = Some(min_e);
                pt_results.gs_states.push(min_state.arr.to_vec());
                pt_results.gs_energies.push(min_e);
                pt_results.gs_time_steps.push(i)
            }
        }

    }
    let mut acceptance_counts = Array1::zeros(num_betas);
    for st in pt_state.iter(){
        let acc = Array1::from(st.num_acceptances_ref().to_owned());
        acceptance_counts += &acc;
    }
    pt_results.acceptance_counts = acceptance_counts.into_raw_vec();
    return pt_results;
    //return ( minimum_state.unwrap().into_raw_vec(), minimum_e.unwrap())
}

#[cfg(test)]
mod tests {
    use ndarray::prelude::Array1;
    use sprs::TriMat;
    use crate::ising::{BqmIsingInstance, rand_ising_state, pt_icm_minimize, PtIcmParams, BetaOptions};
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
        let l = 16;
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

        let mut pt_icm_params = PtIcmParams::default();
        pt_icm_params.num_sweeps = num_sweeps;
        pt_icm_params.beta = BetaOptions::new_geometric(0.1, 10.0, num_betas as u32);
        let opts_str = serde_json::to_string_pretty(&pt_icm_params).unwrap();
        println!("{}", opts_str);
        let beta_arr = pt_icm_params.beta.get_beta_arr();
        let pt_results = pt_icm_minimize(&instance,  &pt_icm_params);
        for (&e, &t) in pt_results.gs_energies.iter().zip(pt_results.gs_time_steps.iter()){
            println!("t={}, e = {}", t, e)
        }
        let acc_counts = Array1::from_vec(pt_results.acceptance_counts);
        let acc_prob = acc_counts.map(|&x|(x as f64)/((2*num_sweeps) as f64));
        for (&b, &p) in beta_arr.iter().zip(acc_prob.iter()){
            println!("beta {} : acc_p = {}", b, p)
        }
    }
}
