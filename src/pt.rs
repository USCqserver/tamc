use std::cmp::min;
use std::ffi::OsStr;
use std::fmt::Formatter;
use std::fs::File;
use std::ops::{AddAssign, Index, IndexMut};
use std::path::Path;
use std::time;

use fixedbitset::FixedBitSet;
use itertools::Itertools;
use log::{debug, info, warn};
use ndarray::prelude::*;
use petgraph::csr::Csr;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use tamc_core::ensembles as ens;
use tamc_core::parallel::ensembles as pens;
use tamc_core::parallel::pt as ppt;
use tamc_core::pt as pt;
use tamc_core::pt::PTState;
use tamc_core::sa::geometric_beta_schedule;
use tamc_core::traits::*;

use crate::{Instance, Prog};
use crate::ising::{BqmIsingInstance, IsingSampler, IsingState, rand_ising_state};
use crate::ising_results::MinResults;

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
    let s1 = ArrayView1::from(&replica1.arr);
    let s2 = ArrayView1::from(&replica2.arr);
    let overlap: Array1<i8> = &s1 * &s2;

    // Select a random spin with q=-1
    let mut idxs: Vec<usize> = Vec::new();
    idxs.reserve(n);
    for (i, &qi) in overlap.iter().enumerate(){
        if qi < 0 {
            idxs.push(i);
        }
    }
    let init_spin = idxs.choose(rng);

    let init_spin = match init_spin{
        None => return None, // No spin has q=-1
        Some(&k) => k as u32
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
        unsafe { std::mem::swap(replica1.arr.get_unchecked_mut(i),  replica2.arr.get_unchecked_mut(i)); }
    }
    // Invalidate energy caches
    replica1.energy_init=false;
    replica2.energy_init=false;
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
    pub params: PtIcmParams,
    #[serde(flatten)]
    pub min_results: MinResults,
    pub acceptance_counts: Vec<u32>
}

impl PtIcmMinResults{
    fn new(params: PtIcmParams, num_betas: u32, instance_size: u32) -> Self{
        let min_results = MinResults::new(num_betas, instance_size);
        let acceptance_counts = Array1::zeros(num_betas as usize).into_raw_vec();
        return Self{
            params,
            min_results,
            acceptance_counts
        };
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PtIcmThermalSamples{
    compression_level: u8,
    pub instance_size: u64,
    pub beta_arr: Vec<f32>,
    pub samples: Vec<Vec<Vec<u8>>>,
    pub e: Vec<Vec<f32>>,
    pub q: Vec<Vec<i32>>,
    pub suscept: Vec<Vec<Vec<f32>>>
}

impl PtIcmThermalSamples{
    fn new(beta_arr: &Vec<f32>, instance_size: u64, capacity: usize, samp_capacity: usize,
           nchi: u32,
           compression_level: u8) -> Self{
        let num_betas = beta_arr.len();
        let beta_arr = beta_arr.iter().map(|&x|x as f32).collect();
        let mut me = Self{
            samples: Vec::with_capacity(num_betas),
            beta_arr,
            instance_size,
            e: Vec::with_capacity(num_betas),
            q: Vec::with_capacity(num_betas),
            suscept: Vec::with_capacity(num_betas),
            compression_level
        };
        for _ in 0..num_betas {
            me.e.push(Vec::with_capacity(2*capacity));
            me.q.push(Vec::with_capacity(capacity));
            me.suscept.push(Vec::new());
        }
        for i in 0..num_betas{
            me.suscept[i].resize(nchi as usize, Vec::new());
            for j in 0..nchi{
                me.suscept[i][j as usize].reserve(capacity);
            }
        }
        if compression_level == 0{ // no compression: save all states at all temperatures
            for _ in 0..num_betas {
                me.samples.push(Vec::with_capacity(samp_capacity));
            }
        } else if compression_level == 1{ // save only the lower half of the temperatures
            for _ in 0..(num_betas/2) {
                me.samples.push(Vec::with_capacity(samp_capacity));
            }
        } else { //save only the lowest temperature samples
            me.samples.push(Vec::with_capacity(samp_capacity));
        }
        return me;
    }

    fn measure(&mut self, pt_state: &mut Vec<pt::PTState<IsingState>>, instance:& BqmIsingInstance) {
        let num_chains = pt_state.len();
        let num_betas = pt_state[0].states.len();
        let n = pt_state[0].states[0].arr.len();
        let nchi = instance.suscept_coefs.len();
        let mut overlap_vec : Vec<i8> = Vec::new();
        if nchi > 0 {
            overlap_vec.resize(n, 0);
        }

        for i in 0..num_betas{
            for j in 0..num_chains{
                let isn = &mut pt_state[j].states[i];
                let e = instance.energy(isn);
                self.e[i].push(e as f32);
            }
            for j in 0..(num_chains/2) {
                let isn1 = &pt_state[2*j].states[i];
                let isn2 = &pt_state[2*j+1].states[i];
                if nchi > 0{
                    for (qi,(&s1, &s2)) in overlap_vec.iter_mut().zip_eq(
                        isn1.arr.iter().zip_eq(isn2.arr.iter())){
                        *qi = s1 * s2;
                    }
                    for k in 0..nchi{
                        let chi = instance.suscept(&overlap_vec, k);
                        self.suscept[i][k].push(chi as f32);
                    }
                }

                let q = isn1.overlap(isn2);
                self.q[i].push(q as i32);
            }
        }
    }
    fn sample_states(&mut self, pt_state: & Vec<pt::PTState<IsingState>>) {
        let num_chains = pt_state.len();
        let num_betas = pt_state[0].states.len();
        if self.compression_level == 0 {
            for i in 0..num_betas {
                for j in 0..num_chains {
                    let isn = &pt_state[j].states[i];
                    self.samples[i].push(isn.as_bytes());
                }
            }
        } else if self.compression_level == 1 {
            for i in 0..(num_betas/2){
                let isamp = num_betas - i - 1;
                for j in 0..num_chains {
                    let isn = &pt_state[j].states[isamp];
                    self.samples[i].push(isn.as_bytes());
                }
            }
        } else {
            for j in 0..num_chains {
                let isn = &pt_state[j].states[num_betas - 1];
                self.samples[0].push(isn.as_bytes());
            }
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct BetaSpec{
    pub beta_min: f32,
    pub beta_max: f32,
    pub num_beta: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum BetaOptions{
    Geometric(BetaSpec),
    Arr(Vec<f32>)
}

impl BetaOptions{
    pub fn new_geometric(beta_min: f32, beta_max: f32, num_beta: u32) -> Self{
        return BetaOptions::Geometric(BetaSpec{beta_min, beta_max, num_beta});
    }
    pub fn get_beta_arr(&self) -> Vec<f32>{
        return match &self {
            BetaOptions::Geometric(b) => {
                geometric_beta_schedule(b.beta_min as f64, b.beta_max as f64, b.num_beta as usize)
                    .into_iter().map(|x| x as f32).collect()
            }
            BetaOptions::Arr(v) => {
                ToOwned::to_owned(v)
            }
        };
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PtIcmParams {
    pub num_sweeps: u32,
    pub warmup_fraction: f64,
    pub beta: BetaOptions,
    pub lo_beta: f32,
    pub icm: bool,
    pub num_replica_chains: u32,
    pub threads: u32,
    pub sample: Option<u32>,
    pub sample_states: Option<u32>,
    pub sample_limiting: Option<u8>
}

impl Default for PtIcmParams{
    fn default() -> Self {
        Self{
            num_sweeps: 256,
            warmup_fraction: 0.5,
            beta: BetaOptions::Geometric(BetaSpec{beta_min:0.1, beta_max:10.0, num_beta: 8}),
            lo_beta: 1.0,
            icm: true,
            num_replica_chains: 2,
            threads: 1,
            sample: Some(32),
            sample_states: Some(64),
            sample_limiting: Some(0)
        }
    }
}

impl PtIcmParams {
    // pub fn check_options(&self) -> Result<(), PtError>{
    //
    //     return Ok(())
    // }
}
pub struct PtIcmRunner<'a>{
    params: &'a PtIcmParams,
    instance: &'a BqmIsingInstance,
    g: Csr<(), ()>,
    beta_vec: Vec<f32>,
    meas_init: u32,
    lo_beta_idx: usize
}
impl<'a> PtIcmRunner<'a>{
    pub fn new(instance: &'a BqmIsingInstance, params: &'a PtIcmParams) -> Self
    {
        let beta_vec = params.beta.get_beta_arr();
        let num_betas = beta_vec.len();
        let beta_arr = Array1::from_vec(beta_vec.clone());
        let beta_diff : Array1<f32> = beta_arr.slice(s![1..]).to_owned() - beta_arr.slice(s![..-1]);
        if !beta_diff.iter().all(|&x|x>=0.0) {
            panic!("beta array must be non-decreasing")
        }

        let lo_beta_ref = beta_vec.iter().enumerate().find(|&(_, &b)| b >= params.lo_beta);
        let lo_beta_idx = match lo_beta_ref{
            None => {
                warn!("Note: lo_beta={} is out of bounds. The largest beta value will be assigned.", params.lo_beta);
                num_betas-1
            }
            Some((i, _)) => {
                i
            }
        };

        // Construct csr graph
        let edges: Vec<_> = instance.coupling.iter()
            .map(|(_, (i,j))| (i as u32,j as u32)).collect();
        let g: Csr<(), ()> = Csr::from_sorted_edges(&edges).unwrap();

        let meas_init = (params.warmup_fraction * (params.num_sweeps as f64)) as u32;

        return Self{params, instance, beta_vec, g, meas_init, lo_beta_idx};
    }


    pub fn run_parallel(&self) -> (PtIcmMinResults, PtIcmThermalSamples, Vec<PTState<IsingState>>){
        let m = self.params.num_replica_chains;
        let num_betas = self.beta_vec.len();
        // seed and create random number generator
        let mut rngt = thread_rng();
        let mut seed_seq = [0u8; 32];
        rngt.fill_bytes(&mut seed_seq);
        let mut rng = Xoshiro256PlusPlus::from_seed(seed_seq);
        // randomly generate initial states
        let mut pt_state = self.generate_init_state(&mut rng);
        // generate ensemble rngs
        let mut rng_vec = Vec::with_capacity(num_betas);
        for _ in 0..m{
            let mut rng_chain = Vec::with_capacity(num_betas);
            for _ in 0..num_betas{
                rng_chain.push(rng.clone());
                rng.jump()
            }
            rng_vec.push(rng_chain);
        };

        let (mut pt_results, pt_samps) = self.parallel_pt_loop(&mut pt_state, &mut rng_vec);
        self.count_acc(&pt_state, &mut pt_results);
        return (pt_results, pt_samps, pt_state);
    }

    pub fn run_seeded(&self, initial_state: Option<Vec<PTState<IsingState>>>) -> (PtIcmMinResults, PtIcmThermalSamples, Vec<PTState<IsingState>>){
        // seed and create random number generator
        let mut rngt = thread_rng();
        let mut seed_seq = [0u8; 32];
        rngt.fill_bytes(&mut seed_seq);
        let mut rng = Xoshiro256PlusPlus::from_seed(seed_seq);
        // randomly generate initial states
        let mut pt_state = match initial_state{
            None => self.generate_init_state(&mut rng),
            Some(st) => { st }
        };
        let (mut pt_results, pt_samps) = self.pt_loop(&mut pt_state, &mut rng);
        self.count_acc(&pt_state, &mut pt_results);
        //pt_results.final_state = pt_state;
        return (pt_results, pt_samps, pt_state);
    }

    pub fn run(&self, initial_state: Option<Vec<PTState<IsingState>>>) -> (PtIcmMinResults, PtIcmThermalSamples, Vec<PTState<IsingState>>){
        // seed and create random number generator
        let mut rngt = thread_rng();
        let mut seed_seq = [0u8; 32];
        rngt.fill_bytes(&mut seed_seq);
        let mut rng = Xoshiro256PlusPlus::from_seed(seed_seq);
        // randomly generate initial states
        let mut pt_state = match initial_state{
            None => self.generate_init_state(&mut rng),
            Some(st) => { st }
        };
        let (mut pt_results, pt_samps) = self.pt_loop(&mut pt_state, &mut rng);
        self.count_acc(&pt_state, &mut pt_results);
        //pt_results.final_state = pt_state;
        return (pt_results, pt_samps, pt_state);
    }

    fn parallel_pt_loop<Rn: Rng+Send>(
        &self, pt_state: &mut Vec<pt::PTState<IsingState>>,
        rng_vec: &mut Vec<Vec<Rn>>
    ) -> (PtIcmMinResults, PtIcmThermalSamples)
    {
        // Initialize samplers
        let n = self.instance.size();
        let num_betas = self.beta_vec.len();
        let num_sweeps = self.params.num_sweeps;
        let num_chains = self.params.num_replica_chains;
        let samp_capacity = if let &Some(nsamp) = &self.params.sample{
            num_chains * (num_sweeps - self.meas_init) / nsamp
        } else {
            0
        } as usize;
        let state_samp_capacity = if let &Some(nsamp) = &self.params.sample_states{
            num_chains * (num_sweeps - self.meas_init) / nsamp
        } else {
            0
        } as usize;
        let samplers: Vec<_> = self.beta_vec.iter()
            .map(|&b | IsingSampler::new(self.instance,b, n as u32))
            .collect();
        let pt_sampler = ppt::parallel_tempering_sampler(samplers);
        let mut pt_results = PtIcmMinResults::new(self.params.clone(),num_betas as u32, n as u32);
        let mut pt_samps = PtIcmThermalSamples::new(&self.beta_vec, n as u64,samp_capacity,
                                                    state_samp_capacity, self.instance.suscept_coefs.len() as u32,
                                                    self.params.sample_limiting.unwrap_or(0));
        let mut pt_chains_sampler = pens::ThreadedEnsembleSampler::new(pt_sampler);
        let mut minimum_e = None;
        info!("-- PT-ICM begin");
        let start = time::Instant::now();
        for i in 0..num_sweeps{
            self.apply_icm(pt_state, &mut rng_vec[0][0]);
            pt_chains_sampler.sweep(pt_state, rng_vec);
            self.apply_measurements(i, pt_state, &mut minimum_e, &mut pt_results.min_results, &mut pt_samps);
        }
        let end = start.elapsed();
        info!("-- PT-ICM Finished");
        info!("Duration: {:5.4} s", end.as_secs_f64());
        pt_results.min_results.timing = end.as_micros() as f64;

        return (pt_results, pt_samps);
    }

    pub fn pt_loop<Rn: Rng>(
        &self, pt_state: &mut Vec<pt::PTState<IsingState>>,
        rng: &mut Rn
    ) -> (PtIcmMinResults, PtIcmThermalSamples)
    {
        // Initialize samplers
        let n = self.instance.size();
        let num_betas = self.beta_vec.len();
        let num_sweeps = self.params.num_sweeps;
        let num_chains = self.params.num_replica_chains;
        let samp_capacity = if let &Some(nsamp) = &self.params.sample{
            num_chains * (num_sweeps - self.meas_init) / nsamp
        } else {
            0
        } as usize;
        let state_samp_capacity = if let &Some(nsamp) = &self.params.sample_states{
            num_chains * (num_sweeps - self.meas_init) / nsamp
        } else {
            0
        } as usize;

        let samplers: Vec<_> = self.beta_vec.iter()
            .map(|&b | IsingSampler::new(self.instance,b, n as u32))
            .collect();
        let pt_sampler = pt::parallel_tempering_sampler(samplers);
        let mut pt_results = PtIcmMinResults::new(self.params.clone(),num_betas as u32, n as u32);

        let mut pt_samps = PtIcmThermalSamples::new(&self.beta_vec, n as u64, samp_capacity,
                                                    state_samp_capacity, self.instance.suscept_coefs.len() as u32,
                                                    self.params.sample_limiting.unwrap_or(0));
        let mut pt_chains_sampler = ens::EnsembleSampler::new(pt_sampler);
        let mut minimum_e = None;
        info!("-- PT-ICM begin");
        let start = time::Instant::now();
        for i in 0..num_sweeps{
            self.apply_icm(pt_state, rng);
            pt_chains_sampler.sweep(pt_state, rng);
            self.apply_measurements(i, pt_state, &mut minimum_e, &mut pt_results.min_results, &mut pt_samps);
        }
        let end = start.elapsed();
        info!("-- PT-ICM Finished");
        info!("Duration: {:5.4} s", end.as_secs_f64());
        pt_results.min_results.timing = end.as_micros() as f64;

        return (pt_results, pt_samps);
    }

    pub fn generate_init_state<Rn: Rng+?Sized>(&self, rng: &mut Rn) -> Vec<pt::PTState<IsingState>>{
        // randomly generate initial states
        let n = self.instance.size() as u32;
        let num_betas = self.beta_vec.len();
        let mut pt_state = Vec::new();
        for _ in 0..self.params.num_replica_chains{
            let mut init_states = Vec::with_capacity(num_betas);
            for _ in 0..num_betas{
                init_states.push(rand_ising_state(n, self.instance, rng));
            }
            pt_state.push(pt::PTState::new(init_states));
        }
        return pt_state;
    }

    fn apply_icm<Rn: Rng+?Sized>(&self, pt_state: &mut Vec<pt::PTState<IsingState>>, rng: &mut Rn)
                                 -> Vec<Option<FixedBitSet>>
    {
        if !self.params.icm{
            return Vec::new();
        }
        // Apply ICM move
        let lo_beta_idx = self.lo_beta_idx;
        let mut icm_vec = Vec::new();
        for pt_pairs in pt_state.chunks_exact_mut(2){
            let (pt0, pt1) = pt_pairs.split_at_mut(1);
            for (replica1, replica2) in pt0[0].states_mut()[lo_beta_idx..].iter_mut()
                .zip(pt1[0].states_mut()[lo_beta_idx..].iter_mut()){
                let icm_cluster = houdayer_cluster_move(
                    replica1, replica2, &self.g, rng);
                icm_vec.push(icm_cluster);
            }
        }
        return icm_vec;
    }

    fn apply_measurements(&self, i: u32, pt_state: &mut Vec<pt::PTState<IsingState>>,
                          minimum_e: &mut Option<f32>, pt_results: &mut MinResults,
                          pt_samples: &mut PtIcmThermalSamples)
    {

        if i >= self.meas_init {
            let stp = i-self.meas_init;
            if let Some(samp_steps) = self.params.sample{
                if stp % samp_steps == 0 || i == self.params.num_sweeps-1{
                    pt_samples.measure(pt_state, &self.instance);
                }
            }
            if let Some(state_samp_steps) = self.params.sample_states{
                if stp % state_samp_steps == 0 || i == self.params.num_sweeps-1{
                    pt_samples.sample_states(pt_state);
                }
            }
            // Measure statistics/lowest energy state so far
            let mut min_energies = Vec::with_capacity(pt_state.len());
            for pts in pt_state.iter_mut() {
                let energies : Vec<f32> = pts.states_mut().iter_mut()
                    .map(|st| self.instance.energy(st)).collect();
                let (i1, &e1) = energies.iter().enumerate()
                    .min_by(|&x, &y| x.1.partial_cmp(&y.1).unwrap())
                    .unwrap();
                min_energies.push((i1, e1))
            }

            let (min_e_ch, &(min_idx, min_e)) = min_energies.iter().enumerate()
                .min_by(|&x, &y| x.1.1.partial_cmp(&y.1.1).unwrap())
                .unwrap();
            let chain = pt_state[min_e_ch].states_ref();
            let min_state = &chain[min_idx];

            if minimum_e.map_or(true, |x| min_e < x) {
                *minimum_e = Some(min_e);
                pt_results.gs_states.push(min_state.as_u64_vec());
                pt_results.gs_energies.push(min_e);
                pt_results.gs_time_steps.push(i)
            }
        }
    }

    fn count_acc(&self, pt_state: & Vec<pt::PTState<IsingState>>, pt_results: &mut PtIcmMinResults){
        let mut acceptance_counts = Array1::zeros(self.beta_vec.len());
        for st in pt_state.iter(){
            acceptance_counts += &st.num_acceptances;
        }
        pt_results.acceptance_counts = acceptance_counts.into_raw_vec();
    }

}
pub fn pt_icm_minimize(instance: &BqmIsingInstance,
                       params: &PtIcmParams)
                       -> PtIcmMinResults
{

    println!(" ** Parallel Tempering - ICM **");
    let pticm = PtIcmRunner::new(instance, params);
    return if params.threads > 1 {
        pticm.run_parallel().0
    } else {
        pticm.run(None).0
    }
}


pub fn run_parallel_tempering(prog: &Prog, params: &PtIcmParams){
    simple_logger::SimpleLogger::new().with_level(log::LevelFilter::Info).env().init().unwrap();
    let sample_output = prog.sample_output.clone().unwrap_or("samples.bin".to_string());
    let mut instance = prog.read_instance();
    if prog.suscepts.len() > 0{
        instance = instance.with_suscept(&prog.suscepts);
    }
    //let results = ising::pt_icm_minimize(&instance,&pt_params);
    println!(" ** Parallel Tempering - ICM **");

    info!("Number of sweeps: {}", params.num_sweeps);
    if params.icm {
        info!("Using ICM")
    } else{
        info!("ICM Disabled")
    }
    let pticm = PtIcmRunner::new(&instance, &params);
    let results = if params.threads > 1 {
        pticm.run_parallel()
    } else {
        pticm.run(None)
    };
    let (gs_results, samp_results, _) = results;
    println!("PT-ICM Done.");
    println!("** Ground state energy **");
    println!("  e = {}", gs_results.min_results.gs_energies.last().unwrap());
    {
        let f = File::create(&prog.output_file)
            .expect("Failed to create yaml output file");
        serde_yaml::to_writer(f, &gs_results)
            .expect("Failed to write to yaml file.")
    }
    {
        let mut f = File::create(&sample_output)
            .expect("Failed to create sample output file");
        let ext = Path::new(&sample_output).extension().and_then(OsStr::to_str);
        if ext == Some("pkl"){
            serde_pickle::to_writer(&mut f, &samp_results, serde_pickle::SerOptions::default());
        } else {
            bincode::serialize_into(&mut f, &samp_results).expect("Failed to serialize");
        }
    }
}


#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use ndarray::prelude::Array1;
    use rand::prelude::*;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use sprs::TriMat;

    use tamc_core::metropolis::MetropolisSampler;
    use tamc_core::pt::{parallel_tempering_sampler, PTState};
    use tamc_core::sa::{geometric_beta_schedule, simulated_annealing};
    use tamc_core::traits::*;

    use crate::ising::{BqmIsingInstance, rand_ising_state};
    use crate::pt::{BetaOptions, pt_icm_minimize, PtIcmParams};
    use crate::ising::tests::make_ising_2d_instance;

    #[test]
    fn test_ising_2d_pt(){
        let l = 16;
        let n: u32 = l*l;
        let beta0 :f64 = 0.02;
        let betaf :f64 = 2.0;
        let num_betas = 16;
        let num_sweeps = 200;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1234);
        let instance = make_ising_2d_instance(l as usize);
        let mut init_states = Vec::with_capacity(num_betas);
        for _ in 0..num_betas{
            init_states.push(rand_ising_state(n, &instance, &mut rng));
        }
        let betas = geometric_beta_schedule(beta0, betaf, num_betas)
            .into_iter().map(|x: f64| {x as f32}).collect_vec();
        let samplers: Vec<_> = betas.iter()
            .map(|&b |MetropolisSampler::new_uniform(&instance,b, n as u32))
            .collect();
        let pt_sampler = parallel_tempering_sampler(samplers);
        let mut init_state = PTState::new(init_states);

        let mut pt_icm_params = PtIcmParams::default();
        pt_icm_params.num_sweeps = num_sweeps;
        pt_icm_params.beta = BetaOptions::new_geometric(0.1, 10.0, num_betas as u32);
        let opts_str = serde_yaml::to_string(&pt_icm_params).unwrap();
        println!("{}", opts_str);
        let beta_arr = pt_icm_params.beta.get_beta_arr();
        let pt_results = pt_icm_minimize(&instance,  &pt_icm_params);
        for (&e, &t) in pt_results.min_results.gs_energies.iter()
                .zip(pt_results.min_results.gs_time_steps.iter()){
            println!("t={}, e = {}", t, e)
        }
        let acc_counts = Array1::from_vec(pt_results.acceptance_counts);
        let acc_prob = acc_counts.map(|&x|(x as f64)/((2*num_sweeps) as f64));
        for (&b, &p) in beta_arr.iter().zip(acc_prob.iter()){
            println!("beta {} : acc_p = {}", b, p)
        }
    }
}