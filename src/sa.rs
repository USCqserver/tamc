use std::ffi::OsStr;
use std::fs::File;
use std::path::Path;
use ndarray::prelude::*;
use serde::{Serialize, Deserialize};
use petgraph::csr::Csr;
use crate::pt::BetaOptions;
use crate::ising::{Spin, BqmIsingInstance, IsingState, rand_ising_state};
use rand::distributions::Uniform;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use tamc_core::traits::Instance;
use tamc_core::sa;
use std::{iter, time};
use itertools::{Itertools, min};
use log::{debug, info};
use tamc_core::metropolis::MetropolisSampler;
use crate::Prog;

#[derive(Clone, Serialize, Deserialize)]
pub struct SaParams {
    pub beta: BetaOptions,
    pub num_replicas: u32,
    pub threads: u32
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AnnealMinResults {
    pub params: SaParams,
    pub timing: f64,
    pub min_energy: f32,
    pub energies: Vec<f32>,
}

impl AnnealMinResults {
    fn new(params: SaParams) -> Self{
        return Self{
            params,
            min_energy: 0.0,
            energies: Vec::new(),
            timing: 0.0
        };
    }

    fn apply_measurements(&mut self, instance: &BqmIsingInstance, sa_state: &mut Vec<IsingState>)
    {
        let energies: Vec<f32> = sa_state.iter().map(|st| instance.energy_ref(st)).collect();
        let &min_energy = energies.iter().min_by(
            |&x,&y| x.partial_cmp(y).unwrap()).unwrap();

        self.min_energy = min_energy;
        self.energies = energies;

    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AnnealState {
    pub instance_size: u32,
    pub num_replicas: u32,
    pub bytes: Vec<u8>
}

impl AnnealState{
    pub fn new(sa_state: &[IsingState]) -> Self{
        let num_replicas = sa_state.len() as u32;
        let instance_size = sa_state.first().unwrap().arr.len() as u32;
        let bpr = sa_state.first().unwrap().num_bytes();
        let total_bytes = (num_replicas as usize) * bpr;
        let mut bytes = (&[0u8]).repeat(total_bytes);
        for (i, s) in sa_state.iter().enumerate(){
            s.write_to_bytes(&mut bytes[i*bpr..(i+1)*bpr]).unwrap();
        }

        Self{
            instance_size, num_replicas, bytes
        }
    }
}
struct SaRunner<'a>{
    params: &'a SaParams,
    instance: &'a BqmIsingInstance,
    beta_vec: Vec<f32>
}
impl<'a> SaRunner<'a>{
    pub fn new(instance: &'a BqmIsingInstance, params: &'a SaParams) -> Self
    {
        let beta_vec = params.beta.get_beta_arr();
        let num_betas = beta_vec.len();
        let beta_arr = Array1::from_vec(beta_vec.clone());
        let beta_diff : Array1<f32> = beta_arr.slice(s![1..]).to_owned() - beta_arr.slice(s![..-1]);
        if !beta_diff.iter().all(|&x|x>=0.0) {
            panic!("beta array must be non-decreasing")
        }
        debug!("Temperature (beta) array:\n\t {:5.4} ", beta_arr);

        return Self{params, instance, beta_vec};
    }

    pub fn run(&self, initial_state: Option<Vec<IsingState>>) -> (AnnealMinResults, Vec<IsingState>){
        // seed and create random number generator
        let mut rngt = thread_rng();
        let mut seed_seq = [0u8; 32];
        rngt.fill_bytes(&mut seed_seq);
        let mut rng = Xoshiro256PlusPlus::from_seed(seed_seq);
        // randomly generate initial states
        let mut sa_state = match initial_state{
            None => self.generate_init_state(&mut rng),
            Some(st) => { st }
        };
        let mut sa_results = self.sa_loop(&mut sa_state, &mut rng);
        //pt_results.final_state = pt_state;
        return (sa_results, sa_state);
    }

    fn sa_loop<Rn: Rng>(
        &self, sa_state: &mut Vec<IsingState>,
        rng: &mut Rn
    ) -> AnnealMinResults
    {
        // Initialize samplers
        let beta0 = self.beta_vec[0];
        let n = self.instance.size() as u32;
        let sampler = MetropolisSampler::new_uniform(self.instance,beta0, n);

        info!("-- SA begin");
        let start = time::Instant::now();
        sa::simulated_annealing(
            sampler, sa_state, &self.beta_vec, rng,
            |_, _| { }
        );
        let end = start.elapsed();
        info!("-- SA Finished");
        let t_sec = end.as_secs_f64();
        let t_us = end.as_micros() as f64;
        info!(r"
Duration: {:5.4} s
Duration per replica: {:5.4e} s
",
            t_sec, t_sec / (self.params.num_replicas as f64));

        let mut sa_results = AnnealMinResults::new(self.params.clone());
        sa_results.apply_measurements(&self.instance, sa_state);

        sa_results.timing = end.as_micros() as f64;

        return sa_results;
    }

    fn generate_init_state<Rn: Rng+?Sized>(&self, rng: &mut Rn) -> Vec<IsingState>{
        // randomly generate initial states
        let n = self.instance.size();
        let num_replicas = self.params.num_replicas;
        let mut sa_states = Vec::with_capacity(num_replicas as usize);
        for _ in 0..num_replicas{
            sa_states.push(rand_ising_state(n as u32, self.instance, rng));
        }
        return sa_states;
    }
}

pub fn run_simulated_annealing(prog: &Prog, params: &SaParams){
    simple_logger::SimpleLogger::new().with_level(log::LevelFilter::Info).env().init().unwrap();
    let mut instance = prog.read_instance();
    let sa_runner = SaRunner::new(&instance, &params);
    println!(" ** Simulated Annealing **");
    let (min_results, final_states) = sa_runner.run(None);

    let sample_output = prog.sample_output.clone().unwrap_or("samples.bin".to_string());
    let ngs = min_results.energies.iter()
        .fold(0u32,
              |n, &e| {if e <=min_results.min_energy{n+1} else {n} });
    let pgs = (ngs as f64) / (min_results.energies.len() as f64);
    let tts = (min_results.timing/1e6/(params.num_replicas as f64))
        * (f64::log10(0.01)/f64::log10(1.0 - pgs));
    info!(r"
** Ground state energy **
  e = {}
** GS Probability  **
  p_gs = {:5.4e}
** Time-to-solution **
  tts = {} s
", min_results.min_energy, pgs, tts);

    {
        let mut f = File::create(&prog.output_file)
            .expect("Failed to create output file");
        let ext = Path::new(&prog.output_file).extension().and_then(OsStr::to_str);
        if ext == Some("pkl"){
            serde_pickle::to_writer(&mut f, &min_results,
                                    serde_pickle::SerOptions::default())
                .expect("Failed to write to pkl file.");
        } else if ext == Some("yml") {
            serde_yaml::to_writer(f, &min_results)
                .expect("Failed to write to yaml file.")
        }
    }
    {
        let gs_energy_states = final_states.iter()
            .zip(min_results.energies.iter())
            .filter(|(s, &e)| e <= min_results.min_energy+1.0e-8 )
            .map(|(s, _)| s.clone()).collect_vec();
        let gs_compressed_states = AnnealState::new(&gs_energy_states);
        let mut f = File::create(&sample_output)
            .expect("Failed to create sample output file");
        let ext = Path::new(&sample_output).extension().and_then(OsStr::to_str);
        if ext == Some("pkl"){
            serde_pickle::to_writer(&mut f, &gs_compressed_states, serde_pickle::SerOptions::default())
                .expect("Failed to write to pkl file");
        } else {
            bincode::serialize_into(&mut f, &gs_compressed_states).expect("Failed to serialize");
        }
    }
}