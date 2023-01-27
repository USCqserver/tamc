use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::error::Error;
use std::ffi::OsStr;
use std::iter::FromIterator;
use std::ops::Index;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use itertools::Itertools;
use log::info;
use petgraph::csr::Csr;
use petgraph::data::DataMap;
use serde::{Serialize, Deserialize};
use tamc_core::traits::Instance;
use petgraph::prelude::*;
use petgraph::visit::{IntoEdgeReferences, IntoNodeReferences};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use sprs::DenseVector;
use crate::ising::{BqmIsingInstance, IsingState};
use crate::Prog;
use crate::pt::{PtIcmParams, PtIcmRunner};
use crate::util::{read_u32_lines, write_data};

#[derive(Clone, Serialize, Deserialize)]
pub struct GlaParams{
    pub num_threads: u32,
    pub pt_params: PtIcmParams,
    pub partition_file: String,
}

impl GlaParams{
    /// Read in the graph partition specification.
    /// Each line contains the indices of spins that belong to one partition.
    /// All spins in the instance must be grouped to exactly one partition.
    pub fn read_graph_partition(&self, instance: &BqmIsingInstance)
            -> Result<(Vec<Vec<u32>>, HashMap<u32, u32>)>{
        // read in the partition data
        let n = instance.size();
        let mut in_part : Vec<bool> = [false].repeat(n );
        let mut partition_mapping = HashMap::new();
        let file = File::open(&self.partition_file)?;
        let partition_vecs = read_u32_lines(file)?;
        for (ip, p) in partition_vecs.iter().enumerate(){
            for i in p.iter().map(|&x| x as usize){
                if i >= n {
                    return Err(anyhow!("Partitition must be specified for instance size {}. Found index {}", n, i))
                }
                if in_part[i] {
                    return Err(anyhow!("Index {} specified more than once in partition {}", i, ip))
                }
                in_part[i] = true;
                partition_mapping.insert(i as u32, ip as u32);
            }
        }
        return Ok((partition_vecs, partition_mapping));
    }
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct GlaResults{
    pub instance_size: u32,
    pub num_partitions: u32,
    pub partition_energies: Vec<f32>,
    pub boundary_energies: Vec<(u32, u32, f32)>,
    pub final_state: Vec<u8>,
    pub total_energy: f32
}

/// Returns:
///  (1) the graph induced by contracting each partition to one node
///        Node data: vector of spins in the instance graph belonging to the partition
///        Edge data: vector of edges (i, j, K) in the instance graph connecting two partitions
///  (2) augmented instance graph
///        Node data: (p, h) where p is the partition of the node and h is the bias
///        Edge data: K, the coupling strength between two connected spins
fn instance_partition(partition_vecs: Vec<Vec<u32>>,
                      partition_mapping: &HashMap<u32, u32>,
                      instance: &BqmIsingInstance)
        -> (Graph<Vec<u32>, Vec<(u32, u32, f32)>>, UnGraph<(u32, f32), f32>){
    // initialize the partition graph
    // this graph should logically be undirected, but is generated as a directed graph
    // to calculate boundary energies more easily
    let mut partition_graph: Graph<Vec<u32>, Vec<(u32, u32, f32)>> = Graph::new();
    for mut p in partition_vecs.into_iter(){
        p.sort();
        partition_graph.add_node(p);
    }
    let graph = instance.to_energy_graph();
    // augment the instance graph with partition data
    let aug_graph = graph.map(
        |i, &h| (partition_mapping[&(i.index() as u32)], h),
        |e, &K| K
    );
    // generate the partition graph edges
    for e in aug_graph.edge_references(){
        let nsrc = e.source();
        let ntgt = e.target();
        let &K = e.weight();
        let psrc = aug_graph[nsrc].0;
        let ptgt = aug_graph[ntgt].0;
        if psrc != ptgt {
            let e_opt = partition_graph.find_edge(psrc.into(), ptgt.into());
            let e = match e_opt {
                None => partition_graph.update_edge(psrc.into(), ptgt.into(), Vec::new()),
                Some(e) => e
            };

            let v = &mut partition_graph[e];
            v.push((nsrc.index() as u32, ntgt.index() as u32, K));
        }
    }

    return (partition_graph, aug_graph);
}

pub fn join_solutions(instance_vec: &Vec<BqmIsingInstance>, ){

}
pub fn run_gla(prog: &Prog, params: &GlaParams) -> Result<()>{
    simple_logger::SimpleLogger::new().with_level(log::LevelFilter::Info).env()
        .with_module_level("tamc::pt", log::LevelFilter::Off).init().unwrap();
    // Read in the instance and partition
    let mut gla_results = GlaResults::default();
    let instance = prog.read_instance();
    gla_results.instance_size = instance.size() as u32;
    let (partition_vecs, partition_mapping) = params.read_graph_partition(&instance)?;
    gla_results.num_partitions = partition_vecs.len() as u32;
    // Split the instance according to the partition data
    let (partition_graph, aug_graph) = instance_partition(partition_vecs, &partition_mapping, &instance);
    // Solve each sub-instance individually
    let sub_instance_indices = partition_graph.node_references()
        .map(|(_, n)| BTreeSet::from_iter(n.iter().copied())).collect_vec();

    let instance_vec = partition_graph.node_references()
        .map(|(_, n)| instance.induced_subgraph_instance_sorted(n))
        .collect_vec();
    // Gather the sub-instance solutions and evaluate the energies
    let num_instances = instance_vec.len();
    // seed and create random number generator
    let mut rngt = thread_rng();
    let mut seed_seq = [0u8; 32];
    rngt.fill_bytes(&mut seed_seq);
    let mut rng = Xoshiro256PlusPlus::from_seed(seed_seq);
    let mut rng_vec = Vec::with_capacity(num_instances as usize);
    for _ in 0..num_instances{
        rng_vec.push(rng.clone());
        rng.jump();
    };
    let runners = instance_vec.iter()
        .map(|inst| PtIcmRunner::new(&inst, &params.pt_params))
        .collect_vec();
    let mut pt_states = runners.iter().zip_eq(rng_vec.iter_mut())
        .map(|(pt, rng)| pt.generate_init_state(rng)).collect_vec();

    info!(" ** Running GLA with PT-ICM sub-solvers ...");
    let mut pt_results = Vec::with_capacity(num_instances);
    runners.par_iter().zip_eq(rng_vec.par_iter_mut().zip_eq(pt_states.par_iter_mut()))
        .map(|(pt, (rng, state))|
            pt.pt_loop(state, rng)).collect_into_vec(&mut pt_results);
    // Join together the ground state solutions
    let min_pt_states = pt_results.iter().zip_eq(instance_vec.iter())
        .map(|(res,inst)| IsingState::from_u64_vec(res.0.min_results.gs_states.last().unwrap(), inst.size() as u32).unwrap()).collect_vec();

    let min_pt_energies = pt_results.iter().map(|(m, t)| m.min_results.gs_energies.last().unwrap().to_owned()).collect_vec();

    let mut total_bnd_energy: f32 = 0.0;
    gla_results.partition_energies = min_pt_energies;
    let bulk_energy : f32 = gla_results.partition_energies.iter().copied().sum();
    info!("Bulk energy: {}", bulk_energy);
    // Evaluate boundary energies
    gla_results.boundary_energies.reserve(partition_graph.edge_count());
    for e in partition_graph.edge_references(){
        let mut bnd_energy = 0.0;
        let pi = e.source();
        let pj = e.target();
        let vi = partition_graph.node_weight(pi).unwrap();
        let vj = partition_graph.node_weight(pj).unwrap();
        for &(i, j, K) in e.weight(){
            let i_sub = vi.binary_search(&i).unwrap();
            let j_sub = vj.binary_search(&j).unwrap();
            bnd_energy += K * ((min_pt_states[pi.index()][i_sub]*min_pt_states[pj.index()][j_sub]) as f32);
        }
        total_bnd_energy += bnd_energy;
        gla_results.boundary_energies.push((pi.index() as u32, pj.index() as u32, bnd_energy));
    }
    info!("Boundary energy: {}", total_bnd_energy);
    gla_results.total_energy = bulk_energy + total_bnd_energy;
    info!("Total energy: {}", gla_results.total_energy);
    // Save the result
    write_data(&prog.output_file, &gla_results).with_context(
        || format!("Failed to write GLA data to {}", prog.output_file))?;
    Ok(())
}