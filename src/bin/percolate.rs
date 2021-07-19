use std::sync::Mutex;
use rayon::prelude::*;
use tamc::percolation::random_percolation;
use tamc::util::{read_adjacency_list_from_file, adj_list_to_csr};
use structopt::StructOpt;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
#[derive(Debug, StructOpt)]
#[structopt(name = "percolate", about = "Perform a site percolation calculation on an arbitrary graph.")]
struct Percolate{
    #[structopt(short, long, default_value="100")]
    num_samples: u32,
    #[structopt(short,long)]
    prob: f64,
    instance: String
}
fn main() {
    let args: Percolate = StructOpt::from_args();
    let instance = read_adjacency_list_from_file(&args.instance).unwrap();
    let csr = adj_list_to_csr(&instance);

    let mut rngt = thread_rng();
    let mut seed_seq = [0u8; 32];
    rngt.fill_bytes(&mut seed_seq);

    let rng_mut = Mutex::new(Xoshiro256PlusPlus::from_seed(seed_seq));


    let num_percs : u32 = (0..args.num_samples).into_par_iter().map_init(
        ||{let mut r = rng_mut.lock().unwrap();
            r.jump();
            r.clone()
        },
        |rng, _|{
        random_percolation(&csr, args.prob, rng).is_some() as u32
    }).sum();
    println!(" * Percolation results *");
    println!("{}/{}", num_percs, args.num_samples);
    println!("{}", (num_percs as f64) / (args.num_samples as f64))
}