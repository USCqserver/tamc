use std::sync::Mutex;
use rayon::prelude::*;
use tamc::percolation::random_percolation;
use tamc::util::{read_adjacency_list_from_file, adj_list_to_graph};
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
    #[structop(short,long)]
    output: Option<String>,
    instance: String
}
fn main() {
    let args: Percolate = StructOpt::from_args();
    let instance = read_adjacency_list_from_file(&args.instance).unwrap();
    let csr = adj_list_to_graph(&instance);

    let mut rngt = thread_rng();
    let mut seed_seq = [0u8; 32];
    rngt.fill_bytes(&mut seed_seq);

    let rng_mut = Mutex::new(Xoshiro256PlusPlus::from_seed(seed_seq));


    let perc_results : Vec<_> = (0..args.num_samples).into_par_iter().map_init(
        ||{let mut r = rng_mut.lock().unwrap();
            r.jump();
            r.clone()
        },
        |rng, _|{
        random_percolation(&csr, args.prob, rng)
    }).collect();
    let num_percs: u32 = perc_results.iter().map(|x|x.0.is_some() as u32).sum();
    println!("# * Percolation results *");
    println!("samps,  num_perc,  frac_perc");
    println!("{}, {}, {}", args.num_samples, num_percs, (num_percs as f64) / (args.num_samples as f64));

    if let Some(output_file ) = &args.output{
        let mut f = std::fs::File::create(output_file).unwrap();
        writeln!("perc, max_cluster");
        for (p, cl) in perc_results.iter(){
            writeln!(f, "{}, {}", p, cl)
        }
    }
}