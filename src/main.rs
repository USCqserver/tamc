use tamc::ising::BqmIsingInstance;
use tamc::util::connectivity_list::parse_line;
use tamc::{Prog, run_program};
use structopt::StructOpt;


fn main() -> Result<(), Box<dyn std::error::Error>>{
    let opts: Prog = StructOpt::from_args();
    let result = run_program(opts);
    result.map_err(|e| {
        eprintln!("tamc terminated with an error:\n{}", &e);
        eprintln!(" * * * * * * ");
        e
    }).map(|x|{
        println!("tamc finished successfully");
        eprintln!(" * * * * * * ");
        x
    })
}