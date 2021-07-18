use tamc::ising::BqmIsingInstance;
use tamc::util::connectivity_list::parse_line;
use tamc::{Prog, run_program};
use structopt::StructOpt;

fn main() {
    let opts: Prog = StructOpt::from_args();
    run_program(opts);
}