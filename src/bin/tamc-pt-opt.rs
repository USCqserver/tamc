use std::io::Write;
use simple_logger::SimpleLogger;
use structopt::StructOpt;
use tamc::Method;
use tamc::ising;
use tamc::ising::{BqmIsingInstance, PtIcmParams};

#[derive(Debug, StructOpt)]
#[structopt(name = "pt-opt", about = "Optimize the PT temperature set over a set of instances.")]
struct PtOptim{
    #[structopt(long, default_value="1000")]
    max_iters: u32,
    #[structopt(long, default_value="opt_params.yml", help="Destination for optimized parameters")]
    opt_params: String,
    #[structopt(long, default_value="tau_hist.csv")]
    tau_hist: String,
    params: String,
    instances: Vec<String>
}

fn main() {
    let prog : PtOptim = StructOpt::from_args();
    SimpleLogger::new().with_level(log::LevelFilter::Warn).init().unwrap();
    let method_file = prog.params;
    let instance_files = &prog.instances;
    let instance_vec : Vec<BqmIsingInstance> = instance_files.iter()
        .map(|s| BqmIsingInstance::from_instance_file(&s)).collect();
    let yaml_str = std::fs::read_to_string(method_file).unwrap();
    let opts: Method= serde_yaml::from_str(&yaml_str).unwrap();
    match opts{
        Method::PT(pt_params) => {
            let (opt_params, tau_hist) = ising::pt_optimize_beta(&instance_vec, &pt_params, prog.max_iters);
            let opt_method = Method::PT(opt_params);
            let yaml_string = serde_yaml::to_string(&opt_method).unwrap();
            println!("{}", &yaml_string);
            println!("\n\n ** Writing to {} **", prog.opt_params);
            std::fs::write(&prog.opt_params, &yaml_string);
            let mut f = std::fs::File::create(&prog.tau_hist).unwrap();
            for row in tau_hist.rows(){
                for x in row.iter(){
                    write!(f, "{},", x);
                }
                write!(f, "\n");
            }
        }
    };
}