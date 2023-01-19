use std::io::Write;
use rayon::prelude::*;
use simple_logger::SimpleLogger;
use log::{info, debug, warn};
use structopt::StructOpt;
use tamc_core::pt::PTState;
use tamc::Method;
use tamc::ising;
use tamc::pt::{BetaOptions, PtIcmParams, PtIcmRunner};
use tamc::ising::{BqmIsingInstance, IsingState};
use ndarray::prelude::*;
use tamc_core::util::monotonic_divisions;

#[derive(Debug, StructOpt)]
#[structopt(name = "pt-opt", about = "Optimize the PT temperature set over a set of instances.")]
struct PtOptim{
    #[structopt(long, default_value="1000")]
    max_iters: u32,
    #[structopt(long, default_value="0.2")]
    step_size: f32,
    #[structopt(long, default_value="0.85")]
    momentum: f32,
    #[structopt(long, default_value="0.02", help="Covergence tolerance criterion (MAE of log-betas)")]
    tolerance: f32,
    #[structopt(long, default_value="opt_params.yml", help="Destination for optimized parameters")]
    opt_params: String,
    #[structopt(long, default_value="tau_hist.csv")]
    tau_hist: String,
    params: String,
    instances: Vec<String>
}


pub fn pt_optimize_beta(
    instances: &Vec<BqmIsingInstance>,
    params: &PtIcmParams,
    num_iters: u32,
    alpha: f32,
    m: f32,
    tol: f32
) -> (PtIcmParams, Array2<f32>) {
    use interp::interp;
    use tamc_core::util::{StepwiseMeasure, finite_differences, monotonic_bisection};

    let num_instances = instances.len();
    let mut params = params.clone();
    let num_chains = params.num_replica_chains;
    // Anneal the step-size over num_iters

    let mut init_states = Vec::with_capacity(instances.len());
    init_states.resize(instances.len(), None);
    let init_beta_vec = params.beta.get_beta_arr();
    let nt = init_beta_vec.len();
    // We use a momentum-directed iteration optimizer
    let mut momentum_beta = Array1::from_vec(init_beta_vec.clone());
    let mut tau_hist : Vec<Array1<f32>> = Vec::new();

    for i in 0..num_iters {
        println!("* Iteration {}.\n* Step Size: {}", i, alpha);

        let beta_vec = params.beta.get_beta_arr();

        let beta_meas = StepwiseMeasure::new(beta_vec.clone());
        let beta_weights = Array1::from_vec(beta_meas.weights.clone());
        let beta_arr = Array1::from_vec(beta_vec.clone());
        let beta_difs = &beta_arr.slice(s![1..]) - &beta_arr.slice(s![0..-1]);
        // Run PT on the current temperature array on all replicas
        let pticm_vec: Vec<PtIcmRunner> = instances.iter()
            .map(|i| PtIcmRunner::new(i, &params)).collect();
        let results: Vec<Vec<PTState<IsingState>>> = pticm_vec.par_iter().zip_eq(init_states.par_iter())
            .map(|(p, s)| p.run(s.clone()).2).collect();
        // Gather the diffusion histograms for each temperature summed over all replica chains
        // Also evaluate the round trip times
        let mut dif_probs_vec : Vec<Array1<f32>> = Vec::with_capacity(num_instances);
        let mut tau_vec: Vec<f32> = Vec::with_capacity(num_instances);
        for s in results.iter(){
            let dif_hists : Vec<ArrayView2<u32>>= s.iter().map(|ptstate| ptstate.diffusion_hist.view()).collect();
            // number of round trips in all replica chains
            let rts : u32 =  s.iter().map(|ptstate| ptstate.round_trips).sum();
            let rt_per_rep = rts as f32 / ((num_chains * nt as u32 ) as f32 );
            // The typical rount-trip time per replica is  num_sweeps / (N_t * \bar{\tau} )
            let tau = if rts == 0 { 0.0 } else { (params.num_sweeps as f32)/rt_per_rep };
            tau_vec.push(tau);
            let n = dif_hists.len();
            let sh = dif_hists[0].raw_dim();
            let mut sum_dif_hists = Array2::zeros(sh);
            for h in dif_hists.iter() {
                sum_dif_hists += h;
            }
            //pseudocount of 1 to prevent division by 0
            let sum_dif_hists = sum_dif_hists.map(|&x| (x + 1) as f32);
            let tots = sum_dif_hists.sum_axis(Axis(1));
            // n_maxbeta / (n_minbeta + n_maxbeta)
            let dif_probs = sum_dif_hists.slice(s![.., 1]).to_owned() / tots;
            dif_probs_vec.push(dif_probs);
        }
        println!("(Peek) diffusion distribution: {:5.4}", &dif_probs_vec[0]);
        let tau_arr = Array1::from_vec(tau_vec);
        let sum_tau = tau_arr.sum();
        println!("Round-trip times (sweeps):\n{}", tau_arr);
        let d_dif_vec : Vec<Array1<f32>> = dif_probs_vec.iter()
            .map(|f| Array1::from_vec(finite_differences(beta_arr.as_slice().unwrap(),
                                                         f.as_slice().unwrap())) )
            .collect();
        let mut weighted_d_dif = Array1::zeros(nt);
        for (&tau, d_dif) in tau_arr.iter().zip(d_dif_vec.iter()){
            weighted_d_dif.scaled_add(tau, d_dif);
        }
        weighted_d_dif /= sum_tau;
        tau_hist.push(tau_arr);
        //println!("Weighed df/dT: {:5.4}", weighted_d_dif);
        let unnorm_eta2 : Array1<f32> = &weighted_d_dif / &beta_weights;
        let unnorm_eta = unnorm_eta2.map(|&x| f32::sqrt(x.max(0.0)));
        // Trapezoid rule correction
        let unnorm_eta_w = (unnorm_eta.slice(s![0..-1]).to_owned() + unnorm_eta.slice(s![1..]))/2.0;
        // eta times the integration measure
        let unnorm_eta_w: Array1<f32> = unnorm_eta_w * beta_difs;
        let z = unnorm_eta_w.sum();
        if z < f32::EPSILON{
            warn!(" ** Insufficient round trips for eta CDF");
            continue;
        }
        let eta_arr_samps = &unnorm_eta / z; // Eta at the temperature points
        let eta_arr_w : Array1<f32> = &unnorm_eta_w / z; // Weighed eta for integration
        //println!("Eta: {:5.4}", eta_arr);
        let eta_w_vec = eta_arr_w.into_raw_vec();
        let eta_cdf : Vec<f32>= eta_w_vec.iter()
            .scan(0.0, |acc, x|{let acc0 = *acc; *acc += x; Some(acc0)})
            .chain(std::iter::once(1.0))
            .collect();
        let eta_cdf_arr = Array1::from_vec(eta_cdf.clone());
        //println!("Eta CDF:\n {:5.4}", eta_cdf_arr);
        let &beta_min = &beta_vec[0];
        let &beta_max = &beta_vec[nt -1];
        let eta_fn = |x|{
            let beta = beta_min + x * (beta_max-beta_min);
            interp(&beta_vec, &eta_cdf, beta)
        };
        let xdivs = monotonic_divisions(eta_fn, (nt -1) as u32);
        let xdivs = Array1::from(xdivs);
        let mut beta_divs : Array1<f32> = xdivs*(beta_max - beta_min) + beta_min;
        let calc_beta_divs = beta_divs.clone();
        // Mean of |log10(b_calc/b_current)|
        let mut err = 0.0;
        //println!("Calculated beta:\n{:5.4}", beta_divs);
        // Update momentum
        for (bp, &b1) in momentum_beta.iter_mut().zip(calc_beta_divs.iter()){
            *bp = f32::exp(m * (*bp).ln() + (1.0 - m) * b1.ln());
        }
        //println!("Momentum beta:\n{:5.4}", momentum_beta);

        for (b2, (b1, bp)) in beta_divs.iter_mut()
            .zip(beta_vec.iter().zip(momentum_beta.iter()))
            .skip(1).take(nt-2){
            let b = f32::exp(alpha * bp.ln() + (1.0-alpha)*b1.ln());
            err += f32::abs(b2.log10() - b1.log10());
            *b2 = b
        }
        err /= nt as f32;

        println!("{:^9}{:^9}{:^9}{:^9}{:^9}{:^9}",
                 "DDif", "eta", "etaCDF", "beta0", "betaM", "beta");
        println!("--------------------------------------------------");

        for i in 0..nt{
            print!(" {:7.3} ", weighted_d_dif[i]);
            print!(" {:7.3} ", eta_arr_samps[i]);
            print!(" {:7.4} ", eta_cdf_arr[i]);
            print!(" {:7.4} ", calc_beta_divs[i]);
            print!(" {:7.4} ", momentum_beta[i]);
            print!(" {:7.4} \n", beta_divs[i]);
        }
        println!("Mean abs log rel_err: {}", err);
        params.beta = BetaOptions::Arr(beta_divs.into_raw_vec());
        // let f_arr = results.iter()
        //     .map(|res| res.final_state.iter().map(|c|))
        //println!(" Diffusion function");
        init_states = results.into_iter()
            .map(|mut res| {
                for r in res.iter_mut(){ r.reset_tags() };
                Some(res) })
            .collect();
        if err < tol {
            println!(" ** Relative Error converged");
            break;
        }
        //params.beta = BetaOptions::Arr()
    }

    let final_beta = Array1::from_vec(params.beta.get_beta_arr());
    println!("Final temperature array:\n{:6.5}", final_beta);
    let tau_view : Vec<ArrayView1<f32>> = tau_hist.iter().map(|v|v.view()).collect();
    let tau_hist_arr = ndarray::stack(Axis(0), &tau_view).unwrap();
    //double the number of sweeps
    params.num_sweeps *= 2;

    return (params, tau_hist_arr);
}

fn main() {
    let prog : PtOptim = StructOpt::from_args();
    SimpleLogger::new().with_level(log::LevelFilter::Warn).init().unwrap();
    let method_file = prog.params;
    let instance_files = &prog.instances;
    let instance_vec : Vec<BqmIsingInstance> = instance_files.iter()
        .map(|s| BqmIsingInstance::from_instance_file(&s, false)).collect();
    let yaml_str = std::fs::read_to_string(method_file).unwrap();
    let opts: Method= serde_yaml::from_str(&yaml_str).unwrap();
    match opts{
        Method::PT(pt_params) => {
            let (opt_params, tau_hist) = pt_optimize_beta(
                &instance_vec, &pt_params, prog.max_iters, prog.step_size, prog.momentum, prog.tolerance);
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
        },
        _ => {
            warn!("Must input a PT method for tamc-pt-opt")
        }
    };
}