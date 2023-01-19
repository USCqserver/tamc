//!
//! Trait descriptions:
//!    An *instance* is a specification of a computational problem, including an
//!    energy (i.e. cost function), which may be parametric,
//!    to optimize or sample defined over a sample space.
//!    It must be immutable and thread-safe. It has associated types
//!
//!    A *state* is a mutable point in the sample space of an instance
//!
//!    An *entropy source* is a generalization of an Rng
//!
//!    A *sampler* advances a state using immutable data, an instance and parameters,
//!    along with an entropy source
//!
//!
use structopt::StructOpt;
use num_traits::{Num, Zero};
use num_traits::real::Real;
use rand::Rng;
use rand::distributions::{Distribution, Standard};
use rand::distributions::uniform::{SampleUniform, Uniform};
use std::marker::PhantomData;
use std::error::Error;
use std::ffi::OsStr;
pub use tamc_core::metropolis;
pub use tamc_core::traits::*;
use serde::{Serialize, Deserialize};
pub mod csr;
pub mod util;
pub mod ising;
pub mod percolation;
pub mod pt;
//pub mod sa;
pub mod ising_results;
use std::fs::File;
use crate::pt::PtIcmParams;
use std::fmt;
use std::path::Path;
use crate::ising::BqmIsingInstance;

#[derive(Serialize, Deserialize)]
pub struct PTOptions{
    pub icm: bool,
    pub beta_points: Vec<f64>
}

#[derive(Debug)]
pub enum PTError {
    IoError(std::io::Error, String),
    MethodParse(Box<dyn Error + 'static>, String)
}

impl fmt::Display for PTError{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self{
            PTError::IoError(e, s) => {
                write!(f, "Could not open '{}'", s)
            },
            PTError::MethodParse(e, s) =>{
                writeln!(f, "Failed to parse '{}' as valid input", s)?;
                write!(f, "{}", e.as_ref())
            }
        }
    }
}

impl std::error::Error for PTError{
    fn source(&self) -> Option<&(dyn Error + 'static)>{
        match self{
            PTError::IoError(e, _) => Some(e),
            PTError::MethodParse(e, _) => Some(e.as_ref())
        }
    }
}

#[derive(Serialize, Deserialize)]
pub enum Method{
    PT(PtIcmParams)
}

#[derive(StructOpt)]
pub struct Prog{
    pub method_file: String,
    pub instance_file: String,
    pub output_file: String,
    #[structopt(long)]
    pub suscepts: Vec<String>,
    #[structopt(long)]
    pub sample_output: Option<String>,
    #[structopt(long)]
    pub qubo: bool
}

impl Prog{
    pub fn read_instance(&self) -> BqmIsingInstance{
        let instance_file = &self.instance_file;
        let instance = ising::BqmIsingInstance::from_instance_file(&instance_file, self.qubo);
        return instance;
    }
    pub fn read_method(&self) -> Result<Method, Box<dyn Error>>{
        let method_file = &self.method_file;
        let yaml_str = std::fs::read_to_string(&method_file)
            .map_err(|e| PTError::IoError(e, method_file.to_string() ))?;
        let opts: Method = serde_yaml::from_str(&yaml_str)
            .map_err(|e| PTError::MethodParse(Box::new(e), method_file.to_string()))?;

        return Ok(opts);
    }
}


pub fn run_program(prog: Prog) -> Result<(), Box<dyn Error>>{
    let opts = prog.read_method()?;
    match &opts{
        Method::PT(pt_params) => {
            pt::run_parallel_tempering(&prog, &pt_params);
        }
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
