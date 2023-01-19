use serde::{Deserialize, Serialize};
use ndarray::Array1;

#[derive(Clone, Serialize, Deserialize)]
pub struct MinResults{
    pub timing: f64,
    pub gs_time_steps: Vec<u32>,
    pub gs_energies: Vec<f32>,
    pub gs_states: Vec<Vec<u64>>,
    pub num_measurements: u32,
    pub instance_size: u32,
    //pub final_state: Vec<PTState<IsingState>>
}

impl MinResults{
    pub fn new( num_betas: u32, instance_size: u32) -> Self{

        return Self{
            gs_states: Vec::new(),
            gs_energies: Vec::new(),
            gs_time_steps: Vec::new(),
            num_measurements: 0,
            timing: 0.0,
            instance_size
            //final_state: Vec::new()
        };
    }
}