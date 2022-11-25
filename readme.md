TAMC: "Tempering and Annealing" Monte Carlo

This library provides utilites for specifying and running 
Monte Carlo sampling algorithms. It is primarily aimed for sampling 
and minimizing discrete binary/Ising problems.

*Usage*: 
```
    tamc [FLAGS] [OPTIONS] <method-file> <instance-file> <output-file>

FLAGS:
    -h, --help       Prints help information
        --qubo       
    -V, --version    Prints version information

OPTIONS:
        --sample-output <sample-output>    
        --suscepts <suscepts>...           

ARGS:
    <method-file>      
    <instance-file>    
    <output-file>    
```

`method-file` is a specification of the simulation and options, 
such as number of replicas and temperatures in YAML format. 
A good recommended starting point for Ising problems with J~1 is
```yaml
---
PT:
  num_sweeps: 2000
  warmup_fraction: 0.5
  beta:
    Geometric:
      beta_min: 0.2
      beta_max: 5.0
      num_beta: 32
  lo_beta: 1.0
  icm: true
  num_replica_chains: 2
  threads: 1
  sample: 32
  sample_states: 32
  sample_limiting: 2
```

`instance-file` is the specification of the Ising problem to sample/solve.
It should follow the informal standard `i j K` format, where `i` and `j` are zero-based
integeres and `K` is a floating point value of the coupling strength.
If `i==j`, then `K` is interpreted as a bias.
```text
0 1 -1.0
1 2 -1.0
2 3 -1.0
......
```

`--suscepts` is an option to provide one or more plain-text new-line delimited
files of `N` floating point numbers, where `N` is the problem size.
If provided, these numbers specify coefficients for weighed replica overlaps,
which are required for susceptibility measurements.
On a square lattice, these should simply be Fourier cofficients.
For general graphs, one can pass eigenvectors of the graph laplacian.

The main `output-file` saves ground state information in readable YAML format.
If thermal sampling is on, `sample-output` saves thermal data in either binary (default) or 
Python pickle format (if the output file extension is `.pkl`).
The thermal sampling output data structure is as follows:
```rust
pub struct PtIcmThermalSamples{
    // number of variables in the problem (N)
    pub instance_size: u64, 
    // simulation temperatures used
    pub beta_arr: Vec<f32>, 
    // PT Chain x Temperature x Time:  Thermal samples of replicas
    // in bit-packed format
    pub samples: Vec<Vec<Vec<u8>>>,
    // Temperature x Time: Thermal samples of replica energy
    pub e: Vec<Vec<f32>>,
    // Temperature x Time: Thermal samples of replica overlap
    pub q: Vec<Vec<i32>>,
    // (If specified) Temperature x Eigenvector x Time
    pub suscept: Vec<Vec<Vec<f32>>>
}
```