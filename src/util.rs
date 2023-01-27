//use nom::multi::many1_countc;
use std::str::FromStr;
use std::io;
use std::io::{BufRead, BufReader};
use std::fs::File;
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::Path;
use log::debug;
use ndarray::prelude::*;
use nom::{InputTake, IResult};
use nom::character::complete::{digit1, space1};
use nom::character::complete::{digit0, multispace0, space0};
use nom::character::complete::char as nom_char;
use nom::combinator::opt;
use nom::multi::{count, many0};
use nom::multi::many1;
use nom::sequence::{delimited, preceded, terminated};
use nom::sequence::{pair, tuple};
use petgraph::prelude::*;
use petgraph::csr::Csr;
use petgraph::Undirected;
use rand::prelude::*;
use serde::Serialize;

pub fn write_data<P: AsRef<Path>+AsRef<OsStr>, T: Serialize>(output_file: &P, ser_data: &T) -> anyhow::Result<()>{

    let mut f = File::create(output_file)?;
    let ext = Path::new(&output_file).extension().and_then(OsStr::to_str);
    if ext == Some("pkl"){
        serde_pickle::to_writer(&mut f, ser_data, serde_pickle::SerOptions::default())?;
    } else {
        bincode::serialize_into(&mut f, ser_data)?;
    }
    Ok(())
}

fn parse_u32(s: &str) -> IResult<&str, u32> {
    digit1(s).map(
        |(i, o)| (i, u32::from_str(&o).unwrap()))
}

fn parse_fixed(s: &str) -> IResult<&str, f64> {
    let fixed_res =
        pair(opt(nom_char('-')),
             pair(digit1,
                  opt(preceded(nom_char('.'), digit0))))(s);

    fixed_res.map(|(i, _ )| {
        let n = s.len() - i.len();
        (i, s.take(n).parse().unwrap())
    }
    )
}

pub mod connectivity_list{
    use super::*;

    pub fn parse_line(line: &str) -> Result<(u32, u32, f64), nom::Err<nom::error::Error<&str>> > {
        let mut parser = pair(
            count(delimited(space0, parse_u32, space1), 2),
            terminated(parse_fixed, multispace0));
        let (i, (ints, d)) = parser(line)?;
        return Ok((ints[0], ints[1], d))
    }
}


pub fn read_adjacency_list_from_file(filename: &str) -> Result<Vec<BTreeMap<usize, f32>>, io::Error> {
    let file = File::open(filename);
    let file = match file {Ok(f) => f, Err(e) => return Err(e) };

    read_adjacency_list(file)
}

pub fn read_adjacency_list<R: io::Read>(input: R) -> Result<Vec<BTreeMap<usize, f32>>, io::Error>
{
    use connectivity_list::parse_line;
    use std::cmp::max;
    use std::error::Error;

    let reader = BufReader::new(input);

    let mut adj_list : Vec<BTreeMap<usize, f32>> = Vec::new();
    for (_i, line) in reader.lines().enumerate(){
        let line = match line{Ok(l) => l, Err(e) => return Err(e)};
        match parse_line(&line){
            Ok((i, j, h)) => {
                let i = i as usize; let j = j as usize;
                let m = max(i,j) + 1; //max number of indices, zero indexed
                if adj_list.len() < m{
                    adj_list.resize(m, Default::default());
                }
                adj_list[i].insert(j, h as f32).map(
                    |h2| println!("The entry ({}, {}) :> {} was overwritten with {}.", i, j, h2, h));
                if i != j {
                    adj_list[j].insert(i, h as f32).map(
                        |h2| println!("The entry ({}, {}) :> {} was overwritten with {}.", i, j, h2, h));
                }
            }
            Err(e) =>{
                println!("Ignoring line {}: {}", _i, e)
            }
        }

    };

    Ok(adj_list)
}


pub fn read_txt_vec<R: io::Read>(input: R) -> Result<Vec<f64>, io::Error>
{
    use std::cmp::max;
    use std::error::Error;

    pub fn parse_line(line: &str) -> Result<f64, nom::Err<nom::error::Error<&str>> >{
        let mut parser = delimited(space0, parse_fixed, multispace0);
        let (i, d) = parser(line)?;
        return Ok(d);
    }

    let reader = BufReader::new(input);

    let mut dvec : Vec<f64> = Vec::new();
    for (_i, line) in reader.lines().enumerate(){
        let line = match line{Ok(l) => l, Err(e) => return Err(e)};
        match parse_line(&line){
            Ok(d) => {
                dvec.push(d);
            }
            Err(e) =>{
                println!("Ignoring line {}: {}", _i, e)
            }
        }
    };

    Ok(dvec)
}


pub fn read_u32_lines<R: io::Read>(input: R) -> Result<Vec<Vec<u32>>, io::Error>
{
    use std::cmp::max;
    use std::error::Error;

    pub fn parse_line(line: &str) -> Result<Vec<u32>, nom::Err<nom::error::Error<&str>> >{
        let mut float_parser = delimited(space0, parse_u32, multispace0);
        let mut line_parser = many1(float_parser);
        let (i, d) = line_parser(line)?;
        return Ok(d);
    }

    let reader = BufReader::new(input);

    let mut dvec : Vec<Vec<u32>> = Vec::new();
    for (_i, line) in reader.lines().enumerate(){
        let line = match line{Ok(l) => l, Err(e) => return Err(e)};
        match parse_line(&line){
            Ok(d) => {
                dvec.push(d);
            }
            Err(e) =>{
                if line.len() > 0 {
                    debug!("Ignoring line {}: {}", _i, e);
                }
            }
        }
    };

    Ok(dvec)
}


pub fn adj_list_to_graph(adj_list: &Vec<BTreeMap<usize, f32>>) -> Graph<f32, f32, Undirected>{
    use petgraph::prelude::NodeIndex as Nd;
    let n = adj_list.len();
    let mut graph = Graph::new_undirected();
    graph.reserve_nodes(n);
    for _ in 0..n{
        graph.add_node(0.0);
    }

    for (i, l) in adj_list.iter().enumerate(){
        for (&j, &K) in l.iter(){
            if i == j{
                graph[Nd::new(i)] = K
            } else {
                graph.add_edge(Nd::new(i ), Nd::new(j ), K);
            }
        }
    }

    return graph;
}

pub fn bayesian_bootstrap<Rn: Rng+?Sized>(num_bootstraps: usize, n: usize, rng: &mut Rn) -> Array2<f64>{
    use rand::distributions::Standard;
    // generate random gaps
    let mut rn_arr = Array2::from_shape_fn(
        (num_bootstraps, n+1),
        |(i,j)|
            if j == 0 { 0.0 }
            else if j < n { rng.sample(Standard) }
            else { 1.0 });
    for i in 0..num_bootstraps{
        let mut row = rn_arr.slice_mut(s![i, ..]);
        row.as_slice_mut().unwrap().sort_by(|&x,y|x.partial_cmp(y).unwrap());
    }
    let gaps : Array2<f64> = rn_arr.slice(s![.., 1..]).to_owned() - rn_arr.slice(s![.., 0..-1]);

    return gaps
}