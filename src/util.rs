//use nom::multi::many1_countc;
use std::str::FromStr;
use std::io;
use std::io::{BufRead, BufReader};
use std::fs::File;
use std::collections::BTreeMap;
use nom::{InputTake, IResult};
use nom::character::complete::{digit1, space1};
use nom::character::complete::{digit0, multispace0, space0};
use nom::character::complete::char as nom_char;
use nom::combinator::opt;
use nom::multi::count;
use nom::sequence::{delimited, preceded, terminated};
use nom::sequence::{pair, tuple};




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


pub fn read_adjacency_list_from_file(filename: &str) -> Result<Vec<BTreeMap<usize, f64>>, std::io::Error> {
    let file = File::open(filename);
    let file = match file {Ok(f) => f, Err(e) => return Err(e) };

    read_adjacency_list(file)
}

pub fn read_adjacency_list<R: io::Read>(input: R) -> Result<Vec<BTreeMap<usize, f64>>, std::io::Error>
{
    use connectivity_list::parse_line;
    use std::cmp::max;
    use std::error::Error;

    let reader = BufReader::new(input);

    let mut adj_list : Vec<BTreeMap<usize, f64>> = Vec::new();
    for (_i, line) in reader.lines().enumerate(){
        let line = match line{Ok(l) => l, Err(e) => return Err(e)};
        match parse_line(&line){
            Ok((i, j, h)) => {
                let i = i as usize; let j = j as usize;
                let m = max(i,j) + 1; //max number of indices, zero indexed
                if adj_list.len() < m{
                    adj_list.resize(m, Default::default());
                }
                adj_list[i].insert(j, h).map(
                    |h2| println!("The entry ({}, {}) :> {} was overwritten with {}.", i, j, h2, h));
                if i != j {
                    adj_list[j].insert(i, h).map(
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