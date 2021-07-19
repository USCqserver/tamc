use rand::Rng;
use rand::seq::IteratorRandom;
use rand::distributions::Bernoulli;
use petgraph::csr::Csr;
use petgraph::{EdgeType};
use petgraph::prelude::*;
use petgraph::visit::{depth_first_search, DfsEvent, Control, NodeFiltered, Dfs};
use std::collections::{HashSet, HashMap};
use num_traits::{Num, Signed};
/// Generate a random top to bottom percolation of the graph
pub fn random_percolation<N, E, Rn: Rng+?Sized, D>(graph: &Csr<N, E, D>, p: f64, rng: &mut Rn)
-> Option<Vec<u32>>
where D: EdgeType, N: Signed + std::cmp::PartialOrd,
{
    use petgraph::visit::IntoNeighbors;
    let n = graph.node_count();
    let d = Bernoulli::new(p).unwrap();
    let rand_bools : Vec<bool> = (0..n).map(|_| rng.sample(d)).collect();
    let filtered_graph = NodeFiltered::from_fn(graph, |n| rand_bools[n as usize]);
    //for n in filtered_graph.
    let init_nodes : HashSet<u32>= (0..n as u32).filter(|&i| graph[i] < N::zero() && rand_bools[i as usize]).collect();
    let tgt_nodes : HashSet<u32>= (0..n as u32).filter(|&i| graph[i] > N::zero() && rand_bools[i as usize]).collect();
    if init_nodes.len() == 0 || tgt_nodes.len() == 0{
        return None;
    }
    // record each predecessor, mapping node → node
    let mut predecessor : HashMap<u32, u32> = HashMap::new();

    let c = depth_first_search(&filtered_graph, init_nodes.clone(), |event| {
        match event{
            DfsEvent::TreeEdge(u, v) => {
                predecessor.insert(v, u);
                if tgt_nodes.contains(&v) {
                    Control::Break(v)
                } else {
                    Control::Continue
                }
            },
            _ => Control::Continue
        }

        // if let DfsEvent::TreeEdge(u, v) = event {
        //     predecessor[v as usize] = u;
        //     if tgt_nodes.contains(&v) {
        //         return Control::Break(v);
        //     }
        // }
        // Control::Continue
    });
    let v = match c {
        Control::Break(v) =>{
            v
        },
        _ => {
            return None
        }
    };

    let mut next = v;
    let mut path = vec![next];
    while !init_nodes.contains(&next) {
        let &pred = predecessor.get(&next).unwrap();
        path.push(pred);
        next = pred;
    }
    return Some(path);
}


#[cfg(test)]
mod tests{
    use rand::prelude::*;
    use petgraph::prelude::*;
    use petgraph::csr::Csr;
    use petgraph::visit::IntoNeighbors;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use crate::percolation::random_percolation;

    fn make_ising_2d(l: u32) -> Csr<i8, (), Undirected>{
        use sprs::TriMat;
        let n = l*l;
        let mut csr = Csr::with_nodes((l*l) as usize);

        for i in 0..l as u32{
            for j in 0..l as u32{
                let q0 = i*l + j;
                if i == 0{
                    csr[q0] = 1;
                } else if i == l-1{
                    csr[q0] = -1
                }
                let q1 = ((i+1)%l)*l + j;
                let q2 = i*l + (j+1)%l;
                if i < l-1 {
                    csr.add_edge(q0, q1, ());
                }
                if j < l-1{
                    csr.add_edge(q0, q2,());
                }
            }
        }
        return csr;
    }
    #[test]
    fn test_percolation_square_2d(){
        let iters = 10000;
        let l = 16;
        let graph = make_ising_2d(l);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1234);
        let p = 0.59;
        let mut num_per = 0;
        for _ in 0..iters{
            let path = random_percolation(&graph, p, &mut rng);
            if path.is_some(){
                num_per += 1;
            }
        }
        println!("{}/{}", num_per, iters)

    }

    #[test]
    fn test_percolation_instance(){
        let p = 0.8;
        let iters = 1000;
        use crate::util::{read_adjacency_list_from_file, adj_list_to_csr};
        let instance = read_adjacency_list_from_file("./examples/data/qac_L8_FY_perc.txt").unwrap();
        let csr = adj_list_to_csr(&instance);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1234);

        let mut num_per = 0;
        for _ in 0..iters{
            let path = random_percolation(&csr, p, &mut rng);
            if path.is_some(){
                num_per += 1;
            }
        }
        println!("{}/{}", num_per, iters);
    }
}
