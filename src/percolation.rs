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
pub fn random_percolation<N, E, Rn: Rng+?Sized, D>(graph: &Graph<N, E, D>, p: f64, rng: &mut Rn)
-> (Option<Vec<NodeIndex>>, usize)
where D: EdgeType, N: Signed + std::cmp::PartialOrd,
{
    use petgraph::prelude::NodeIndex as Nd;
    use petgraph::visit::IntoNeighbors;
    use petgraph::algo::kosaraju_scc;
    let n = graph.node_count();
    let d = Bernoulli::new(p).unwrap();
    let rand_bools : Vec<bool> = (0..n).map(|_| rng.sample(d)).collect();
    let filtered_graph = NodeFiltered::from_fn(graph, |n| rand_bools[n.index()]);
    // Find the largest connected cluster
    let components = kosaraju_scc(&filtered_graph);
    let g_max = components.into_iter().max_by(|x,y| x.len().cmp(&y.len())).unwrap();

    //for n in filtered_graph.
    let init_nodes : HashSet<_>= (0..n).map(|i| Nd::new(i))
            .filter(|&i| graph[i] < N::zero() && rand_bools[i.index()]).collect();
    let tgt_nodes : HashSet<_>= (0..n).map(|i| Nd::new(i))
            .filter(|&i| graph[i] > N::zero() && rand_bools[i.index()]).collect();
    if init_nodes.len() == 0 || tgt_nodes.len() == 0{
        return (None, g_max.len());
    }
    // record each predecessor, mapping node → node
    let mut predecessor = HashMap::new();

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
            return (None, g_max.len())
        }
    };

    let mut next = v;
    let mut path = vec![next];
    while !init_nodes.contains(&next) {
        let &pred = predecessor.get(&next).unwrap();
        path.push(pred);
        next = pred;
    }
    return (Some(path), g_max.len());
}


#[cfg(test)]
mod tests{
    use ndarray::prelude::*;
    use rand::prelude::*;
    use petgraph::prelude::*;
    use petgraph::csr::Csr;
    use petgraph::visit::IntoNeighbors;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use crate::percolation::random_percolation;


    fn make_ising_2d(l: u32) -> Graph<i8, (), Undirected>{
        use sprs::TriMat;
        use petgraph::graph::NodeIndex as Nd;
        let n = l*l;

        let mut graph = Graph::with_capacity(n as usize, 2*n as usize);
        for _ in 0..n{
            graph.add_node(0);
        }
        let l = l as usize;
        for i in 0..l {
            for j in 0..l{
                let q0 = Nd::new(i*l + j);
                if i == 0{
                    graph[q0] = 1;
                } else if i == l-1{
                    graph[q0] = -1
                }
                let q1 = Nd::new(((i+1)%l)*l + j);
                let q2 = Nd::new(i*l + (j+1)%l);
                if i < l-1 {
                    graph.add_edge(q0, q1, ());
                }
                if j < l-1{
                    graph.add_edge(q0, q2,());
                }
            }
        }
        return graph;
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
            let (path, _) = random_percolation(&graph, p, &mut rng);
            if path.is_some(){
                num_per += 1;
            }
        }
        println!("{}/{}", num_per, iters)

    }

    #[test]
    fn test_percolation_instance(){
        let p = 0.4;
        let iters = 1000;
        use crate::util::{read_adjacency_list_from_file, adj_list_to_graph, bayesian_bootstrap};
        let instance = read_adjacency_list_from_file("./examples/data/qac_L8_FY_perc.txt").unwrap();
        let csr = adj_list_to_graph(&instance);
        let n = csr.node_count();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1234);

        let mut num_per = 0;
        let mut cl_vec = Vec::with_capacity(iters);
        for _ in 0..iters{
            let (path, cl) = random_percolation(&csr, p, &mut rng);
            cl_vec.push(cl as f64 / n as f64 );
            if path.is_some(){
                num_per += 1;
            }
        }
        let cl_arr = Array1::from_vec(cl_vec);
        let cl2: Array1<f64> = &cl_arr * &cl_arr;
        let cl3 = &cl2 * &cl_arr;
        let cl4: Array1<f64> = &cl2 * &cl2;
        let boots = bayesian_bootstrap(10,iters, &mut rng);
        let mean_cl2 = (boots.to_owned() * cl2.slice(s![NewAxis, ..])).sum_axis(Axis(1));
        let mean_cl4 = (boots.to_owned() * cl4.slice(s![NewAxis, ..])).sum_axis(Axis(1));
        let binder: Array1<f64> = (-mean_cl4 / (&mean_cl2 * &mean_cl2) + 3.0)/2.0;
        let g = binder.mean().unwrap();
        let g_err = binder.std(0.0);
        let cl_mean = cl_arr.mean().unwrap();
        println!("{}/{}", num_per, iters);
        println!("{}", cl_mean);
        println!("{} +- {}", g, g_err);
    }
}
