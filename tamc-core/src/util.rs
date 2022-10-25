use std::fmt::Display;
use num_traits::{Float};

/// Evaluate the finite differences derivatives at each point (x, y)
/// where the x grid may be irregular
pub fn finite_differences<F:Float>(x: &[F], y: &[F]) -> Vec<F>
{
    let n = x.len();
    let delta_x: Vec<F> = x.iter().zip(x.iter().skip(1))
        .map(|(&x1, &x2)| x2 - x1).collect();
    let delta_y: Vec<F> = y.iter().zip(y.iter().skip(1))
        .map(|(&y1, &y2)| y2 - y1).collect();
    let mut deriv = Vec::with_capacity(n);
    deriv.resize(n, F::zero());
    deriv[0] = delta_y[0] / delta_x[0];
    *deriv.last_mut().unwrap() = *delta_y.last().unwrap()/ *delta_x.last().unwrap();
    for i in 1..n-1{
        deriv[i] = delta_y[i] / delta_x[i-1] + delta_y[i-1]/delta_x[i];
    }
    return deriv;
}

pub struct StepwiseMeasure<F>{
    pub step_pnts: Vec<F>,
    pub weights: Vec<F>
}
impl<F: Float> StepwiseMeasure<F>{
    pub fn new(step_pnts: Vec<F>) -> Self{
        let n = step_pnts.len();
        let delta_x : Vec<F> = step_pnts.iter().zip(step_pnts.iter().skip(1))
            .map(|(&x1, &x2)| (x2 - x1)).collect();
        let mut weights = Vec::with_capacity(n);
        weights.resize(n, F::zero());
        weights[0] = delta_x[0]/F::from(2.0).unwrap();
        for i in 1..n-1{
            weights[i] = (delta_x[i-1] + delta_x[i])/F::from(2.0).unwrap();
        }
        weights[n-1] = delta_x[n-2]/F::from(2.0).unwrap();

        return Self{step_pnts, weights}
    }
}

pub enum BisResult<Ft>{
    Root((Ft, Ft)),
    LoBnd((Ft, Ft)),
    UpBnd((Ft, Ft)),
    Failed
}

impl<Ft: Copy> BisResult<Ft>{
    pub fn unwrap_root(&self) -> (Ft, Ft){
        if let &Self::Root((x,y)) = self{
            return (x,y);
        } else{
            panic!("BisResult::unwrap_root failed");
        }
    }
}

pub fn monotonic_bisection<F: Fn(Ft)->Ft, Ft: Float> (
    f: F, y0: Ft, xmin: Ft, xmax: Ft, tol: Ft, iters_max: u32,
            )
-> BisResult<Ft>
{
    assert!(xmin < xmax);

    let mut x0 = xmin;
    let mut x1 = xmax;
    for _ in 0..iters_max {
        let xmid = (x0 + x1)/Ft::from(2.0).unwrap();
        let y = f(xmid);
        if (y-y0).abs() < tol{
            return BisResult::Root((xmid, y))
        }
        if (xmid - xmin).abs() < tol{
            return BisResult::LoBnd((xmid,y))
        }
        if (xmid - xmax).abs() < tol{
            return BisResult::UpBnd((xmid, y))
        }

        if y < y0{
            x0 = xmid
        } else {
            x1 = xmid;
        }
    }

    return BisResult::Failed;
}

/// roots n
/// ytgt n
fn bisect_array<F: Fn(Ft)->Ft + Copy,
    Ft: Float + Display>(f: F, roots: &mut [Ft], ytgt: &[Ft],
                                 xmin: Ft, xmax: Ft){
    // Base cases
    if roots.len() == 0{
        return;
    } else if roots.len() == 1{
        let y0 = ytgt[0];
        let bis = monotonic_bisection(f, y0, xmin, xmax,  Ft::from(1.0e-4).unwrap(), 100000);
        match bis{
            BisResult::Root((x,y)) => { roots[0] = x;}
            BisResult::LoBnd((x, y)) => {
                panic!("Bisection for {} in the interval [{},{}] terminated at the lower bound ({}, {})",
                       y0, xmin, xmax, x, y)}
            BisResult::UpBnd((x, y)) => { panic!("Bisection for {} in the  interval [{},{}] terminated at the upper bound ({}, {})",
                                         y0, xmin, xmax, x, y)}
            BisResult::Failed => { panic!("Bisection for {} in the  interval [{},{}] failed ", y0, xmin, xmax)}
        }
        let (x,y) = monotonic_bisection(f, y0, xmin, xmax,  Ft::from(1.0e-4).unwrap(), 10000).unwrap_root();
        roots[0] = x;
    } else {
        let n = roots.len();
        let mid = n/2;
        bisect_array(f, &mut roots[mid..mid+1], &ytgt[mid..mid+1], xmin, xmax );
        let xmid = roots[mid];
        bisect_array(f, &mut roots[0..mid], &ytgt[0..mid], xmin, xmid);
        bisect_array(f, &mut roots[mid+1..n], & ytgt[mid+1..n], xmid, xmax);
    }
}

/// divide the domain of a monotonic function f: [0, 1] -> [0, 1]
/// into m intervals [x_k, x_{k+1}] such that f([x_k, x_{k+1}]) = [k/m, (k+1)/m]
/// Returns a vector of length m+1 consisting of {x_k; k=0..m}
pub fn monotonic_divisions<F, Ft: Float + Display>(f: F, m: u32) -> Vec<Ft>
where F : Fn(Ft)->Ft + Copy{
    let m = m as usize;
    let parts : Vec<Ft> = (1..m).map(|i| Ft::from((i as f64) / (m as f64)).unwrap()).collect();
    let mut x_parts = Vec::with_capacity((m+1) as usize);
    x_parts.resize((m+1) as usize, Ft::zero());
    *x_parts.last_mut().unwrap() = Ft::one();

    bisect_array(f, &mut x_parts[1..m], &parts, Ft::zero(), Ft::one() );

    return x_parts;
}