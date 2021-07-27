
/// Evaluate the finite differences derivatives at each point (x, y)
/// where the x grid may be irregular
pub fn finite_differences(x: &[f64], y: &[f64]) -> Vec<f64>
{
    let n = x.len();
    let delta_x: Vec<f64> = x.iter().zip(x.iter().skip(1))
        .map(|(&x1, &x2)| x2 - x1).collect();
    let delta_y: Vec<f64> = y.iter().zip(y.iter().skip(1))
        .map(|(&y1, &y2)| y2 - y1).collect();
    let mut deriv = Vec::with_capacity(n);
    deriv.resize(n, 0.0);
    deriv[0] = delta_y[0] / delta_x[0];
    *deriv.last_mut().unwrap() = delta_y.last().unwrap()/ delta_x.last().unwrap();
    for i in 1..n-1{
        deriv[i] = &delta_y[i] / delta_x[i-1] + &delta_y[i-1]/delta_x[i];
    }
    return deriv;
}

pub struct StepwiseMeasure{
    step_pnts: Vec<f64>
}
impl StepwiseMeasure{
    pub fn new(step_pnts: Vec<f64>){
        let n = step_pnts.len();
        let delta_x : Vec<f64> = step_pnts.iter().zip(step_pnts.iter().skip(1))
            .map(|(&x1, &x2)| x2 - x1).collect();
        let mut weights = Vec::with_capacity(n);
        weights.resize(n, 0.0);
        weights[0] = delta_x[0]/2.0;
        for i in 1..n-1{
            weights[i] = (delta_x[i-1] + delta_x[i])/2.0;
        }
        weights[n-1] = delta_x[n-2]/2.0
    }
}

pub enum BisResult{
    Root((f64, f64)),
    LoBnd,
    UpBnd,
    Failed
}

impl BisResult{
    pub fn unwrap_root(&self) -> (f64, f64){
        if let &Self::Root((x,y)) = self{
            return (x,y);
        } else{
            panic!("BisResult::unwrap_root failed");
        }
    }
}

pub fn monotonic_bisection<F: Fn(f64)->f64>(
    f: F, y0: f64, xmin: f64, xmax: f64, tol: f64, iters_max: u32,
            )
-> BisResult
{
    assert!(xmin < xmax);

    let mut x0 = xmin;
    let mut x1 = xmax;
    for _ in 0..iters_max {
        let xmid = (x0 + x1)/2.0;
        let y = f(xmid);
        if (xmid - xmin).abs() < tol{
            return BisResult::LoBnd
        }
        if (xmid - xmax).abs() < tol{
            return BisResult::UpBnd
        }
        if (y-y0).abs() < tol{
            return BisResult::Root((xmid, y))
        }
        if y < y0{
            x1 = xmid
        } else {
            x0 = xmid;
        }
    }

    return BisResult::Failed;
}

/// roots n
/// ytgt n
fn bisect_array<F: Fn(f64)->f64 + Copy>(f: F, roots: &mut [f64], ytgt: &[f64],
                                 xmin: f64, xmax: f64){
    // Base cases
    if roots.len() == 0{
        return;
    } else if roots.len() == 1{
        let y0 = ytgt[0];
        let (x,y) = monotonic_bisection(f, y0, xmin, xmax,  1.0e-4, 10000).unwrap_root();
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
/// into intervals [x_k, x_{k+1}] such that f([x_k, x_{k+1}]) = [k/m, (k+1)/m]
pub fn monotonic_divisions<F>(f: F, m: u32) -> Vec<f64>
where F : Fn(f64)->f64 + Copy{
    let m = m as usize;
    let parts : Vec<f64> = (1..m).map(|i| (i as f64) / (m as f64)).collect();
    let mut x_parts = Vec::with_capacity((m+1) as usize);
    x_parts.resize((m+1) as usize, 0.0);
    *x_parts.last_mut().unwrap() = 1.0;

    bisect_array(f, &mut x_parts[1..m], &parts, 0.0, 1.0 );

    return x_parts;
}