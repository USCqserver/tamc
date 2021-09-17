use std::vec::Vec;
use std::collections::BTreeMap;
use num_traits::FromPrimitive;
use num_traits::ToPrimitive;
use std::convert::TryFrom;
use std::fmt::Debug;

// wow
// so cheap
// much csr

pub struct CheapCSR<N, R>{
    row_idxs: Vec<N>,
    row_entries: Vec<R>
}

impl<N: Copy+FromPrimitive+ToPrimitive, R: Clone> CheapCSR<N, R>{
    pub fn from_adj_list(adj_vecs: &Vec<Vec<R>>) -> Self
    {
        let mut row_idxs = Vec::with_capacity(adj_vecs.len() + 1);
        let mut row_entries = Vec::new();
        row_idxs.push(N::from_usize(0).unwrap());
        let mut c = 0;
        for  e in adj_vecs.iter(){
            let nz = e.len();
            c += nz;
            row_idxs.push(N::from_usize(c).unwrap());
            for r in e.iter(){
                row_entries.push(r.clone())
            }
        }
        row_entries.shrink_to_fit();
        return Self{ row_idxs, row_entries };
    }

    pub unsafe fn uget_rows(&self, i :usize) -> &[R]{
        let j_beg = self.row_idxs.get_unchecked(i).to_usize().unwrap();
        let j_end =  self.row_idxs.get_unchecked(i+1).to_usize().unwrap();
        let row_slice = self.row_entries.get_unchecked(j_beg..j_end);
        return row_slice;
    }

    pub fn iter(&self) -> CheapCsrIter<N, R>{
        return CheapCsrIter::new(&self.row_idxs, &self.row_entries);
    }
}

pub struct CheapCsrIter<'a, N, R>{
    row_idxs: &'a [N],
    row_entries: &'a [R],
    n_rows: usize,
    current_row: usize,
    current_col_idx: usize,
    current_col_range: (usize, usize)
}

impl<'a, N: Copy+ToPrimitive, R: 'a> CheapCsrIter<'a, N, R>{
    fn new(row_idxs: &'a [N],
           row_entries: &'a [R],) -> Self{
        let n_rows = row_idxs.len() - 1;
        let current_row = 0;
        let current_col_idx = 0;
        let current_col_range = (row_idxs[0].to_usize().unwrap(),
                                 row_idxs[1].to_usize().unwrap());
        return Self{
            row_idxs, row_entries, n_rows,
            current_row, current_col_idx, current_col_range
        };
    }
    fn next_row(&mut self) -> Option<()>{
        self.current_row += 1;
        if self.current_row >= self.n_rows{
            return None;
        }
        let idx = self.current_row.to_usize().unwrap();
        unsafe {
            self.current_col_range = (self.row_idxs.get_unchecked(idx).to_usize().unwrap(),
                                      self.row_idxs.get_unchecked(idx + 1).to_usize().unwrap());
        }
        self.current_col_idx = self.current_col_range.0;
        return Some(());
    }
}

impl<'a, N: Copy+ToPrimitive, R: 'a> Iterator for CheapCsrIter<'a, N, R>{
    type Item = (usize, &'a R);

    fn next(&mut self) -> Option<Self::Item> {
        loop{ // Continue after end of previous row (also skips empty rows)
            if self.current_row >= self.n_rows{
                return None;
            }
            if self.current_col_idx == self.current_col_range.1{
                self.next_row()?;
            } else {
                break;
            }
        }
        let i = self.current_row;
        let r = unsafe{
            self.row_entries.get_unchecked(self.current_col_idx)
        };

        self.current_col_idx += 1;
        return Some((i, r));
    }
}

#[cfg(test)]
mod tests{
    use std::collections::BTreeMap;
    use std::iter::FromIterator;
    use crate::csr::CheapCSR;

    #[test]
    fn test_csr_mat(){
        // construct a 7 x 7 sparse matrix
        let adj_list : Vec<Vec<(u32, f32)>> = vec![
            Vec::new(),
            vec![(1, 1.0)],
            Vec::new(),
            Vec::new(),
            vec![(2, 2.0), (3, 3.0)],
            vec![(0, 4.0)],
            vec![(6, 5.0)]
        ];
        let csr = CheapCSR::<u32, (u32, f32)>::from_adj_list(&adj_list);

        for (i, &(j, r)) in csr.iter(){
            println!("({}, {}): {:3.2}", i, j, r);
        }
    }
}