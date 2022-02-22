use crate::{Error, Solver, ComputeDescriptor};
use crate::fdtd::{TransmissionLine, VSource, Terminator};

/// Describes the composition of a `StandardSolver`.
pub struct FdtdSolverDescriptor<L: TransmissionLine> {
    pub tline: L,
    pub source: Box<dyn VSource>,
    pub terminator: Box<dyn Terminator>,
}

/// Does single threaded computations on the CPU.
pub struct FdtdSolver<L: TransmissionLine> {
    tline: L,
    source: Box<dyn VSource>,
    terminator: Box<dyn Terminator>,
}

impl<L: TransmissionLine> FdtdSolver<L> {
    #[inline]
    pub fn new(desc: FdtdSolverDescriptor<L>) -> Self {
        Self {
            tline: desc.tline,
            source: desc.source,
            terminator: desc.terminator,
        }
    }
}

impl<L: TransmissionLine> Solver for FdtdSolver<L> {
    #[inline]
    fn compute(
        &mut self,
        desc: ComputeDescriptor,
    ) -> Result<(ndarray::Array2<f32>, ndarray::Array2<f32>), Error> {
        let total_points: usize = 1 + self.tline.npoints();

        // create storage arrays for voltage and current
        let mut voltages = ndarray::Array2::<f32>::zeros((desc.nsteps+1, total_points + 1));
        voltages.slice_mut(ndarray::s![0, ..]).assign(&desc.state.voltages);
        let mut currents = ndarray::Array2::<f32>::zeros((desc.nsteps+1, total_points));
        currents.slice_mut(ndarray::s![0, ..]).assign(&desc.state.currents);

        // loop through time
        for t_index in 0..desc.nsteps {
            let t = (t_index as f32)*desc.sim_params.delta_t + desc.state.time;

            // calculate first voltage from vsource
            voltages[[t_index+1, 0]] = self.source.next_voltage(
                t,
                voltages[[t_index, 0]],
                currents[[t_index, 0]],
                &desc.sim_params,
            );

            // get 1D views of voltages at relevent times
            let (volts1, mut volts2) = voltages
                .view_mut()
                .split_at(ndarray::Axis(0), t_index+1);
            let last_volts = volts1.row(t_index);
            let mut next_volts = volts2.row_mut(0);
            // get 1D views of currents at relevent times
            let (currs1, mut currs2) = currents
                .view_mut()
                .split_at(ndarray::Axis(0), t_index+1);
            let last_currs = currs1.row(t_index);
            let mut next_currs = currs2.row_mut(0);

            let npoints = self.tline.npoints();
            ndarray::Zip::from(&mut next_volts.slice_mut(ndarray::s![1..(1+npoints)]))
                .and(&last_volts.slice(ndarray::s![1..(1+npoints)]))
                .and(last_currs.slice(ndarray::s![0..(1+npoints)]).windows(2))
                .and(&(0..(npoints)).collect::<Vec<usize>>())
                .for_each(|nv, &lv, lc, &z| {
                    self.tline.next_voltage(nv, lv, lc, z, &desc.sim_params);
                });
            // calculate last voltage
            let last_ind = total_points;
            voltages[[t_index+1, last_ind]] = self.terminator.next_voltage(
                last_volts[last_ind],
                last_currs[last_ind-1],
                &desc.sim_params,
            );

            // calculate currents for next time step
            let last_volts = voltages.row(t_index+1);
            let npoints = self.tline.npoints();
            ndarray::Zip::from(&mut next_currs.slice_mut(ndarray::s![0..npoints]))
                .and(last_volts.slice(ndarray::s![0..(1+npoints)]).windows(2))
                .and(&last_currs.slice(ndarray::s![0..npoints]))
                .and(&(0..(npoints)).collect::<Vec<usize>>())
                .for_each(|nv, lv, &lc, &z| {
                    self.tline.next_current(nv, lv, lc, z, &desc.sim_params);
                });
            // calculate last current
            currents[[t_index+1, last_ind-1]] = self.terminator.next_current(
                last_volts.slice(ndarray::s![-2..=-1]),
                last_currs[last_ind-1],
                &desc.sim_params,
            );

            if let Some(ref bar) = desc.bar {
                bar.inc(1)
            }
        }

        Ok((voltages, currents))
    }

    fn npoints(&self) -> usize {
        self.tline.npoints()
    }
}
