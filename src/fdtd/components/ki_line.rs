use crate::SimulationParameters;
use crate::fdtd::{TransmissionLine, Component};

pub struct KiLineDescriptor<
    Fc: Fn(f32) -> f32, Fl: Fn(f32) -> f32,
    Fk: Fn(f32) -> f32, Fi: Fn(f32) -> f32,
>{
    pub length: f32,
    pub npoints: usize,
    pub capacitance_fn: Fc,
    pub inductance_fn: Fl,
    pub kinetic_inductance_fn: Fk,
    pub critical_current_fn: Fi,
}

pub struct KiLine {
    cap: Vec<f32>,
    ind0: Vec<f32>,
    crit_cur: Vec<f32>,
    npoints: usize,
    length: f32,
}
impl KiLine {
    #[inline]
    pub fn new<
        Fc: Fn(f32) -> f32, Fl: Fn(f32) -> f32,
        Fk: Fn(f32) -> f32, Fi: Fn(f32) -> f32,
    >(
        desc: KiLineDescriptor<Fc, Fl, Fk, Fi>,
    ) -> Self {
        let delta_z = desc.length / (desc.npoints as f32);

        Self {
            cap: (0..desc.npoints)
                .map(|n| { (desc.capacitance_fn)((n as f32 + 0.5) * delta_z) })
                .collect::<Vec<_>>(),
            ind0: (0..desc.npoints)
                .map(|n| {
                    let ki_ind = (desc.kinetic_inductance_fn)((n as f32 + 0.5) * delta_z);
                    let ind = (desc.inductance_fn)((n as f32 + 0.5) * delta_z);
                    ind + ki_ind
                })
                .collect::<Vec<_>>(),
            crit_cur: (0..desc.npoints)
                .map(|n| {
                    let ki_ind = (desc.kinetic_inductance_fn)((n as f32 + 0.5) * delta_z);
                    let ind = (desc.inductance_fn)((n as f32 + 0.5) * delta_z);
                    let crit_cur = (desc.critical_current_fn)((n as f32 + 0.5) * delta_z);
                    crit_cur * f32::sqrt((ind + ki_ind) / ki_ind)
                })
                .collect::<Vec<_>>(),
            npoints: desc.npoints,
            length: desc.length,
        }
    }
}
impl Component for KiLine {
    #[inline]
    fn next_voltage(
        &self,
        next_volt: &mut f32,
        last_volt: f32,
        last_currs: ndarray::ArrayView1<f32>,
        index: usize,
        sim_params: &SimulationParameters,
    ) {
        let d_ratio = sim_params.delta_z / sim_params.delta_t;

        *next_volt = (d_ratio*self.cap[index]).recip()
            * ( d_ratio*self.cap[index]*last_volt + (last_currs[0] - last_currs[1]) );
    }
    #[inline]
    fn next_current(
        &self,
        next_curr: &mut f32,
        last_volts: ndarray::ArrayView1<f32>,
        last_curr: f32,
        index: usize,
        sim_params: &SimulationParameters,
    ) {
        let ind = self.ind0[index];
        let i_crit = self.crit_cur[index];
        let delta_z = sim_params.delta_z;
        let delta_t = sim_params.delta_t;
        let dv = last_volts[1] - last_volts[0];

        let a = 1.0;
        let b = last_curr;
        let c = i_crit.powi(2) - last_curr.powi(2);
        let d = i_crit.powi(2) * delta_t * dv / (delta_z * ind)
            - i_crit.powi(2)*last_curr - last_curr.powi(3);

        let mut next_guess = last_curr;
        let mut this_guess;
        for _ in 0..3 {
            this_guess = next_guess;

            next_guess = this_guess
                - (a*this_guess.powi(3)+b*this_guess.powi(2)+c*this_guess+d)
                / (3.0*a*this_guess.powi(2)+2.0*b*this_guess+c);
        }

        *next_curr = next_guess;
    }
}
impl TransmissionLine for KiLine {
    #[inline]
    fn npoints(&self) -> usize {
        self.npoints
    }
    #[inline]
    fn length(&self) -> f32 {
        self.length
    }
    #[inline]
    fn max_phase_velocity(&self) -> f32 {
        self.ind0.iter().zip(self.cap.iter())
            .map(|(ind0, cap)| f32::sqrt(ind0 * cap).recip())
            .reduce(|accum, item| if accum >= item { accum } else { item })
            .unwrap()
    }
}
