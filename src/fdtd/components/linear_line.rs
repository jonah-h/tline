use crate::SimulationParameters;
use crate::fdtd::{TransmissionLine, Component};

pub struct LinearLineDescriptor<
    Fc: Fn(f32) -> f32, Fl: Fn(f32) -> f32,
    Fr: Fn(f32) -> f32, Fg: Fn(f32) -> f32,
>{
    pub length: f32,
    pub npoints: usize,
    pub capacitance_fn: Fc,
    pub inductance_fn: Fl,
    pub resistance_fn: Fr,
    pub conductance_fn: Fg,
}

pub struct LinearLine {
    cap: Vec<f32>,
    ind: Vec<f32>,
    res: Vec<f32>,
    cond: Vec<f32>,
    npoints: usize,
    length: f32,
}
impl LinearLine {
    pub fn new<
        Fc: Fn(f32) -> f32, Fl: Fn(f32) -> f32,
        Fr: Fn(f32) -> f32, Fg: Fn(f32) -> f32,
    >(
        desc: LinearLineDescriptor<Fc, Fl, Fr, Fg>,
    ) -> Self {
        let delta_z = desc.length / (desc.npoints as f32);

        Self {
            cap: (0..desc.npoints)
                .map(|n| { (desc.capacitance_fn)((n as f32 + 0.5) * delta_z) })
                .collect::<Vec<_>>(),
            ind: (0..desc.npoints)
                .map(|n| { (desc.inductance_fn)((n as f32 + 0.5) * delta_z) })
                .collect::<Vec<_>>(),
            res: (0..desc.npoints)
                .map(|n| { (desc.resistance_fn)((n as f32 + 0.5) * delta_z) })
                .collect::<Vec<_>>(),
            cond: (0..desc.npoints)
                .map(|n| { (desc.conductance_fn)((n as f32 + 0.5) * delta_z) })
                .collect::<Vec<_>>(),
            npoints: desc.npoints,
            length: desc.length,
        }
    }
}
impl Component for LinearLine {
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

        *next_volt = (d_ratio*self.cap[index] + sim_params.delta_z*self.cond[index]/2.0).recip()
            * ( (d_ratio*self.cap[index] - sim_params.delta_z*self.cond[index]/2.0) * last_volt
                + (last_currs[0] - last_currs[1]) );
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
        let d_ratio = sim_params.delta_z / sim_params.delta_t;

        *next_curr = (d_ratio*self.ind[index] + sim_params.delta_z*self.res[index]/2.0).recip()
            *  ( (d_ratio*self.ind[index] - sim_params.delta_z*self.res[index]/2.0) * last_curr
                + (last_volts[0] - last_volts[1]) );
    }
}
impl TransmissionLine for LinearLine {
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
        self.ind.iter().zip(self.cap.iter())
            .map(|(ind, cap)| f32::sqrt(ind * cap).recip())
            .reduce(|accum, item| if accum >= item { accum } else { item })
            .unwrap()
    }
}
