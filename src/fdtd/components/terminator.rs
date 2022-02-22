use crate::SimulationParameters;
use crate::fdtd::Terminator;

pub struct MatchedTerminator {
    pub inductance: f32,
    pub capacitance: f32,
    pub resistance: f32,
    pub conductance: f32,
}
impl Terminator for MatchedTerminator {
    fn next_voltage(
        &self,
        last_volt: f32,
        last_curr: f32,
        sim_params: &SimulationParameters,
    ) -> f32 {
        let load_conductance = f32::sqrt(self.capacitance / self.inductance);
        let total_conductance = sim_params.delta_z*self.conductance + load_conductance;
        let d_ratio = sim_params.delta_z / sim_params.delta_t;

        (d_ratio*self.capacitance + total_conductance/2.0).recip()
            * ( (d_ratio*self.capacitance - total_conductance/2.0) * last_volt
                + last_curr )
    }

    fn next_current(
        &self,
        last_volts: ndarray::ArrayView1<f32>,
        last_curr: f32,
        sim_params: &SimulationParameters,
    ) -> f32 {
        let d_ratio = sim_params.delta_z / sim_params.delta_t;

        (d_ratio*self.inductance + sim_params.delta_z*self.resistance/2.0).recip()
            *  ( (d_ratio*self.inductance - sim_params.delta_z*self.resistance/2.0) * last_curr
                + (last_volts[0] - last_volts[1]) )
    }
}
