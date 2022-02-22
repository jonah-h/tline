use crate::SimulationParameters;
use crate::fdtd::VSource;

/// A simple voltage source.
pub struct MatchedVSource<Fs> where Fs: Fn(f32)->f32 {
    pub source_fn: Fs,
    pub capacitance: f32,
    pub inductance: f32,
    pub resistance: f32,
    pub conductance: f32,
}
impl<Fs> VSource for MatchedVSource<Fs> where Fs: Fn(f32)->f32 {
    fn next_voltage(
        &self,
        t: f32,
        last_volt: f32,
        last_curr: f32,
        sim_params: &SimulationParameters,
    ) -> f32 {
        // calculate first voltage from vsource
        let impedance = f32::sqrt(self.inductance / self.capacitance);
        let total_resistance = sim_params.delta_z*self.resistance + impedance;
        let d_ratio = sim_params.delta_z / sim_params.delta_t;

        let last_source_curr = (d_ratio*self.inductance + total_resistance/2.0).recip()
            *  ( (d_ratio*self.inductance - total_resistance/2.0) * last_curr
                + (self.generate(t) - last_volt) );

        (d_ratio*self.capacitance + sim_params.delta_z*self.conductance/2.0).recip()
            * ( (d_ratio*self.capacitance - sim_params.delta_z*self.conductance/2.0) * last_volt
                + (last_source_curr - last_curr) )
    }

    fn generate(&self, time: f32) -> f32 {
        (self.source_fn)(time)
    }
}
