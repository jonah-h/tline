pub mod components;

mod fdtd_solver;

pub use fdtd_solver::{FdtdSolver, FdtdSolverDescriptor};

use crate::SimulationParameters;

/// Describes the behavior of the main simulated line.
pub trait TransmissionLine: Component {
    fn npoints(&self) -> usize;
    fn length(&self) -> f32;
    fn max_phase_velocity(&self) -> f32;
    fn calculate_simulation_parameters(&self, courant: f32) -> SimulationParameters {
        let delta_z = self.length() / (self.npoints() as f32);
        let delta_t = delta_z / (courant * self.max_phase_velocity());

        SimulationParameters { delta_z, delta_t }
    }
}

/// Defines the voltage and current response of a circuit element.
pub trait Component {
    fn next_voltage(
        &self,
        next_volt: &mut f32,
        last_volt: f32,
        last_currs: ndarray::ArrayView1<f32>,
        index: usize,
        sim_info: &SimulationParameters,
    );

    fn next_current(
        &self,
        next_curr: &mut f32,
        last_volts: ndarray::ArrayView1<f32>,
        last_curr: f32,
        index: usize,
        sim_info: &SimulationParameters,
    );
}

/// Generates a voltage output at the start of a transmission line.
pub trait VSource {
    fn next_voltage(
        &self,
        t: f32,
        last_volt: f32,
        last_curr: f32,
        sim_params: &SimulationParameters,
    ) -> f32;
    fn generate(&self, time: f32) -> f32;
}

/// Handles end of line boundary conditions, representing a physical terminator.
pub trait Terminator {
    fn next_voltage(
        &self,
        last_volt: f32,
        last_curr: f32,
        sim_params: &SimulationParameters,
    ) -> f32;
    fn next_current(
        &self,
        last_volts: ndarray::ArrayView1<f32>,
        last_curr: f32,
        sim_params: &SimulationParameters,
    ) -> f32;
}
