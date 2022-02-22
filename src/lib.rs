//! A framework for simulating 1-dimensional linear and nonlinear transmission lines.
//!
//! To get started, refer to the `\examples` directory in the main repository.

mod simulation;

pub mod fdtd;
pub mod prelude;

pub use simulation::{
    RunDescriptor, SaveSettings, SaveType, Simulation, SimulationDescriptor, SimulationParameters,
    SimulationState,
};

/// Represents an error in the simulation.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Init {array_name} array does not have expected length \
        ( {array_name} array length: {input_length}, \
        expected length: {expected_length} )")]
    BadInit {
        array_name: String,
        input_length: usize,
        expected_length: usize,
    },
    #[error("There was an error during computation")]
    ComputationError(i32),
    #[error(transparent)]
    H5Error(#[from] hdf5::Error),
}

/// Manages actual computations.
pub trait Solver {
    /// Generates voltage and current data for a set of times.
    fn compute(
        &mut self,
        desc: ComputeDescriptor,
    ) -> Result<(ndarray::Array2<f32>, ndarray::Array2<f32>), Error>;

    fn npoints(&self) -> usize;
}

/// Describes how a `StandardSolver` should do computations.
pub struct ComputeDescriptor<'a> {
    pub state: &'a SimulationState,
    pub sim_params: SimulationParameters,
    pub nsteps: usize,
    pub bar: &'a Option<indicatif::ProgressBar>,
}
