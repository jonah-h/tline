#![allow(clippy::reversed_empty_ranges)]

use std::cmp::min;
use std::path::Path;

use crate::{Error, Solver, ComputeDescriptor};

/// Simulation specific parameters.
#[derive(Copy, Clone)]
pub struct SimulationParameters {
    /// The physical size of each spacial step along the transmission line.
    pub delta_z: f32,
    /// The length of each temperal step in the simulation.
    pub delta_t: f32,
}

/// Describes the  transmission line state at the current time step.
pub struct SimulationState {
    /// The time of the last time step of the simulation.
    pub time: f32,
    /// The voltages of each point along the transmission line at `time`.
    pub voltages: ndarray::Array1<f32>,
    /// The currents of each point along the transmission line at `time`.
    pub currents: ndarray::Array1<f32>,
}

/// Describes a simulation.
pub struct SimulationDescriptor<S: Solver> {
    /// The `Solver` for the simulation.
    pub solver: S,
    /// The parameters for the simulation.
    pub sim_params: SimulationParameters,
    /// The state that the simulation starts in.
    pub init_state: Option<SimulationState>,
}

/// Describes a simulation run.
pub struct RunDescriptor<P: AsRef<Path>> {
    /// How long, in temperal units, the simulation should run.
    pub time_duration: f32,
    /// Whether or not to print information to the console.
    pub verbose: bool,
    /// What, if any, information to save to file.
    pub save_settings: Option<SaveSettings<P>>,
}

/// How data should be saved to file.
#[derive(Debug)]
pub struct SaveSettings<P: AsRef<Path>> {
    /// The path to the save file.
    pub filename: P,
    /// What information to save.
    pub save_type: SaveType,
    /// Whether or not to overwrite any possible saved data.
    pub overwrite: bool,
}

/// Represents what data to save.
#[derive(PartialEq, Debug)]
pub enum SaveType {
    /// Save voltage and current data for every point on the line.
    Full,
    /// Save voltage and current data for only the end points.
    End,
}

/// The main `struct` of the framework.
pub struct Simulation<S: Solver> {
    solver: S,
    sim_params: SimulationParameters,
    state: SimulationState,
}

impl<S: Solver> Simulation<S> {
    /// Creates a new `Simulation` instance.
    #[inline]
    pub fn new(desc: SimulationDescriptor<S>) -> Result<Self, Error> {
        let total_points: usize = 1 + desc.solver.npoints();

        // create arrays for initial data
        let state = desc.init_state.unwrap_or(SimulationState {
            time: 0.0,
            voltages: ndarray::Array1::<f32>::zeros(total_points + 1),
            currents: ndarray::Array1::<f32>::zeros(total_points),
        });
        if state.voltages.len() != (total_points + 1) {
            return Err(Error::BadInit {
                array_name: "Voltage".to_string(),
                input_length: state.voltages.len(),
                expected_length: total_points + 1,
            })
        }
        if state.currents.len() != total_points {
            return Err(Error::BadInit {
                array_name: "Current".to_string(),
                input_length: state.voltages.len(),
                expected_length: total_points,
            })
        }

        Ok(Self {
            state,
            solver: desc.solver,
            sim_params: desc.sim_params,
        })
    }

    /// Does a computational run.
    #[inline]
    pub fn run<P: AsRef<Path>>(
        &mut self,
        desc: RunDescriptor<P>,
    ) -> Result<(), Error> {
        let nsteps = (desc.time_duration / self.sim_params.delta_t).ceil() as usize;
        let total_points: usize = 1 + self.solver.npoints();
        let store_size = min(nsteps + 1, (100_000_000 / total_points) + 1);
        let mut full_offset = 0;
        let mut end_offset = 0;

        // optionally create file
        if let Some(SaveSettings {
            ref filename,
            ref save_type,
            overwrite,
        }) = desc.save_settings {
            let filename = filename.as_ref();
            if filename.exists() && !overwrite {
                let file = hdf5::File::append(filename)?;

                let previous_end_size = file.dataset("end/voltages")?.shape()[0];
                end_offset = previous_end_size;

                // resize end datasets
                file.dataset("end/voltages")?.resize(previous_end_size + nsteps)?;
                file.dataset("end/currents")?.resize(previous_end_size + nsteps)?;
                file.dataset("start/voltages")?.resize(previous_end_size + nsteps)?;
                file.dataset("start/currents")?.resize(previous_end_size + nsteps)?;

                if *save_type == SaveType::Full {
                    if let Ok(full_group) = file.group("full") {
                        let previous_full_size = file.dataset("full/voltages")?.shape()[0];
                        full_offset = previous_full_size;
                        // resize full datasets
                        full_group.dataset("voltages")?.resize(
                            (previous_full_size + nsteps, total_points + 1)
                        )?;
                        full_group.dataset("currents")?.resize(
                            (previous_full_size + nsteps, total_points)
                        )?;
                    } else {
                        // create full datasets
                        let full_group = file.create_group("full")?;
                        full_group.new_dataset::<f32>()
                            .shape((hdf5::Extent::resizable(nsteps), total_points + 1))
                            .create("voltages")?;
                        full_group.new_dataset::<f32>()
                            .shape((hdf5::Extent::resizable(nsteps), total_points))
                            .create("currents")?;
                    }
                }

                file.close()?;
            } else {
                let file = hdf5::File::create(filename)?;

                // create end datasets
                let end_group = file.create_group("end")?;
                end_group.new_dataset::<f32>()
                    .shape(hdf5::Extent::resizable(nsteps))
                    .create("voltages")?;
                end_group.new_dataset::<f32>()
                    .shape(hdf5::Extent::resizable(nsteps))
                    .create("currents")?;
                let start_group = file.create_group("start")?;
                start_group.new_dataset::<f32>()
                    .shape(hdf5::Extent::resizable(nsteps))
                    .create("voltages")?;
                start_group.new_dataset::<f32>()
                    .shape(hdf5::Extent::resizable(nsteps))
                    .create("currents")?;

                if *save_type == SaveType::Full {
                    // create full datasets
                    let full_group = file.create_group("full")?;
                    full_group.new_dataset::<f32>()
                        .shape((hdf5::Extent::resizable(nsteps), total_points + 1))
                        .create("voltages")?;
                    full_group.new_dataset::<f32>()
                        .shape((hdf5::Extent::resizable(nsteps), total_points))
                        .create("currents")?;
                }

                // save deltas as file attributes
                let dt_attr = file.new_attr::<f32>()
                    .shape(hdf5::Extents::Scalar)
                    .create("time_step");
                if let Ok(attr) = dt_attr {
                    attr.write_scalar(&self.sim_params.delta_t)?;
                }
                let dz_attr = file.new_attr::<f32>()
                    .shape(hdf5::Extents::Scalar)
                    .create("length_step");
                if let Ok(attr) = dz_attr {
                    attr.write_scalar(&self.sim_params.delta_z)?;
                }

                file.close()?;
            }
        }

        // setup output if verbose
        let bar = if desc.verbose {
            println!("# of time steps: {}", nsteps);
            Some(indicatif::ProgressBar::new(nsteps as u64))
        } else {
            None
        };

        // separate calculations into sets of time steps per loop
        let nloops = ((nsteps-1) / (store_size-1)) + 1;
        for i in 0..nloops {
            let start_index = (store_size-1) * i;
            let end_index = min((store_size-1)*(i+1), nsteps);
            let niters = end_index - start_index;

            // do calculations
            let (voltages, currents) = self.solver.compute(ComputeDescriptor {
                state: &self.state,
                sim_params: self.sim_params,
                nsteps: niters,
                bar: &bar,
            })?;

            // optionally write full data to file
            if let Some(SaveSettings {
                ref filename,
                ref save_type,
                ..
            }) = desc.save_settings {
                let file = hdf5::File::open_rw(filename)?;

                // save end data
                file.dataset("end/voltages")?
                    .write_slice(
                        voltages.slice(ndarray::s![1..=niters, -1]).to_owned().view(),
                        ndarray::s![(start_index+end_offset)..(end_index+end_offset)],
                    )?;
                file.dataset("end/currents")?
                    .write_slice(
                        currents.slice(ndarray::s![1..=niters, -1]).to_owned().view(),
                        ndarray::s![(start_index+end_offset)..(end_index+end_offset)],
                    )?;
                file.dataset("start/voltages")?
                    .write_slice(
                        voltages.slice(ndarray::s![1..=niters, 0]).to_owned().view(),
                        ndarray::s![(start_index+end_offset)..(end_index+end_offset)],
                    )?;
                file.dataset("start/currents")?
                    .write_slice(
                        currents.slice(ndarray::s![1..=niters, 0]).to_owned().view(),
                        ndarray::s![(start_index+end_offset)..(end_index+end_offset)],
                    )?;

                // optionally save full data
                if *save_type == SaveType::Full {
                    // save full data
                    file.dataset("full/voltages")?
                        .write_slice(
                            voltages.slice(ndarray::s![1..=niters, ..]),
                            ndarray::s![(start_index+full_offset)..(end_index+full_offset), ..],
                        )?;
                    file.dataset("full/currents")?
                        .write_slice(
                            currents.slice(ndarray::s![1..=niters, ..]),
                            ndarray::s![(start_index+full_offset)..(end_index+full_offset), ..],
                        )?;
                }

                file.close()?;
            }

            // update state
            self.state.voltages.assign(&voltages.row(niters));
            self.state.currents.assign(&currents.row(niters));
            self.state.time += (niters as f32)*self.sim_params.delta_t;
        }

        if let Some(ref bar) = bar {
            bar.finish();
        }

        Ok(())
    }
}
