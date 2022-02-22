use tline::prelude::*;
use tline::fdtd::*;

use std::f32::consts::PI;

fn main() {
    let capacitance = 400e-12; // [F / m]
    let inductance = 1e-6; // [H / m]
    let resistance = 0.0; // [Ω / m]
    let conductance = 0.0; // [S / m]

    let npoints = 10_000;

    // create a simple lossless transmission line
    let tline = components::LinearLine::new(components::LinearLineDescriptor {
        npoints,
        length: 2.0, // [m]
        capacitance_fn: |_| capacitance,
        inductance_fn: |_| inductance,
        resistance_fn: |_| resistance,
        conductance_fn: |_| conductance,
    });

    let sim_params = tline.calculate_simulation_parameters(2.0);

    let mut simulation = Simulation::new(SimulationDescriptor {
        solver: FdtdSolver::new(FdtdSolverDescriptor {
            tline,
            source: Box::new(components::MatchedVSource {
                source_fn: |t| {
                    f32::sin(2.0*PI *4e8 * t)
                },
                inductance,
                capacitance,
                resistance,
                conductance,
            }),
            terminator: Box::new(components::MatchedTerminator {
                inductance,
                capacitance,
                resistance,
                conductance,
            }),
        }),
        sim_params,
        init_state: None,
    }).unwrap();

    println!(
        "\n-- General Simulation Info --\n\
        # of points:  {}\n\
        Δz:           {:<9.2e} m\n\
        Δt:           {:<9.2e} s\n",
        npoints,
        sim_params.delta_z,
        sim_params.delta_t,
    );

    println!("-- Run Part 1 --");
    // get to a steady state and save end data
    simulation.run(RunDescriptor {
        time_duration: 1e-7, // [s]
        verbose: true,
        save_settings: Some(SaveSettings {
            filename: "data/simple_tline.h5",
            save_type: SaveType::End,
            overwrite: true,
        }),
    })
    .unwrap();

    println!("-- Run Part 2 --");
    // save full data at steady state
    simulation.run(RunDescriptor {
        time_duration: 1e-7,
        verbose: true,
        save_settings: Some(SaveSettings {
            filename: "data/simple_tline.h5",
            save_type: SaveType::Full,
            overwrite: false,
        }),
    })
    .unwrap();
}
