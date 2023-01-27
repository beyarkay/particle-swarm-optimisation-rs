#![feature(test)]
extern crate test;
mod benchmarks;
mod control_params;
mod evaluation;
mod swarm;
extern crate rand;
use crate::benchmarks::Benchmark;
use crate::control_params::ControlParams;
use crate::swarm::Swarm;
use chrono;
use chrono::{Datelike, Utc};
use indicatif::ParallelProgressIterator;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{prelude::SliceRandom, thread_rng};
use rayon::prelude::*;
use std::thread;

fn main() {
    let mut rng = thread_rng();
    let mut benchmarks = benchmarks::get_benchmarks();

    let num_particles = 30;
    let num_dimensions = 30;
    let num_iterations = 4000;
    let num_repetitions = 30;

    // Run many PSO algorithms to see how the do across a wide range of control parameter
    // configurations.
    if false {
        println!("Starting find_optimal_cps");
        benchmarks.shuffle(&mut rng);
        find_optimal_cps(
            num_particles,
            num_dimensions,
            num_repetitions,
            num_iterations,
            &benchmarks,
        );
        println!("Now run the command `cat data/raw/opt_*_rep_*.csv | cut -d, -f5,6,7,8,9,10,11,13,14 | awk -F',' '$6 == \"3900\" {{ print $0 }}' > data/last_iter.csv` to extract only the important information from the experiment.");
    }

    // Train an ET-PSO on each of the benchmark function's data
    if true {
        println!("Starting train_bm_etpso");
        benchmarks.shuffle(&mut rng);
        train_bm_etpso(
            num_particles,
            num_dimensions,
            num_repetitions,
            num_iterations,
            &benchmarks,
        );
    }

    if false {
        println!("Starting eval_std_pso");
        benchmarks.shuffle(&mut rng);
        eval_std_pso(
            num_particles,
            num_dimensions,
            num_repetitions,
            num_iterations,
            &benchmarks,
        );
    }
    // if false {
    //     println!("Starting random_poli_sampling");
    //     benchmarks.shuffle(&mut rng);
    //     _random_poli_sampling( num_particles, num_dimensions, num_repetitions, num_iterations, &benchmarks,);
    // }
}

/// TODO
fn train_bm_etpso(
    num_particles: usize,
    num_dimensions: usize,
    num_repetitions: usize,
    num_iterations: usize,
    benchmarks: &Vec<Benchmark>,
) {
    let num_runs = num_repetitions * benchmarks.len() * benchmarks.len();
    let mut pbar_progress = 0;
    for rep_num in 0..num_repetitions {
        for (b_idx, init_bm) in benchmarks.iter().enumerate() {
            // Get the control parameters for this benchmark
            let cp_probs = ControlParams::generate_from_data("data/last_iter.csv", init_bm);
            let pbar = ProgressBar::new(num_runs as u64)
                .with_style(ProgressStyle::default_bar()
                            .template("[-{eta} +{elapsed} {prefix}] {bar:40.cyan/blue} {pos:>7}/{len:7} ({per_sec}) {msg}")
                            .progress_chars("=>~"))
                .with_prefix(format!("[{rep_num}/{num_repetitions}] [{b_idx}/{}] {}", benchmarks.len(), init_bm.name))
                .with_position(pbar_progress);
            pbar.reset_eta();

            benchmarks
                .par_iter()
                .progress_with(pbar)
                .for_each(move |eval_bm| {
                    let filename = format!("data/init-{}.csv", init_bm.name);
                    // 1. Initialise an ET-PSO
                    let mut swarm_et = Swarm::new(
                        num_particles,
                        num_dimensions,
                        &(cp_probs.clone()[0].0),
                        &eval_bm,
                        0,
                        Strategy::EmpiricallyTuned(1.0),
                    );
                    // Evaluate that ET-PSO on every benchmark function
                    swarm_et.evaluate(&eval_bm, 0, false);
                    swarm_et.solve(
                        &eval_bm,
                        num_iterations as i32,
                        Some(cp_probs.clone()),
                        false,
                    );
                    swarm_et.log_to(&filename, &eval_bm.name, rep_num, 0.0, 3);
                });
            pbar_progress += benchmarks.len() as u64;
            println!(
                "{} [rep {}/{}] [benchmark {}/{}] ({})",
                chrono::offset::Local::now(),
                rep_num,
                num_repetitions,
                b_idx,
                benchmarks.len(),
                init_bm.name
            );
        }
    }
}

/// Evaluate a PSO against w=0.7, c1=c2=1.4
fn eval_std_pso(
    _num_particles: usize,
    _num_dimensions: usize,
    num_repetitions: usize,
    _num_iterations: usize,
    benchmarks: &Vec<Benchmark>,
) {
    let mut _run_idx = 1;
    let _num_runs = benchmarks.len() * num_repetitions;
    let _cp = ControlParams::new(0.7, 1.4, 1.4);
    let now = Utc::now();
    let _filename = format!(
        "data/raw/std_pso_{}-{:02}-{:02}.csv",
        now.year(),
        now.month(),
        now.day()
    );

    // for benchmark in benchmarks {
    //     print_details(run_idx, num_runs, 0, 0.0, &benchmark.name, &cp);
    //     for rep_num in 0..num_repetitions {
    //         let mut swarm = Swarm::new(num_particles, num_dimensions, &cp, &benchmark, 0.0);
    //         swarm.evaluate(&benchmark, 0, false);
    //         swarm.solve(&benchmark, num_iterations as i32, false);
    //         swarm.log_to(&filename, &benchmark, rep_num, 0.0, 0);
    //         run_idx += 1;
    //     }
    // }
}

/// Go through every benchmark and then grid-search for the best control parameters
fn find_optimal_cps(
    num_particles: usize,
    num_dimensions: usize,
    num_repetitions: usize,
    num_iterations: usize,
    benchmarks: &Vec<Benchmark>,
) {
    let mut rng = thread_rng();
    let mut cps = ControlParams::generate_multiple_in_grid();
    cps.shuffle(&mut rng);
    let num_runs = cps.len() * benchmarks.len() * num_repetitions;
    let now = Utc::now();

    let mut pbar_progress = 0;
    for rep_num in 0..num_repetitions {
        let filename = format!(
            "data/raw/opt_{}-{:02}-{:02}_rep_{}.csv",
            now.year(),
            now.month(),
            now.day(),
            rep_num
        );

        for (b_idx, benchmark) in benchmarks.iter().enumerate() {
            let bm = benchmark.clone();
            let fname = filename.clone();
            // TODO this progress bar should continue from the last one
            let pbar = ProgressBar::new(num_runs as u64);
            pbar.set_position(pbar_progress);
            pbar.set_style(ProgressStyle::default_bar()
                            .template("[-{eta} +{elapsed} {prefix}] {bar:40.cyan/blue} {pos:>7}/{len:7} ({per_sec}) {msg}")
                            .progress_chars("=>~"));
            pbar.set_prefix(format!(
                "[{rep_num}/{num_repetitions}] [{b_idx}/{}] {}",
                benchmarks.len(),
                benchmark.name
            ));
            cps.par_iter().progress_with(pbar).for_each(move |cp| {
                let mut swarm = Swarm::new(
                    num_particles,
                    num_dimensions,
                    &cp,
                    &bm.clone(),
                    0,
                    Strategy::None,
                );
                swarm.evaluate(&bm, 0, false);
                swarm.solve(&bm, num_iterations as i32, None, false);

                let benchmark_name = bm.name.clone();
                let fname = fname.clone();
                thread::spawn(move || {
                    swarm.log_to(&fname, &benchmark_name, rep_num, 0.0, 0);
                });
            });
            pbar_progress += cps.len() as u64;
            println!(
                "{} [rep {}/{}] [benchmark {}/{}] ({})",
                chrono::offset::Local::now(),
                rep_num,
                num_repetitions,
                b_idx,
                benchmarks.len(),
                benchmark.name
            );
        }
    }
}

#[derive(Debug)]
pub enum Strategy {
    EmpiricallyTuned(f64),
    RandomAccelerationCoefficients,
    None,
}

/// Go through every benchmark and randomly resample a control parameter every iteration, or upon
/// stagnation
fn _random_poli_sampling(
    _num_particles: usize,
    _num_dimensions: usize,
    num_repetitions: usize,
    _num_iterations: usize,
    benchmarks: &Vec<Benchmark>,
) {
    let max_stagnent_iters_vec = vec![0, 1, 3, 9, 27, 81, 243, 729, 2187, 5000];
    let _cp = ControlParams::generate_by_poli();
    let _num_runs = benchmarks.len() * num_repetitions * max_stagnent_iters_vec.len();
    let now = Utc::now();
    let _filename = format!(
        "data/raw/resample_{}-{:02}-{:02}.csv",
        now.year(),
        now.month(),
        now.day()
    );

    // for benchmark in benchmarks {
    //     for max_stagnent_iters in &max_stagnent_iters_vec {
    //         print_details(
    //             run_idx,
    //             num_runs,
    //             *max_stagnent_iters,
    //             0.0,
    //             &benchmark.name,
    //             &cp,
    //         );
    //         for rep_num in 0..num_repetitions {
    //             let mut swarm = Swarm::new(
    //                 num_particles,
    //                 num_dimensions,
    //                 &cp,
    //                 &benchmark,
    //                 *max_stagnent_iters,
    //                 Strategy::RandomAccelerationCoefficients
    //             );
    //             swarm.evaluate(&benchmark, 0, false);
    //             swarm.solve(&benchmark, num_iterations as i32, false);
    //             swarm.log_to(&filename, &benchmark, rep_num, 0.0, *max_stagnent_iters);
    //             run_idx += 1;
    //         }
    //     }
    // }
}

struct _Particle {
    pbest: Vec<f64>,
    loc: Vec<f64>,
    vel: Vec<f64>,
    num_dims: usize,
    cp: ControlParams,
    stagnant_iters: usize,
}

#[derive(PartialEq, Debug)]
enum PositionRepair {
    /// Re-initializes any invalid position component uniformly within the search space.
    Random,
    /// Moves the particle back onto the boundary in every violated dimension, and velocity is set
    /// to zero.
    _Absorb,
    /// Do not re-evaluate infeasible particle’s fitness.
    Invisible,
    /// Move violated components to position between particle’s personal best position and
    /// violated boundary, using exponential distribution.
    _Exponential,
    /// Update personal best positions only when solution quality improves AND the particle is
    /// feasible (within bounds)
    _Infinity,
}
