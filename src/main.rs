#![feature(test)]
extern crate test;
mod benchmarks;
mod control_params;
mod evaluation;
mod swarm;
extern crate rand;
use rayon::prelude::*;
use crate::benchmarks::Benchmark;
use crate::control_params::ControlParams;
use crate::swarm::Swarm;
use chrono::{Datelike, Utc};
use chrono;
use indicatif::{ProgressBar, ProgressStyle};
use indicatif::ParallelProgressIterator;
use rand::{prelude::SliceRandom, thread_rng};
use std::thread;

fn main() {
    let mut rng = thread_rng();
    let mut benchmarks = benchmarks::get_benchmarks();

    let num_particles = 30;
    let num_dimensions = 30;
    let num_iterations = 4000;
    let num_repetitions = 30;

    if true {
        println!("Starting find_optimal_cps");
        benchmarks.shuffle(&mut rng);
        find_optimal_cps(
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
    if false {
        println!("Starting eval_et_pso");
        benchmarks.shuffle(&mut rng);
        eval_et_pso(
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

fn _print_details(
    _run_idx: usize,
    _num_runs: usize,
    _max_stagnent_iters: usize,
    _dist: f64,
    _benchmark_name: &str,
    _cp: &ControlParams,
) {
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
    //         let mut swarm = Swarm::new(num_particles, num_dimensions, &cp, &benchmark, 0, 0.0);
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
            pbar.set_prefix(format!("[{rep_num}/{num_repetitions}] [{b_idx}/{}] {}", benchmarks.len(), benchmark.name));
            cps.par_iter().progress_with(pbar).for_each(move |cp| {
                let mut swarm = Swarm::new(
                    num_particles,
                    num_dimensions,
                    &cp,
                    &bm.clone(),
                    0,
                    0.0,
                    Strategy::None,
                );
                swarm.evaluate(&bm, 0, false);
                swarm.solve(&bm, num_iterations as i32, false);

                let benchmark_name = bm.name.clone();
                let fname = fname.clone();
                thread::spawn(move || {
                    swarm.log_to(
                        &fname,
                        &benchmark_name,
                        rep_num,
                        0.0,
                        0,
                    );
                });
            });
            pbar_progress += cps.len() as u64;
            println!("{} [rep {}/{}] [benchmark {}/{}] ({})", chrono::offset::Local::now(), rep_num, num_repetitions, b_idx, benchmarks.len(), benchmark.name);
        }
    }
}

fn eval_et_pso(
    num_particles: usize,
    num_dimensions: usize,
    num_repetitions: usize,
    num_iterations: usize,
    benchmarks: &Vec<Benchmark>,
) {
    let cp = ControlParams::generate_for_et_pso(0.12);
    let num_runs = benchmarks.len() * num_repetitions;
    let now = Utc::now();

    let pbar = ProgressBar::new(num_runs as u64);
    pbar.set_style(ProgressStyle::default_bar()
                    .template("[-{eta} +{elapsed} {prefix}] {bar:40.cyan/blue} {pos:>7}/{len:7} ({per_sec}) {msg}")
                    .progress_chars("=>~"));
    pbar.set_prefix(format!("{}p {}D", num_particles, num_dimensions));

    for rep_num in 0..num_repetitions {
        let filename = format!(
            "data/raw/et_pso_{}-{:02}-{:02}_rep_{}.csv",
            now.year(),
            now.month(),
            now.day(),
            rep_num
        );
        for benchmark in benchmarks {
            pbar.println(format!(
                "benchmark: {: <14}, w={:.2}, c1={:.2}, c2={:.2}",
                benchmark.name, cp.w, cp.c1, cp.c2,
            ));

            pbar.set_message("RAC");
            let mut swarm_rac = Swarm::new(
                num_particles,
                num_dimensions,
                &cp,
                &benchmark,
                1,
                0.0,
                Strategy::RandomAccelerationCoefficients,
            );
            swarm_rac.evaluate(&benchmark, 0, false);
            swarm_rac.solve(&benchmark, num_iterations as i32, false);
            swarm_rac.log_to(&filename, &benchmark.name, rep_num, 0.0, 1);

            pbar.set_message("ET(1.00)");
            let mut swarm_et = Swarm::new(
                num_particles,
                num_dimensions,
                &cp,
                &benchmark,
                3,
                0.0,
                Strategy::EmpiricallyTuned(1.0),
            );
            swarm_et.evaluate(&benchmark, 0, false);
            swarm_et.solve(&benchmark, num_iterations as i32, false);
            swarm_et.log_to(&filename, &benchmark.name, rep_num, 0.0, 3);

            pbar.set_message("ET(0.5)");
            let mut swarm_et = Swarm::new(
                num_particles,
                num_dimensions,
                &cp,
                &benchmark,
                3,
                0.0,
                Strategy::EmpiricallyTuned(0.5),
            );
            swarm_et.evaluate(&benchmark, 0, false);
            swarm_et.solve(&benchmark, num_iterations as i32, false);
            swarm_et.log_to(&filename, &benchmark.name, rep_num, 0.0, 3);

            pbar.set_message("ET(0.25)");
            let mut swarm_et = Swarm::new(
                num_particles,
                num_dimensions,
                &cp,
                &benchmark,
                3,
                0.0,
                Strategy::EmpiricallyTuned(0.25),
            );
            swarm_et.evaluate(&benchmark, 0, false);
            swarm_et.solve(&benchmark, num_iterations as i32, false);
            swarm_et.log_to(&filename, &benchmark.name, rep_num, 0.0, 3);

            pbar.set_message("ET(0.12)");
            let mut swarm_et = Swarm::new(
                num_particles,
                num_dimensions,
                &cp,
                &benchmark,
                3,
                0.0,
                Strategy::EmpiricallyTuned(0.12),
            );
            swarm_et.evaluate(&benchmark, 0, false);
            swarm_et.solve(&benchmark, num_iterations as i32, false);
            swarm_et.log_to(&filename, &benchmark.name, rep_num, 0.0, 3);

            pbar.inc(1);
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
    //                 0.0,
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
