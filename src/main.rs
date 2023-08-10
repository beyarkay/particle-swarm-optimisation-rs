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
use std::fs::OpenOptions;
use std::io::{prelude::*, BufWriter};
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
    if false {
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

    if false {
        println!("Starting eval_rac_pso");
        benchmarks.shuffle(&mut rng);
        eval_rac_pso(
            num_particles,
            num_dimensions,
            num_repetitions,
            num_iterations,
            &benchmarks,
        );
    }

    if true {
        println!("Starting save_bm_values");
        benchmarks.shuffle(&mut rng);
        save_bm_values(
            num_particles,
            num_dimensions,
            num_repetitions,
            num_iterations,
            &benchmarks,
        );
    }
}

/// Initialise ET-PSO on each benchmark function, and then evaluate that ET-PSO on every benchmark
/// function. This will assess how well ET-PSO generalizes to unseen benchmarks.
fn train_bm_etpso(
    num_particles: usize,
    num_dimensions: usize,
    num_repetitions: usize,
    num_iterations: usize,
    benchmarks: &Vec<Benchmark>,
) {
    let num_runs = num_repetitions * benchmarks.len() * benchmarks.len();
    let pbar = ProgressBar::new(num_runs as u64)
        .with_style(ProgressStyle::default_bar()
                    .template("[-{eta} +{elapsed} {prefix}] {bar:40.cyan/blue} {pos:>7}/{len:7} ({per_sec}) {msg}")
                    .progress_chars("=>~"));
    for rep_num in 0..num_repetitions {
        for (b_idx, init_bm) in benchmarks.iter().enumerate() {
            pbar.set_prefix(format!(
                "[rep {rep_num}/{num_repetitions}] [benchmark {b_idx}/{}] ({})",
                benchmarks.len(),
                init_bm.name
            ));
            // Get the control parameters for this benchmark
            let cp_probs = ControlParams::generate_from_data("data/last_iter.csv", init_bm);
            benchmarks.par_iter().for_each(move |eval_bm| {
                let eval_name = eval_bm.name.replace(" ", "-").to_lowercase();
                let init_name = init_bm.name.replace(" ", "-").to_lowercase();
                let filename = format!("data/et-pso/init-{init_name}-eval-{eval_name}.csv");
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
            pbar.inc(benchmarks.len().try_into().unwrap());
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

fn eval_std_pso(
    num_particles: usize,
    num_dimensions: usize,
    num_repetitions: usize,
    num_iterations: usize,
    benchmarks: &Vec<Benchmark>,
) {
    let num_runs = num_repetitions * benchmarks.len();
    let pbar = ProgressBar::new(num_runs as u64)
        .with_style(ProgressStyle::default_bar()
                    .template("[-{eta} +{elapsed} {prefix}] {bar:40.cyan/blue} {pos:>7}/{len:7} ({per_sec}) {msg}")
                    .progress_chars("=>~"));
    let cp = ControlParams::new(1.4, 0.7, 0.7);
    for rep_num in 0..num_repetitions {
        pbar.set_prefix(format!("[rep {rep_num}/{num_repetitions}]"));
        // Get the control parameters for this benchmark
        benchmarks.par_iter().for_each(move |eval_bm| {
            let eval_name = eval_bm.name.replace(" ", "-").to_lowercase();
            let filename = format!("data/std-pso/eval-{eval_name}.csv");
            // 1. Initialise a std pso
            let mut swarm_et = Swarm::new(
                num_particles,
                num_dimensions,
                &(cp.clone()),
                &eval_bm,
                0,
                Strategy::None,
            );
            // Evaluate that ET-PSO on every benchmark function
            swarm_et.evaluate(&eval_bm, 0, false);
            swarm_et.solve(&eval_bm, num_iterations as i32, None, false);
            swarm_et.log_to(&filename, &eval_bm.name, rep_num, 0.0, 0);
        });
        pbar.inc(benchmarks.len().try_into().unwrap());
        println!(
            "{} [rep {}/{}]",
            chrono::offset::Local::now(),
            rep_num,
            num_repetitions
        );
    }
}

fn eval_rac_pso(
    num_particles: usize,
    num_dimensions: usize,
    num_repetitions: usize,
    num_iterations: usize,
    benchmarks: &Vec<Benchmark>,
) {
    let num_runs = num_repetitions * benchmarks.len();
    let pbar = ProgressBar::new(num_runs as u64)
        .with_style(ProgressStyle::default_bar()
                    .template("[-{eta} +{elapsed} {prefix}] {bar:40.cyan/blue} {pos:>7}/{len:7} ({per_sec}) {msg}")
                    .progress_chars("=>~"));
    let cp = ControlParams::new(1.4, 0.7, 0.7);
    for rep_num in 0..num_repetitions {
        pbar.set_prefix(format!("[rep {rep_num}/{num_repetitions}]"));
        // Get the control parameters for this benchmark
        benchmarks.par_iter().for_each(move |eval_bm| {
            let eval_name = eval_bm.name.replace(" ", "-").to_lowercase();
            let filename = format!("data/rac-pso/eval-{eval_name}.csv");
            // 1. Initialise a std pso
            let mut swarm_et = Swarm::new(
                num_particles,
                num_dimensions,
                &(cp.clone()),
                &eval_bm,
                1,
                Strategy::RandomAccelerationCoefficients,
            );
            // Evaluate that ET-PSO on every benchmark function
            swarm_et.evaluate(&eval_bm, 0, false);
            swarm_et.solve(&eval_bm, num_iterations as i32, None, false);
            swarm_et.log_to(&filename, &eval_bm.name, rep_num, 0.0, 1);
        });
        pbar.inc(benchmarks.len().try_into().unwrap());
        println!(
            "{} [rep {}/{}]",
            chrono::offset::Local::now(),
            rep_num,
            num_repetitions
        );
    }
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

/// Go through every benchmark and then grid-search for the best control parameters
fn save_bm_values(
    _num_particles: usize,
    _num_dimensions: usize,
    _num_repetitions: usize,
    _num_iterations: usize,
    benchmarks: &Vec<Benchmark>,
) {
    let mut rng = thread_rng();
    let mut cps = ControlParams::generate_multiple_in_grid();
    cps.shuffle(&mut rng);
    let now = Utc::now();
    let num_points = 500;

    let pbar = ProgressBar::new(benchmarks.len().try_into().unwrap())
        .with_style(ProgressStyle::default_bar()
                    .template("[-{eta} +{elapsed} {prefix}] {bar:40.cyan/blue} {pos:>7}/{len:7} ({per_sec}) {msg}")
                    .progress_chars("=>~"));
    benchmarks
        .par_iter()
        .progress_with(pbar)
        .for_each(|benchmark| {
            let filename = format!(
                "data/bm_values/{}_{}-{:02}-{:02}.csv",
                benchmark.name,
                now.year(),
                now.month(),
                now.day()
            );
            println!("Starting {}", filename);
            let range = benchmark.xmax - benchmark.xmin;
            for xi in 0..num_points {
                for yi in 0..num_points {
                    let v = vec![
                        xi as f64 / num_points as f64 * range + benchmark.xmin,
                        yi as f64 / num_points as f64 * range + benchmark.xmin,
                    ];
                    let bound_check = benchmark.is_in_bounds(&v);
                    let r = if bound_check[0] && bound_check[1] {
                        (benchmark.func)(&v)
                    } else {
                        f64::NAN
                    };
                    log_to(
                        &filename,
                        "benchmark,x,y,value",
                        &format!("{},{},{},{}", benchmark.name, v[0], v[1], r),
                    )
                }
            }
            println!("Finished {}", filename);
        });
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

fn log_to(path: &str, header: &str, line: &str) {
    let mut file;
    let file_result = OpenOptions::new()
        .create_new(true)
        .write(true)
        .append(true)
        .open(path);
    if let Ok(f) = file_result {
        file = f;
        writeln!(file, "{}", header).expect("Write to file failed");
    } else {
        file = OpenOptions::new()
            .append(true)
            .open(path)
            .expect("Couldn't open file for appending");
    }

    let mut file = BufWriter::new(file);
    writeln!(file, "{}", line).expect("Failed to write to file");
    file.flush().expect("Failed to flush the BufWriter");
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
