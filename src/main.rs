#![feature(test)]
extern crate test;
mod benchmarks;
mod swarm;
mod control_params;
mod evaluation;
extern crate rand;

use crate::benchmarks::Benchmark;
use crate::control_params::ControlParams;
use crate::swarm::Swarm;
use chrono::{Datelike, Utc};
use rand::{prelude::SliceRandom, thread_rng};

fn main() {

    let mut rng = thread_rng();
    let mut benchmarks = benchmarks::get_benchmarks();

    let num_particles = 30;
    let num_dimensions = 30;
    let num_iterations = 5000;
    let num_repetitions = 10;

    if true {
        println!("Starting eval_std_pso");
        benchmarks.shuffle(&mut rng);
        eval_std_pso(num_particles, num_dimensions, num_repetitions, num_iterations, &benchmarks);
    }
    if false {
        println!("Starting find_optimal_cps");
        benchmarks.shuffle(&mut rng);
        find_optimal_cps(num_particles, num_dimensions, num_repetitions, num_iterations, &benchmarks);
    }
    if false {
        println!("Starting random_poli_sampling");
        benchmarks.shuffle(&mut rng);
        random_poli_sampling(num_particles, num_dimensions, num_repetitions, num_iterations, &benchmarks);
    }
}

fn print_details(run_idx: usize, num_runs: usize, max_stagnent_iters: usize, dist: f64, benchmark_name: &str, cp: &ControlParams) {
    println!("({}/{} {:.2}%) max_stagnent_iters: {}, dist: {:.2}, benchmark: {: <14}, w={:.2}, c1={:.2}, c2={:.2}", 
             run_idx, 
             num_runs,
             run_idx as f64 / num_runs as f64 * 100.0,
             max_stagnent_iters,
             dist,
             benchmark_name,
             cp.w,
             cp.c1,
             cp.c2,
             );
}


/// Evaluate a PSO against w=0.7, c1=c2=1.4
fn eval_std_pso(num_particles: usize, num_dimensions: usize, num_repetitions: usize, num_iterations: usize, benchmarks: &Vec<Benchmark>) {
    let mut run_idx = 1;
    let num_runs = benchmarks.len() * num_repetitions;
    let cp = ControlParams::new(0.7, 1.4, 1.4);
    let now = Utc::now();
    let filename = format!("data/raw/std_pso_{}-{:02}-{:02}.csv", now.year(), now.month(), now.day());

    for benchmark in benchmarks {
        print_details(run_idx, num_runs, 0, 0.0, &benchmark.name, &cp);
        for rep_num in 0..num_repetitions {
            let mut swarm = Swarm::new(num_particles, num_dimensions, &cp, &benchmark, 0, 0.0);
            swarm.evaluate(&benchmark, 0, false);
            swarm.solve(&benchmark, num_iterations as i32, false);
            swarm.log_to(&filename, &benchmark, rep_num, 0.0, 0);
            run_idx += 1;
        }
    }
}

/// Go through every benchmark and then grid-search for the best control parameters
fn find_optimal_cps(num_particles: usize, num_dimensions: usize, num_repetitions: usize, num_iterations: usize, benchmarks: &Vec<Benchmark>) {
    let mut run_idx = 1;
    let mut rng = thread_rng();
    let mut cps = ControlParams::generate_multiple_in_grid();
    cps.shuffle(&mut rng);
    let num_runs = cps.len() * benchmarks.len() * num_repetitions;
    let now = Utc::now();
    let filename = format!("data/raw/opt_{}-{:02}-{:02}.csv", now.year(), now.month(), now.day());

    for benchmark in benchmarks {
        for cp in &cps {
            print_details(run_idx, num_runs, 0, 0.0, &benchmark.name, &cp);
            for rep_num in 0..num_repetitions {
                let mut swarm = Swarm::new(num_particles, num_dimensions, &cp, &benchmark, 0, 0.0);
                swarm.evaluate(&benchmark, 0, false);
                swarm.solve(&benchmark, num_iterations as i32, false);
                swarm.log_to(&filename, &benchmark, rep_num, 0.0, 0);
                run_idx += 1;
            }
        }
    }
}

/// Go through every benchmark and randomly resample a control parameter every iteration, or upon
/// stagnation
fn random_poli_sampling(num_particles: usize, num_dimensions: usize, num_repetitions: usize, num_iterations: usize, benchmarks: &Vec<Benchmark>) {
    let mut run_idx = 1;
    let max_stagnent_iters_vec = vec![0, 1, 3, 9, 27, 81, 243, 729, 2187, 5000];
    let cp = ControlParams::generate_by_poli();
    let num_runs = benchmarks.len() * num_repetitions * max_stagnent_iters_vec.len();
    let now = Utc::now();
    let filename = format!("data/raw/resample_{}-{:02}-{:02}.csv", now.year(), now.month(), now.day());

    for benchmark in benchmarks {
        for max_stagnent_iters in &max_stagnent_iters_vec {
            print_details(run_idx, num_runs, *max_stagnent_iters, 0.0, &benchmark.name, &cp);
            for rep_num in 0..num_repetitions {
                let mut swarm = Swarm::new(num_particles, num_dimensions, &cp, &benchmark, *max_stagnent_iters, 0.0);
                swarm.evaluate(&benchmark, 0, false);
                swarm.solve(&benchmark, num_iterations as i32, false);
                swarm.log_to(&filename, &benchmark, rep_num, 0.0, *max_stagnent_iters);
                run_idx += 1;
            }
        }
    }
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

#[cfg(test)]
mod tests {
    use crate::benchmarks::get_benchmarks;

    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_swarm_evaluate(b: &mut Bencher) {
        let benchmarks = get_benchmarks();
        let benchmark = benchmarks.last().expect("There aren't any benchmarks");
        let mut swarm = Swarm::new(30, 30, &ControlParams::_generate_by_poli(), &benchmark, 3, 0.5);
        b.iter(|| swarm.evaluate(&benchmark, 0, false))
    }
}
