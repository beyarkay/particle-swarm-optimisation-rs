use crate::benchmarks::Benchmark;
use rand::prelude::*;
use crate::evaluation::Evaluation;
use crate::ControlParams;
use crate::PositionRepair;
use crate::Strategy;
use core::fmt;

use rand::thread_rng;
use rand::Rng;
use std::cmp::min_by;
use std::cmp::Ordering;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::BufWriter;

#[derive(Debug)]
pub struct Swarm {
    /// The number of particles in the swarm.
    num_particles: usize,
    /// The number of dimensions that the swarm is exploring.
    num_dims: usize,
    /// The locations of each particle in the swarm.
    locs: Vec<Vec<f64>>,
    /// The velocities of each particle in the swarm.
    vels: Vec<Vec<f64>>,
    /// The personal best locations of each particle in the swarm.
    pbests: Vec<Vec<f64>>,
    num_neighs: usize,
    /// The global best location that the swarm has found.
    gbest_loc: Option<Vec<f64>>,
    /// The evaluations of the swarm every
    evals: Vec<Evaluation>,
    pos_rep: PositionRepair,
    /// The number of stagnant iterations before Control Parameters should be changed.
    max_stagnent_iters: usize,
    /// The number of consecutive iterations without pbest improvement, per particle.
    stagnant_iters: Vec<usize>,
    /// The control parameters for each particle.
    cps: Vec<ControlParams>,
    w: Option<f64>,
    c1: Option<f64>,
    c2: Option<f64>,
    strat: Strategy,
}

impl Swarm {
    /// Return a swarm with the given parameters, random starting locations and zero velocity
    /// vectors. The benchmark is just used to ensure the particles are initialised in the correct
    /// range.
    pub fn new(
        num_particles: usize,
        num_dimensions: usize,
        cp: &ControlParams,
        benchmark: &Benchmark,
        max_stagnent_iters: usize,
        strat: Strategy,
    ) -> Swarm {
        let mut rng = thread_rng();
        let locs: Vec<Vec<f64>> = (0..num_particles)
            .into_iter()
            .map(|_| {
                (0..num_dimensions)
                    .into_iter()
                    .map(|_| rng.gen_range(benchmark.xmin, benchmark.xmax))
                    .collect()
            })
            .collect();
        let vels = (0..num_particles)
            .into_iter()
            .map(|_| {
                (0..num_dimensions)
                    .into_iter()
                    .map(|_| rng.gen_range(-1.0, 1.0))
                    .collect()
            })
            .collect();
        let cp_probs = ControlParams::generate_from_data("data/last_iter.csv", benchmark);
        let cps: Vec<ControlParams> = match strat {
            Strategy::EmpiricallyTuned(_val) => (0..num_particles)
                .into_iter()
                .map(|_|  {
                    cp_probs
                        .choose_weighted(&mut rng, |opt| opt.1)
                        .expect("No options given")
                        .0
                }).collect(),
            Strategy::RandomAccelerationCoefficients => (0..num_particles)
                .into_iter()
                .map(|_| ControlParams::generate_by_poli())
                .collect(),
            Strategy::None => vec![cp.clone(); num_particles],
        };
        let max_stagnent_iters = match strat {
            Strategy::EmpiricallyTuned(_) => 1,
            Strategy::RandomAccelerationCoefficients => 1,
            Strategy::None => max_stagnent_iters,
        };

        Swarm {
            num_particles,
            num_dims: num_dimensions,
            locs: locs.clone(),
            vels,
            pbests: locs,
            num_neighs: num_particles - 1,
            gbest_loc: None,
            evals: vec![],
            pos_rep: PositionRepair::Random,
            max_stagnent_iters,
            stagnant_iters: vec![0; num_particles],
            cps,
            w: Some(cp.w),
            c1: Some(cp.c1),
            c2: Some(cp.c2),
            strat,
        }
    }

    /// Go through and update the ControlParams if stagnant_iters > max_stagnent_iters or
    /// increment stagnant_iters.
    pub fn resample_cps(&mut self, cp_probs: &Option<Vec<(ControlParams, f64)>>) {
        if self.max_stagnent_iters != 0 {
            let mut rng = thread_rng();
            self.stagnant_iters
                .iter_mut()
                .zip(self.cps.iter_mut())
                .map(|(si, cp)| {
                    if *si >= self.max_stagnent_iters {
                        *si = 0;
                        *cp = if let Some(cp_probs) = cp_probs {
                            cp_probs
                                .choose_weighted(&mut rng, |opt| opt.1)
                                .expect("No options given")
                                .0
                        } else {
                            ControlParams::generate_by_poli()
                        }
                    }
                })
                .collect()
        }
    }

    pub fn step(&mut self, benchmark: &Benchmark, iteration: usize, cp_probs: &Option<Vec<(ControlParams, f64)>>, verbose: bool) {
        // Go through the pbests, and update them if needed
        let prev_pbests = self.pbests.clone();
        self.pbests = self
            .pbests
            .iter()
            .zip(self.locs.iter())
            .map(|(pb, loc)| {
                min_by(pb, loc, |a, b| {
                    if self.pos_rep == PositionRepair::Invisible
                        && !benchmark.is_in_bounds(&b).iter().all(|x| *x)
                    {
                        // If the current position is OOB, ignore it for pbest calculations
                        return Ordering::Less;
                    }
                    let fna = (benchmark.func)(a);
                    let fnb = (benchmark.func)(b);
                    // Ignore NaNs during comparisons
                    match fna.partial_cmp(&fnb) {
                        Some(t) => t,
                        None => {
                            if fna.is_nan() {
                                Ordering::Greater
                            } else {
                                Ordering::Less
                            }
                        }
                    }
                })
                .to_vec()
            })
            .collect();

        self.stagnant_iters = prev_pbests
            .iter()
            .zip(self.pbests.iter())
            .zip(self.stagnant_iters.iter())
            .map(|((prev_vec, curr_vec), stag_iter)| {
                stag_iter
                    + if prev_vec
                        .iter()
                        .zip(curr_vec.iter())
                        .all(|(prev, curr)| prev == curr)
                    {
                        1
                    } else {
                        0
                    }
            })
            .collect();

        // TODO solve calls step, but both step and solve call resample_cps()...
        self.resample_cps(cp_probs);

        // let before = mag(&avg(&self.vels));
        // Go through the velocities, and update them
        self.vels = self
            .vels
            .iter()
            .zip(self.pbests.iter())
            .zip(self.cps.to_vec())
            .zip(self.locs.to_vec())
            .map(|(((vel, pbest), cp), loc)| {
                // Update each component of the velocity

                // let before = mag(&vel);
                let tmp = vel
                    .iter()
                    .zip(self.gbest_loc.as_ref().unwrap())
                    .zip(pbest)
                    .zip(loc)
                    .map(|(((v, gbest_comp), pbest_comp), x)| {
                        let mut rng = thread_rng();
                        let a = cp.w * v;
                        let r1 = rng.gen_range(0f64, 1f64);
                        let r2 = rng.gen_range(0f64, 1f64);
                        let b = cp.c1 * r1 * (pbest_comp - x);
                        let c = cp.c2 * r2 * (gbest_comp - x);
                        let result = a + b + c;
                        // println!("{:.2} = {:.2} * {:.2} + {:.2} * {:.2} * ({:.2} - {:.2}) + {:.2} * {:.2} * ({:.2} - {:.2})",
                        //          result,
                        //          cp.w, x,
                        //          cp.c1, r1, pbest_comp, x,
                        //          cp.c2, r2, gbest_comp, x);
                        return result;
                    })
                    .collect();
                tmp
            })
            .collect();

        //println!("{:.2?}", mag(&avg(&self.vels)) - before);

        // Update the particle positions
        self.locs = self
            .locs
            .iter()
            .zip(&self.vels)
            .map(|(loc, vel)| {
                let new_loc = loc.iter().zip(vel).map(|(l, v)| l + v).collect();
                new_loc
            })
            .collect();

        // Evaluate after the velocities and positions are updated, but before the out of bounds
        // particles are reset
        let should_log = (iteration < 100)
            || (iteration < 1_000 && iteration % 10 == 0)
            || (iteration < 10_000 && iteration % 100 == 0);
        if should_log {
            self.evaluate(&benchmark, iteration, verbose);
        }

        // Repair particle positions
        self.locs = self
            .locs
            .clone()
            .into_iter()
            .map(|loc| {
                match self.pos_rep {
                    PositionRepair::Random => {
                        let bounds = benchmark.is_in_bounds(&loc);
                        if bounds.iter().all(|x| *x) {
                            loc
                        } else {
                            // Only re-instantiate the dimensions which are out of bounds
                            loc.into_iter()
                                .zip(bounds.iter())
                                .map(|(l, b)| {
                                    if *b {
                                        l
                                    } else {
                                        let mut rng = thread_rng();
                                        rng.gen_range(benchmark.xmin, benchmark.xmax)
                                    }
                                })
                                .collect()
                        }
                    }
                    PositionRepair::Invisible => loc,
                    _ => unimplemented!(
                        "PositionRepair::{:?} hasn't been implemented yet",
                        self.pos_rep
                    ),
                }
            })
            .collect();
    }

    /// Evaluate the current position of the swarm.
    /// Get metrics about swarm diversity, the distribution of the particles' solutions, the best
    /// solution, and the distance from the known best solution(s).
    pub fn evaluate(&mut self, benchmark: &Benchmark, iteration: usize, _verbose: bool) {
        let in_bounds: Vec<bool> = self
            .locs
            .clone()
            .into_iter()
            // exclude locations that are OOB
            .map(|loc| (benchmark.is_in_bounds(&loc).iter().all(|b| *b)))
            .collect();

        let num_oob = in_bounds
            .iter()
            .map(|in_bound| if *in_bound { 0.0 } else { 1.0 })
            .sum::<f64>();
        let one_over_n = 1.0 / self.locs.len() as f64;
        let perc_oob = num_oob / self.locs.len() as f64;
        for (loc, within_bounds) in self.locs.iter().zip(in_bounds) {
            // Calculate the best location over all time
            if within_bounds {
                self.gbest_loc = min_by(self.gbest_loc.clone(), Some(loc.clone()), |a, b| {
                    if a.is_none() {
                        Ordering::Greater
                    } else {
                        let fna = (benchmark.func)(a.as_ref().unwrap());
                        let fnb = (benchmark.func)(b.as_ref().unwrap());
                        // Ignore NaNs during comparisons
                        match fna.partial_cmp(&fnb) {
                            Some(t) => t,
                            None => {
                                if fna.is_nan() {
                                    Ordering::Greater
                                } else {
                                    Ordering::Less
                                }
                            }
                        }
                    }
                })
            }
        }

        let loc_avg = avg(&self.locs);
        let _loc_mag = mag(&loc_avg);

        let vel_avg = avg(&self.vels);
        let _vel_mag = mag(&vel_avg);

        let gvariance_vec = self
            .locs
            .iter()
            .map(|loc| {
                // calculate 1/n * (x - xbar)^2
                loc.iter()
                    .zip(loc_avg.iter())
                    .map(|(l, l_avg)| one_over_n * (l - l_avg).powi(2))
                    .collect()
            })
            .reduce(|acc: Vec<f64>, curr| {
                // calculate Sum( 1/n * (x - xbar)^2 )
                acc.iter().zip(curr.iter()).map(|(a, c)| a + c).collect()
            })
            .expect("self.locs is empty");

        // Square root variance to get diversity
        let gdiversity_vec: Vec<f64> = gvariance_vec.into_iter().map(|v| v.sqrt()).collect();

        let gdiversity = mag(&gdiversity_vec);

        if iteration < 5 {
            // println!("perc_oob: {:.2}, {:?}", perc_oob, loc_avg.iter().zip(vel_avg.iter()).collect::<Vec<(&f64, &f64)>>());
        }

        self.evals.push(Evaluation {
            iteration,
            gbest_fit: (benchmark.func)(&self.gbest_loc.as_ref().unwrap()),
            curr_gbest_fit: None, //if global_best_loc.is_some() { Some((benchmark.func)(&global_best_loc.unwrap().clone())) } else { None },
            gdiversity,
            _vel_mags: vec![], //vel_mags,
            _loc_mags: vec![], //loc_mags,
            perc_oob,
        });
    }

    /// Log the swarm's progress to file.
    /// This logs the following parameters as a CSV line to disk (all floats are rounded to 4dp):
    /// - Number of dimensions
    /// - Number of particles
    /// - The maximum number of stagnant iterations
    /// - dist
    /// - $w$
    /// - $c_1$
    /// - $c_1$
    /// - The name of the benchmark
    /// - the repetition number
    /// - the iteration number
    /// - the global best fitness
    /// - the current best fitness
    /// - the swarm's diversity
    /// - The percentage of particles out of bounds
    /// - The strategy (EmpiricallyTuned or RandomAccelerationCoefficients)
    pub fn log_to(
        &self,
        path: &str,
        benchmark_name: &str,
        rep_num: usize,
        dist: f64,
        max_stagnent_iters: usize,
    ) {
        let mut file;
        let file_result = OpenOptions::new()
            .create_new(true)
            .write(true)
            .append(true)
            .open(path);
        if let Ok(f) = file_result {
            file = f;
            writeln!(file, "num_dims,num_particles,max_stagnent_iters,dist,w,c1,c2,benchmark,rep_num,iter_num,gbest_fit,curr_gbest_fit,gdiversity,perc_oob,strat")
                .expect("Write to file failed");
        } else {
            file = OpenOptions::new()
                .append(true)
                .open(path)
                .expect("Couldn't open file for appending");
        }

        let mut file = BufWriter::new(file);

        for eval in self.evals.iter() {
            writeln!(
                file,
                "{},{},{},{:.4e},{:.4e},{:.4e},{:.4e},{},{},{},{:.4e},{:.4e},{:.4e},{:.4e},{:?}",
                // Predictor variables
                self.num_dims,
                self.num_particles,
                max_stagnent_iters,
                dist,
                self.w.unwrap_or(f64::NAN),
                self.c1.unwrap_or(f64::NAN),
                self.c2.unwrap_or(f64::NAN),
                benchmark_name,
                rep_num,
                eval.iteration,
                // Response variables
                eval.gbest_fit,
                eval.curr_gbest_fit.unwrap_or(f64::NAN),
                eval.gdiversity,
                eval.perc_oob,
                self.strat
            )
            .expect("Failed to write to file");
        }
        file.flush().expect("Failed to flush the BufWriter");
    }

    pub fn solve(&mut self, benchmark: &Benchmark, num_iterations: i32, cp_probs: Option<Vec<(ControlParams, f64)>>, verbose: bool) {
        // let pbar = ProgressBar::new(num_iterations as u64);
        // if !verbose {
        //     pbar.set_style(ProgressStyle::default_bar()
        //                 .template("      [-{eta} +{elapsed} {prefix}] {bar:40.cyan/blue} {pos:>7}/{len:7} ({per_sec}) {msg}")
        //                 .progress_chars("=>~"));
        //     pbar.set_prefix(format!("{}p {}D", self.num_particles, self.num_dims));
        // }

        for i in 1..num_iterations {
            // if verbose { println!("{}-th Eval: {}", i-1, self.evals.last().expect("evals is empty")); }
            // if !verbose { pbar.inc(1); }

            self.step(benchmark, i as usize, &cp_probs, verbose);

            // TODO solve calls step, but both step and solve call resample_cps()...
            // self.resample_cps();

            // if !verbose {
            //     let eval = self.evals.last().expect("evals is empty");
            //     pbar.set_message(format!("perc_oob:{:.4}, gbest_fit_diff:{:.4e}, gdiversity_diff:{:.4e}",
            //                              eval.perc_oob,
            //                              eval.gbest_fit - self.evals.first().expect("evals is empty").gbest_fit,
            //                              eval.gdiversity - self.evals.first().expect("evals is empty").gdiversity));
            // }
        }
        // if !verbose { pbar.finish(); }
    }
}

fn mag(vec: &Vec<f64>) -> f64 {
    vec.iter().map(|mag| mag * mag).sum::<f64>().sqrt()
}

fn avg(vec: &Vec<Vec<f64>>) -> Vec<f64> {
    let one_over_n = 1.0 / vec.len() as f64;
    vec.clone()
        .into_iter()
        .reduce(|acc, curr| {
            acc.iter()
                .zip(curr.iter())
                .map(|(a, c)| (a + one_over_n * c))
                .collect()
        })
        .expect("vec is empty")
        .to_vec()
}

impl fmt::Display for Swarm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Swarm: dims={}, particles={}, neighbours={}, \n||vel||={:?}, \n||loc||={:?}",
            self.num_dims,
            self.num_particles,
            self.num_neighs,
            self.vels
                .iter()
                .map(|vel| (vel.iter().map(|v| v.powi(2)).sum::<f64>())
                    .powf(1f64 / vel.len() as f64))
                .collect::<Vec<f64>>(),
            self.locs
                .iter()
                .map(|loc| (loc.iter().map(|l| l.powi(2)).sum::<f64>())
                    .powf(1f64 / loc.len() as f64))
                .collect::<Vec<f64>>()
        )
    }
}
