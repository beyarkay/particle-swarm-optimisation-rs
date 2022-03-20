use crate::ControlParams;
use crate::PositionRepair;
use crate::evaluation::Evaluation;
use std::cmp::Ordering;
use std::cmp::min_by;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::io::prelude::*;
use std::path::Path;
use indicatif::{ProgressBar, ProgressStyle};
use core::fmt;
use rand::prelude::ThreadRng;
use rand::thread_rng;
use rand::Rng;
use crate::benchmarks::Benchmark;


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
    /// The random number generator used by the swarm to initialise.
    rng: ThreadRng,
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
    /// The horizontal distance to the Poli boundary by which control parameters should be selected
    distance_to_boundary: f64,
    w: Option<f64>,
    c1: Option<f64>,
    c2: Option<f64>
}


impl Swarm {
    /// Return a swarm with the given parameters, random starting locations and zero velocity
    /// vectors
    pub fn new(num_particles: usize, num_dims: usize, cp: &ControlParams, benchmark: &Benchmark, max_stagnent_iters: usize, distance_to_boundary: f64) -> Swarm {
        let mut rng = thread_rng();
        let locs: Vec<Vec<f64>> = (0..num_particles).into_iter().map(|_| {
            (0..num_dims).into_iter().map(|_| { rng.gen_range(benchmark.xmin, benchmark.xmax) }).collect()
        }).collect();
        let vels = (0..num_particles).into_iter().map(|_| {
            (0..num_dims).into_iter().map(|_| { rng.gen_range(-1.0, 1.0) }).collect()
        }).collect();
        let cps = if max_stagnent_iters == 0 {
            vec![cp.clone(); num_particles]
        } else {
            (0..num_particles).into_iter().map(|_| ControlParams::generate_random_in_grid()).collect()
        };
        Swarm { 
            num_particles,
            num_dims,
            locs: locs.clone(),
            vels,
            pbests: locs,
            num_neighs: num_particles - 1,
            rng,
            gbest_loc: None,
            evals: vec![],
            pos_rep: PositionRepair::Random,
            max_stagnent_iters,
            stagnant_iters: vec![0; num_particles],
            cps,
            distance_to_boundary,
            w: Some(cp.w),
            c1: Some(cp.c1),
            c2: Some(cp.c2),
        }
    }

    /// Go through and update the ControlParams if stagnant_iters > max_stagnent_iters or
    /// increment stagnant_iters.
    pub fn resample_cps(&mut self) {
        if self.max_stagnent_iters != 0 {
            self.stagnant_iters.iter_mut().zip(self.cps.iter_mut()).map(|(si, cp)| {
                if *si >= self.max_stagnent_iters {
                    *si = 0;
                    *cp = ControlParams::generate_random_in_grid();
                }
            }).collect()
        }
    }

    pub fn step(&mut self, benchmark: &Benchmark, iteration: usize, verbose: bool) {
        // Go through the pbests, and update them if needed
        let prev_pbests = self.pbests.clone();
        self.pbests = self.pbests.iter().zip(self.locs.iter()).map(|(pb, loc)| {
            min_by(pb, loc, |a, b| {
                if self.pos_rep == PositionRepair::Invisible && !benchmark.is_in_bounds(&b).iter().all(|x| *x) {
                    // If the current position is OOB, ignore it for pbest calculations
                    return Ordering::Less;
                }
                let fna = (benchmark.func)(a); 
                let fnb = (benchmark.func)(b); 
                // Ignore NaNs during comparisons
                match fna.partial_cmp(&fnb) {
                    Some(t) => t,
                    None => if fna.is_nan() { Ordering::Greater } else { Ordering::Less },
                }
            }).to_vec()
        }).collect();

        self.stagnant_iters = prev_pbests.iter()
            .zip(self.pbests.iter())
            .zip(self.stagnant_iters.iter())
            .map(|((prev_vec, curr_vec), stag_iter)| {
                stag_iter + if prev_vec.iter().zip(curr_vec.iter()).all(|(prev, curr)| prev == curr ) { 1 } else { 0 }
            }).collect();

        self.resample_cps();

        // Go through the velocities, and update them
        self.vels = self.vels.iter()
            .zip(self.pbests.iter())
            .zip(self.cps.to_vec())
            .map(|((vel, pbest), cp)| {
                // Update each component of the velocity
                vel.iter()
                    .zip(self.gbest_loc.as_ref().unwrap())
                    .zip(pbest)
                    .map(|((v, gbest_comp), pbest_comp)| {
                        cp.w * v
                            + cp.c1 * self.rng.gen_range(0f64, 1f64) * (pbest_comp - v) 
                            + cp.c2 * self.rng.gen_range(0f64, 1f64) * (gbest_comp - v)
                    }).collect()
            }).collect();

        // Update the particle positions
        self.locs = self.locs.iter()
            .zip(&self.vels)
            .map(|(loc, vel)| {
                let new_loc = loc.iter()
                    .zip(vel)
                    .map(|(l, v)| l + v)
                    .collect();
                new_loc
            }).collect();

        // Evaluate after the velocities and positions are updated, but before the out of bounds
        // particles are reset
        let should_log = (iteration < 100 )
                || (iteration < 1_000 && iteration % 10 == 0)
                || (iteration < 10_000 && iteration % 100 == 0);
        if should_log {
            self.evaluate(&benchmark, iteration, verbose);
        }

        // Repair particle positions
        self.locs = self.locs.clone().into_iter()
            .map(|loc| {
                match self.pos_rep {
                    PositionRepair::Random => {
                        let bounds = benchmark.is_in_bounds(&loc);
                        if bounds.iter().all(|x| *x) {
                            loc
                        } else {
                            // Only re-instantiate the dimensions which are out of bounds
                            loc.into_iter().zip(bounds.iter()).map(|(l, b)| {
                                if *b { l } else { self.rng.gen_range(benchmark.xmin, benchmark.xmax) }
                            }).collect()
                        }
                    },
                    PositionRepair::Invisible => { loc }
                    _ => unimplemented!("PositionRepair::{:?} hasn't been implemented yet", self.pos_rep)
                }
            }).collect();

        
    }

    /// Evaluate the current position of the swarm.
    /// Get metrics about swarm diversity, the distribution of the particles' solutions, the best
    /// solution, and the distance from the known best solution(s).
    // 453,448 ns/iter (+/- 131,991) before for-loop refactor
    // 127,529 ns/iter (+/- 39,874)  after for-loop refactor
    // 119,068 ns/iter (+/- 40,489) after perc_oob refactor
    pub fn evaluate(&mut self, benchmark: &Benchmark, iteration: usize, verbose: bool) {
        let in_bounds: Vec<bool> = self.locs.clone().into_iter()
            // exclude locations that are OOB
            .map(|loc| (benchmark.is_in_bounds(&loc).iter().all(|b| *b)))
            .collect();

        let num_oob = in_bounds.iter().map(|in_bound| if *in_bound { 0.0 } else { 1.0 }).sum::<f64>();
        let one_over_n = 1.0 / self.locs.len() as f64;
        let perc_oob = num_oob / self.locs.len() as f64;
        let mut loc_avg = vec![0f64; self.num_dims];

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
                                None => if fna.is_nan() { Ordering::Greater } else { Ordering::Less },
                            }
                        }
                })
            }
            // Calculate the sum of all locations across each dimension
            loc_avg = loc_avg.iter().zip(loc).map(|(acc, curr)|  acc + one_over_n * curr ).collect();
        }
        let vel_avg: Vec<f64> = self.vels.clone().into_iter().reduce(|acc, curr| {
            acc.iter().zip(curr.iter()).map(|(a, c)| a + one_over_n * c).collect()
        }).expect("self.vels is empty").to_vec();

        let gvariance_vec = self.locs.iter().map(|loc| {
            // calculate 1/n * (x - xbar)^2
            loc.iter().zip(loc_avg.iter()).map(|(l, l_avg)| one_over_n * (l - l_avg).powi(2)).collect()
        }).reduce(|acc: Vec<f64>, curr| {
            // calculate Sum( 1/n * (x - xbar)^2 )
            acc.iter().zip(curr.iter()).map(|(a, c)| a + c).collect()
        }).expect("self.locs is empty");

        // Square root variance to get diversity
        let gdiversity_vec: Vec<f64> = gvariance_vec.into_iter().map(|v| v.sqrt()).collect();

        if verbose { println!("{}-th  one_over_n: {}, \n\tloc_avg: {:.2?}, \n\tvel_avg: {:.2?}, \n\tdiversity: {:.2?}", iteration, one_over_n, loc_avg, vel_avg, gdiversity_vec); }
        // Magnitiude is sqrt( sum of squares )
        let gdiversity = gdiversity_vec.iter().map(|mag| {
            mag * mag
        }).sum::<f64>().sqrt();
        
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

    pub fn log_to(&self, path: &str, benchmark: &Benchmark, rep_num: usize, dist: f64, max_stagnent_iters: usize) {
        let mut file;
        if !Path::new(path).exists() {
            // If the file doesn't exist, create it and write the csv header line
            file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .append(true)
                .open(path)
                .expect("Failed to create new file");
            writeln!(file, "num_dims,num_particles,max_stagnent_iters,dist,w,c1,c2,benchmark,rep_num,iter_num,gbest_fit,curr_gbest_fit,gdiversity,perc_oob")
                .expect("Write to file failed");
        } else {
            file = OpenOptions::new().append(true).open(path).expect("Couldn't open file for appending");
        }
        let mut file = BufWriter::new(file);

        for eval in self.evals.iter() {
            writeln!(
                file, "{},{},{},{:.4e},{:.4e},{:.4e},{:.4e},{},{},{},{:.4e},{:.4e},{:.4e},{:.4e}", 
                // Predictor variables
                self.num_dims,
                self.num_particles,
                max_stagnent_iters,
                dist,
                self.w.unwrap_or(f64::NAN),
                self.c1.unwrap_or(f64::NAN),
                self.c2.unwrap_or(f64::NAN),
                benchmark.name,
                rep_num,
                eval.iteration,
                // Response variables
                eval.gbest_fit,
                eval.curr_gbest_fit.unwrap_or(f64::NAN),
                eval.gdiversity,
                eval.perc_oob
            ).expect("Failed to write to file");
        }
        file.flush().expect("Failed to flush the BufWriter");
    }

    pub fn solve(&mut self, benchmark: &Benchmark, num_iterations: i32, verbose: bool) {
        let pbar = ProgressBar::new(num_iterations as u64);
        if !verbose {
            pbar.set_style(ProgressStyle::default_bar()
                        .template("      [-{eta} +{elapsed} {prefix}] {bar:40.cyan/blue} {pos:>7}/{len:7} ({per_sec}) {msg}")
                        .progress_chars("=>~"));
            pbar.set_prefix(format!("{}p {}D", self.num_particles, self.num_dims));
        }

        for i in 1..num_iterations {
            if verbose { println!("{}-th Eval: {}", i-1, self.evals.last().expect("evals is empty")); }
            if !verbose { pbar.inc(1); }

            self.step(benchmark, i as usize, verbose);

            self.resample_cps();

            if !verbose { 
                let eval = self.evals.last().expect("evals is empty");
                pbar.set_message(format!("perc_oob:{:.4}, gbest_fit_diff:{:.4e}, gdiversity_diff:{:.4e}", 
                                         eval.perc_oob, 
                                         eval.gbest_fit - self.evals.first().expect("evals is empty").gbest_fit,
                                         eval.gdiversity - self.evals.first().expect("evals is empty").gdiversity));
            }
        }
        if !verbose { pbar.finish(); }
    }
}

impl fmt::Display for Swarm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f, 
            "Swarm: dims={}, particles={}, neighbours={}, \n||vel||={:?}, \n||loc||={:?}", 
            self.num_dims,
            self.num_particles,
            self.num_neighs,
            self.vels.iter().map(|vel| (vel.iter().map(|v| v.powi(2)).sum::<f64>()).powf(1f64 / vel.len() as f64)).collect::<Vec<f64>>(),
            self.locs.iter().map(|loc| (loc.iter().map(|l| l.powi(2)).sum::<f64>()).powf(1f64 / loc.len() as f64)).collect::<Vec<f64>>()
        )
    }
}