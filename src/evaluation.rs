use core::fmt::Display;

/// Contains metrics used to evaluate the PSO for one iteration.
/// Eventually logged as a vector to a CSV file for graphing.
#[derive(Debug)]
pub struct Evaluation {
    pub iteration: usize,
    /// The global best fitness value over all time
    pub gbest_fit: f64,
    /// The global best fitness value this iteration
    pub curr_gbest_fit: Option<f64>,
    /// The global diversity of all particles
    pub gdiversity: f64,
    /// The magnitudes of each particles' velocity vector
    pub _vel_mags: Vec<f64>,
    /// The magnitudes of each particles' location vector
    pub _loc_mags: Vec<f64>,
    /// The percentage of particles that are out of bounds
    pub perc_oob: f64,
}
impl Display for Evaluation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "perc_oob:{:.4}, gbest_fit:{:.4e}, gdiversity:{:.4e}", self.perc_oob, self.gbest_fit, self.gdiversity)
    }
}
