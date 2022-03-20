use crate::rand::Rng;
use rand::thread_rng;

#[derive(Debug, Clone, Copy)]
pub struct ControlParams {
    pub w: f64,
    pub c1: f64,
    pub c2: f64,
}

impl ControlParams {
    pub fn new(w: f64, c1: f64, c2: f64) -> ControlParams {
        ControlParams { w, c1, c2 }
    }

    pub fn generate_by_grid(num_w: usize, num_c1_plus_c2: usize) -> ControlParams {
        let mut rng = thread_rng();
        let mut w;
        let mut c1_plus_c2;
        let mut c1_plus_c2_idx;
        loop {
            let w_idx = rng.gen_range(0, num_w);
            c1_plus_c2_idx = rng.gen_range(0, num_c1_plus_c2);
            w = f64::max(0.1, 4.0 * (w_idx as f64 / num_w as f64));
            c1_plus_c2 = (24.0 - 24.0 * w * w) / (7.0 - 5.0 * w);
            if c1_plus_c2 > 0.0 { break; }
        }
        let c1: f64 = c1_plus_c2_idx as f64 * c1_plus_c2 as f64 / num_c1_plus_c2 as f64;
        let c2 = c1_plus_c2 - c1;
        return ControlParams::new(w, c1, c2);
    }

    pub fn generate_by_dist(dist: f64) -> ControlParams {
        let mut rng = thread_rng();
        assert!(dist > 0.0 && dist < 4.03, "dist must be < 4.03 and > 0.0, but was {}", dist);
        let mut c1_plus_c2;
        let mut w;
        loop {
            w = rng.gen_range(0.0, 1.0);
            c1_plus_c2 = (24.0 - 24.0 * w * w) / (7.0 - 5.0 * w) - dist;
            if c1_plus_c2 > 0.0 { break; }
        }
        let c1: f64 = rng.gen_range(0.0, c1_plus_c2);
        let c2 = c1_plus_c2 - c1;
        return ControlParams::new(w, c1, c2);
    }

    pub fn generate_random_in_grid() -> ControlParams {
        let step_size = 0.1;
        let c_max = 4.0;
        let w_max = 1.0;
        let mut rng = thread_rng();
        let w = step_size * rng.gen_range(0, (w_max / step_size) as usize + 1) as f64;
        let c = step_size * rng.gen_range(0, (c_max / step_size) as usize + 1) as f64;
        return ControlParams::new(w, c, c);
    }
    pub fn generate_multiple_in_grid() -> Vec<ControlParams> {
        let mut cps: Vec<ControlParams> = vec![];
        let step_size_w = 0.1;
        let step_size_c = 0.4;
        let mut c = 4.0;
        let c_min = 0.0;
        let mut w = 1.0;
        let w_min = 0.0;

        while c >= c_min {
            while w >= w_min {
                cps.push(ControlParams::new(w, c / 2.0, c / 2.0));
                w -= step_size_w;
            }
            w = 1.0;
            c -= step_size_c;
        }
        return cps;
    }

    pub fn _generate_poli_biased(distance_to_boundary: f64) -> ControlParams {
        assert!(distance_to_boundary > 0.0 && distance_to_boundary < 4.03, "distance_to_boundary must be < 4.03 and > 0.0");
        let mut w;
        let mut c1;
        let mut c2;
        let mut poli_satisfied;
        let mut close_to_boundary;
        let mut rng = thread_rng();
        loop {
            w = rng.gen_range(0f64, 1f64);
            c1 = rng.gen_range(0.0, 2.0);
            c2 = rng.gen_range(0.0, 2.0);
            poli_satisfied = c1 + c2 < (24.0 - 24.0 * w * w) / (7.0 - 5.0 * w);
            close_to_boundary = c1 + c2 > (24.0 - 24.0 * w * w) / (7.0 - 5.0 * w) - distance_to_boundary;
            if poli_satisfied && close_to_boundary {
                break;
            }
        }
        ControlParams { w, c1, c2 }
    }

    pub fn _generate_by_poli() -> ControlParams {
        let mut w;
        let mut c1;
        let mut c2;
        let mut rng = thread_rng();
        loop {
            w = rng.gen_range(0f64, 1f64);
            c1 = rng.gen_range(-1f64, 1f64);
            c2 = rng.gen_range(-1f64, 1f64);
            if c1 + c2 < (24.0 - 24.0 * w * w) / (7.0 - 5.0 * w) {
                break;
            }
        }
        ControlParams { w, c1, c2 }
    }

    pub fn _generate_no_check() -> ControlParams {
        let w;
        let c1;
        let c2;
        let mut rng = thread_rng();
        w = rng.gen_range(0f64, 1f64);
        c1 = rng.gen_range(-1f64, 1f64);
        c2 = rng.gen_range(-1f64, 1f64);
        ControlParams { w, c1, c2 }
    }
}

