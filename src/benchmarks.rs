pub struct Benchmark {
    pub func: Box<dyn Fn(&Vec<f64>) -> f64>,
    pub name: String,
    pub xmin: f64,
    pub xmax: f64,
    pub reference: String,
}

impl Benchmark {
    /// Return true if the location is within bounds
    pub fn is_in_bounds(&self, vec: &Vec<f64>) -> Vec<bool> {
        vec.iter()
            .map(|x| (self.xmin).lt(&x) && (self.xmax).ge(&x))
            .collect()
    }
}

// TODO Build out ALL the benchmarks
pub fn get_benchmarks() -> Vec<Benchmark> {
    let brown_function = Benchmark {
        name: "Brown Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            x.iter()
                .zip(x.iter().skip(1))
                .map(|(x1, x2)| {
                    (x1 * x1).powf(x2 * x2 + 1f64) 
                        + (x2 * x2).powf(x1 * x1 + 1f64) 
                }).sum()
        }),
        xmin: -1.0,
        xmax: 4.0,
        reference: "O. Begambre, J. E. Laier, “A hybrid Particle Swarm Optimization - Simplex Algorithm (PSOS) for Structural Damage Identification,” Journal of Advances in Engineering Software, vol. 40, no. 9, pp. 883-891, 2009.".to_string(),
    };

    // https://www.al-roomi.org/benchmarks/unconstrained/n-dimensions/166-cosine-mixture-function
    let cosine_mixture_function = Benchmark {
        name: "Cosine Mixture Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            -(0.1 * x.iter().map(|xi| (5.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
                - x.iter().map(|xi| xi * xi).sum::<f64>()
)        }),
        xmin: -1.0,
        xmax: 1.0,
        reference: "M. M. Ali, C. Khompatraporn, Z. B. Zabinsky, “A Numerical Evaluation of Several Stochastic Algorithms on Selected Continuous Global Optimization Test Problems,” Journal of Global Optimization, vol. 31, pp. 635-672, 2005.".to_string(),
    };

    let deb_2_function = Benchmark {
        name: "Deb 3 Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            - (1f64 / x.len() as f64) * 
                 x.iter().map(|xi| (5f64 * std::f64::consts::PI * (xi.powf(0.75f64) - 0.05f64)).sin().powi(6)).sum::<f64>()
        }),
        xmin: -1.0,
        xmax: 1.0,
        reference: " Ali R. Al-Roomi (2015). Unconstrained Single-Objective Benchmark Functions Repository [https://www.al-roomi.org/benchmarks/unconstrained]. Halifax, Nova Scotia, Canada: Dalhousie University, Electrical and Computer Engineering.".to_string(),
    };

    // https://www.al-roomi.org/benchmarks/unconstrained/n-dimensions/231-deb-s-function-no-01
    let deb_1_function = Benchmark {
        name: "Deb 1 Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            - (1f64 / x.len() as f64) 
                * x.iter().map(|xi| (5f64 * std::f64::consts::PI * xi).sin().powi(6)).sum::<f64>()
                + 1.0 // Add one so the minimum is zero
        }),
        xmin: -1.0,
        xmax: 1.0,
        reference: " Ali R. Al-Roomi (2015). Unconstrained Single-Objective Benchmark Functions Repository [https://www.al-roomi.org/benchmarks/unconstrained]. Halifax, Nova Scotia, Canada: Dalhousie University, Electrical and Computer Engineering.".to_string(),
    };

    // https://www.al-roomi.org/benchmarks/unconstrained/n-dimensions/168-exponential-function
    let exponential_function = Benchmark {
        name: "Exponential Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            -(-0.5 * x.iter().map(|xi| xi.powi(2)).sum::<f64>()).exp()
        }),
        xmin: -1.0,
        xmax: 1.0,
        reference: "S. Rahnamyan, H. R. Tizhoosh, N. M. M. Salama, Opposition-Based Differential Evolution (ODE) with Variable Jumping Rate, IEEE Sympousim Foundations Com- putation Intelligence, Honolulu, HI, pp. 81-88, 2007.".to_string(),
    };

    // https://www.al-roomi.org/benchmarks/unconstrained/n-dimensions/169-generalized-griewank-s-function
    let griewank_function = Benchmark {
        name: "Griewank Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            x.iter().map(|xi| (xi * xi) / 4000.0).sum::<f64>() 
                - x.iter().enumerate().map(|(i, xi)| (xi / (1.0 + i as f64).sqrt()).cos()).product::<f64>()
                + 1.0
        }),
        xmin: -100.0,
        xmax: 100.0,
        reference: "A. O. Griewank, Generalized Descent for Global Optimization, Journal of Optimization Theory and Applications, vol. 34, no. 1, pp. 11-39, 1981.".to_string(),
    };

    let generalized_giunta_function = Benchmark {
        name: "Generalized Guinta Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            0.6 + x.iter().map(|xi| {
                let tmp = 16.0/15.0 * xi - 1.0;
                tmp.sin() + tmp.sin().powi(2) + 0.02 * (4.0*tmp).sin()
            }).sum::<f64>()
        }),
        xmin: -1.0,
        xmax: 1.0,
        reference: "S. K. Mishra, Global Optimization By Differential Evolution and Particle Swarm Methods: Evaluation On Some Benchmark Functions, Munich Research Papers in Economics, [Available Online]: http://mpra.ub.uni-muenchen.de/1005/".to_string(),
    };

    let generalized_paviani_function = Benchmark {
        name: "Generalized Paviani Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            x.iter().map(|xi| {
                (10.0 - xi).ln().powi(2) + (xi - 2.0).ln().powi(2)
            }).sum::<f64>() - x.iter().product::<f64>().powi(2)
        }),
        xmin: 2.0001,
        xmax: 10.0,
        reference: "R. I. Jennrich, P. F. Sampson, Application of Stepwise Regression to Non-Linear estimation, Techometrics, vol. 10, no. 1, pp. 63-72, 1968. http://www.jstor.org/discover/10.2307/1266224?uid=3737864&uid=2129&uid=2&uid=70&uid=4&sid=2".to_string(),
    };

    let generalized_price_2_function = Benchmark {
        name: "Generalized Price 2 Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            1.0 + x.iter().map(|xi| xi.sin().powi(2)).sum::<f64>()
                - 0.1 * (x.iter().map(|xi| xi * xi).sum::<f64>()).exp()
        }),
        xmin: -10.0, // TODO 
        xmax: 10.0,  // TODO
        reference: "W. L. Price, A Controlled Random Search Procedure for Global Optimisa- tion, Computer journal, vol. 20, no. 4, pp. 367-370, 1977. [Available Online]: http://comjnl.oxfordjournals.org/content/20/4/367.full.pdf".to_string(),
    };

    let generalized_egg_crate_function = Benchmark { 
        name: "Generalized Egg Crate Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            x.iter().map(|xi| xi * xi ).sum::<f64>() 
                + 24.0 * x.iter().map(|xi| xi.sin().powi(2)).sum::<f64>()
        }),
        xmin: -5.0,
        xmax: 5.0,
        reference: "Jamil, M.; Yang, X. A Literature Survey of Benchmark Functions For Global Optimization Problems. CoRR 2013, abs/1308.4008, [1308.4008]. doi:https://doi.org/10.1504/ijmmno.2013.055204.".to_string(),
    };

    let mishra_7_function = Benchmark {
        name: "Mishra 7 Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            let dim_factorial: f64 = (1..x.len())
                .reduce(|acc, curr| acc * curr)
                .unwrap() as f64;
            assert!(dim_factorial.is_finite());
            (x.iter().product::<f64>() * dim_factorial).powi(2)
        }),
        xmin: -10.0,
        xmax: 10.0,
        reference: "S. K. Mishra, Global Optimization By Differential Evolution and Particle Swarm Methods: Evaluation On Some Benchmark Functions, Munich Research Papers in Economics, [Available Online]: http://mpra.ub.uni-muenchen.de/1005/".to_string(),
    };

    let mishra_1_function = Benchmark {
        name: "Mishra 1 Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            let n = x.len() as f64;
            let x_m: f64 = n - x.iter().take(n as usize - 1).sum::<f64>();
            (1.0 + x_m).powf(x_m)
        }),
        xmin: 0.0,
        xmax: 1.0,
        reference: "S. K. Mishra, Performance of Differential Evolution and Particle Swarm Methods on Some Relatively Harder Multi-modal Benchmark Functions, [Available Online]: http://mpra.ub.uni-muenchen.de/449/".to_string(),
    };

    let pathological_function = Benchmark {
        name: "Pathological Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            x.iter().zip(x.iter().skip(1)).map(|(xi, xi1)| {
                0.5 + ((100.0 * xi * xi + xi1 * xi1).sin().powi(2) - 0.5) 
                    / (1.0 + 0.001 * (xi * xi - 2.0 * xi * xi1 + xi1 * xi1).powi(2))
            }).sum::<f64>()
        }),
        xmin: -100.0,
        xmax: 100.0,
        reference: "S. Rahnamyan, H. R. Tizhoosh, N. M. M. Salama, A Novel Population Initialization Method for Accelerating Evolutionary Algorithms, Computers and Mathematics with Applications, vol. 53, no. 10, pp. 1605-1614, 2007.".to_string(),
    };

    let pinter_function = Benchmark { 
        name: "Pinter Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            let tri_x_iterator = x.iter()
                .enumerate()
                .zip(x.iter().cycle().skip(1))
                .zip(x.iter().cycle().skip(x.len() - 1));

            let first = x.iter()
                .enumerate()
                .map(|(i, xi)| i as f64 * xi * xi)
                .sum::<f64>();
            let a = tri_x_iterator.clone().map(|(((_i, xi), xi_plus_1), xi_less_1)| {
                    xi_less_1 * xi.sin() + xi_plus_1.sin()
                }).sum::<f64>();
            let b = tri_x_iterator.map(|(((_i, xi), xi_plus_1), xi_less_1)| {
                    xi_less_1.powi(2) 
                        - 2.0 * xi 
                        + 3.0 * xi_plus_1 
                        - xi.cos() 
                        + 1.0
                }).sum::<f64>();

             first + a + b
        }),
        xmin: -10.0,
        xmax: 10.0,
        reference: "J. D. Pintér, Global Optimization in Action: Continuous and Lipschitz Optimization Algorithms, Implementations and Applications, Kluwer, 1996.".to_string(),
    };

    let qing_function = Benchmark {
        name: "Qing Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            x.iter().enumerate().map(|(i, xi)| (xi * xi - i as f64).powi(2)).sum::<f64>()
        }),
        xmin: -500.0,
        xmax: 500.0,
        reference: "A. Qing, “Dynamic Differential Evolution Strategy and Applications in Electromag- netic Inverse Scattering Problems,” IEEE Transactions on Geoscience and remote Sens- ing, vol. 44, no. 1, pp. 116-125, 2006.".to_string(),
    };

    let rosenbrock_function = Benchmark { 
        name: "Rosenbrock Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            x.iter().zip(x.iter().take(x.len() - 1)).map(|(xi, xi_plus_1)| {
                100.0 * (xi_plus_1 - xi * xi).powi(2) + (xi - 1.0).powi(2)
            }).sum()
        }),
        xmin: -30.0,
        xmax: 30.0,
        reference: "H. H. Rosenbrock, An Automatic Method for Finding the Greatest or least Value of a Function, Computer Journal, vol. 3, no. 3, pp. 175-184, 1960. [Available Online]: http://comjnl.oxfordjournals.org/content/3/3/175.full.pdf".to_string(),
    };

    let ripple_25_function = Benchmark {
        name: "Ripple 25 Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            x.iter().map(|xi| {
                let exp_comp: f64 = -2.0 * 2f64.ln() * ((xi - 0.1) / 0.8).powi(2);
                let sin_comp: f64 = (5.0 * std::f64::consts::PI * xi).sin().powi(6);
                -(exp_comp).exp() * sin_comp
            }).sum()
        }),
        xmin: 0.0,
        xmax: 1.0,
        reference: "Jamil, M.; Yang, X. A Literature Survey of Benchmark Functions For Global Optimization Problems. CoRR 2013, abs/1308.4008, [1308.4008]. doi:https://doi.org/10.1504/ijmmno.2013.055204.".to_string(),
    };

    let sphere_function = Benchmark {
        name: "Sphere Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            x.iter().map(|x| { x * x }).sum()
        }),
        xmin: -1.0,
        xmax:  1.0,
        reference: "M. A. Schumer, K. Steiglitz, Adaptive Step Size Random Search, IEEE Transactions on Automatic Control. vol. 13, no. 3, pp. 270-276, 1968.".to_string(),
    };

    let schumer_steiglitz_function = Benchmark {
        name: "Schumer Steiglitz Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            x.iter().map(|xi| xi.powi(4)).sum()
        }),
        xmin: -100.0,
        xmax: 100.0,
        reference: " M. A. Schumer, K. Steiglitz, “Adaptive Step Size Random Search,” IEEE Transactions on Automatic Control. vol. 13, no. 3, pp. 270-276, 1968".to_string(),
    };

    // https://www.al-roomi.org/benchmarks/unconstrained/n-dimensions/184-salomon-s-function
    let salomon_function = Benchmark {
        name: "Salomon Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            let sum_sq_sqrt = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
            1.0 - (2.0 * std::f64::consts::PI * sum_sq_sqrt).cos() 
                + 0.1 * sum_sq_sqrt
        }),
        xmin: -100.0,
        xmax: 100.0,
        reference: "R. Salomon, Re-evaluating Genetic Algorithm Performance Under Corodinate Rota- tion of Benchmark Functions: A Survey of Some Theoretical and Practical Aspects of Genetic Algorithms, BioSystems, vol. 39, no. 3, pp. 263-278, 1996.".to_string(),
    };

    let schwefel_1_function = Benchmark {
        name: "Schwefel 1 Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            let alpha = 0.5;
            x.iter().map(|xi| xi * xi).sum::<f64>().powf(alpha)
        }),
        xmin: -100.0,
        xmax: 100.0,
        reference:
            "H. P. Schwefel, Numerical Optimization for Computer Models, John Wiley Sons, 1981."
                .to_string(),
    };

    // https://www.al-roomi.org/benchmarks/unconstrained/n-dimensions/194-step-function-no-3
    let step_3_function = Benchmark {
        name: "Step 3 Function".to_string(),
        func: Box::new(|x: &Vec<f64>| {
            x.iter().map(|xi| xi.powi(2).floor()).sum()
        }),
        xmin: -100.0,
        xmax: 100.0,
        reference: "Jamil, M.; Yang, X. A Literature Survey of Benchmark Functions For Global Optimization Problems. CoRR 2013, abs/1308.4008, [1308.4008]. doi:https://doi.org/10.1504/ijmmno.2013.055204.".to_string(),
    };

    vec![
        deb_2_function,
        generalized_paviani_function,
        generalized_price_2_function,
        mishra_1_function,
        // brown_function,
        cosine_mixture_function,
        deb_1_function,
        exponential_function,
        generalized_egg_crate_function,
        generalized_giunta_function,
        griewank_function,
        // mishra_7_function,
        pathological_function,
        pinter_function,
        qing_function,
        ripple_25_function,
        rosenbrock_function,
        salomon_function,
        schumer_steiglitz_function,
        schwefel_1_function,
        sphere_function,
        step_3_function,
    ]
}
