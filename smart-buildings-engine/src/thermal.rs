//! Backward Euler, interior mass elimination and implicit air-side coupling.
//! Equations ported from Google's Smart Buildings Simulator (Apache-2.0) and
//! smart-buildings-2026/starter-kit finite_difference_solver.py.

use std::sync::Arc;

use faer::Mat;
use faer::prelude::Solve;
use faer::sparse::linalg::solvers::Lu;
use faer::sparse::{SparseColMat, Triplet};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

impl SparseMatrix {
    fn validate(&self, name: &str) -> Result<(), String> {
        if self
            .entries
            .iter()
            .any(|&(r, c, v)| r >= self.rows || c >= self.cols || !v.is_finite())
        {
            return Err(format!("{name}: invalid sparse index or nonfinite value"));
        }
        Ok(())
    }

    fn triplets(&self) -> Vec<Triplet<usize, usize, f64>> {
        self.entries
            .iter()
            .map(|&(row, col, val)| Triplet::new(row, col, val))
            .collect()
    }

    fn transpose_mul(&self, vector: &Mat<f64>, rhs: usize) -> Vec<f64> {
        let mut result = vec![0.0; self.cols];
        for &(r, c, v) in &self.entries {
            result[c] += v * vector[(r, rhs)];
        }
        result
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalConfig {
    pub full_size: usize,
    pub solved: Vec<usize>,
    pub conduction: SparseMatrix,
    pub capacitance_rate: Vec<f64>,
    pub dirichlet_conductance: Vec<f64>,
    pub mass_coupling: Vec<f64>,
    pub mass_retained: Vec<f64>,
    pub mass_gained: Vec<f64>,
    pub ambient_horizontal_area: Vec<f64>,
    pub ambient_vertical_area: Vec<f64>,
    pub horizontal: Vec<f64>,
    pub vertical: Vec<f64>,
    pub conductivity: Vec<f64>,
    pub applied: SparseMatrix,
    pub sensed: SparseMatrix,
}

impl ThermalConfig {
    fn validate(&self) -> Result<(), String> {
        let n = self.solved.len();
        if self.full_size == 0
            || n == 0
            || self.solved.iter().any(|&i| i >= self.full_size)
            || self.solved.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(
                "solved must be nonempty, strictly increasing indices into the full field".into(),
            );
        }
        for (name, values, positive) in [
            ("capacitance_rate", &self.capacitance_rate, true),
            ("mass_coupling", &self.mass_coupling, false),
            ("mass_retained", &self.mass_retained, false),
            ("mass_gained", &self.mass_gained, false),
            (
                "ambient_horizontal_area",
                &self.ambient_horizontal_area,
                false,
            ),
            ("ambient_vertical_area", &self.ambient_vertical_area, false),
            ("horizontal", &self.horizontal, true),
            ("vertical", &self.vertical, true),
            ("conductivity", &self.conductivity, true),
        ] {
            if values.len() != n
                || values
                    .iter()
                    .any(|&v| !v.is_finite() || v < 0.0 || (positive && v == 0.0))
            {
                return Err(format!(
                    "{name} must contain {n} finite {} values",
                    if positive { "positive" } else { "nonnegative" }
                ));
            }
        }
        for (name, matrix) in [
            ("conduction", &self.conduction),
            ("applied", &self.applied),
            ("sensed", &self.sensed),
        ] {
            matrix.validate(name)?;
            if matrix.rows != n {
                return Err(format!("{name} must have {n} solved rows"));
            }
        }
        if self.conduction.cols != n || self.applied.cols != self.sensed.cols {
            return Err(
                "conduction must be square; applied and sensed term counts must agree".into(),
            );
        }
        let mut row_scale = vec![0.0; n];
        for &(row, _, value) in &self.conduction.entries {
            row_scale[row] += value.abs();
        }
        // Upstream subtracts two four-face sums; preserve their signed roundoff.
        if self.dirichlet_conductance.len() != n
            || row_scale.iter().any(|v| !v.is_finite())
            || self
                .dirichlet_conductance
                .iter()
                .zip(row_scale)
                .any(|(&v, scale)| !v.is_finite() || v < -8.0 * f64::EPSILON * scale)
        {
            return Err(
                "dirichlet conductance must be nonnegative apart from row-sum roundoff".into(),
            );
        }
        if self
            .applied
            .entries
            .iter()
            .chain(&self.sensed.entries)
            .any(|e| e.2 < 0.0)
        {
            return Err("air-side weights must be nonnegative".into());
        }
        if self
            .mass_retained
            .iter()
            .chain(&self.mass_gained)
            .any(|&v| v > 1.0)
        {
            return Err("mass retained/gained fractions must be at most one".into());
        }
        Ok(())
    }
}

struct Factorization {
    h: f64,
    exposure: Vec<f64>,
    lu: Lu<usize, f64>,
    sensed_response: Mat<f64>,
}

#[derive(Clone)]
pub struct ThermalSolver {
    config: Arc<ThermalConfig>,
    factorization: Option<Arc<Factorization>>,
    mass_temp: Vec<f64>,
}

impl ThermalSolver {
    pub fn new(config: ThermalConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            mass_temp: vec![0.0; config.solved.len()],
            config: Arc::new(config),
            factorization: None,
        })
    }

    pub fn reset(&mut self, temperature: &[f64]) -> Result<(), String> {
        self.validate_field(temperature, "temperature")?;
        for (i, &full) in self.config.solved.iter().enumerate() {
            self.mass_temp[i] = if self.config.mass_coupling[i] > 0.0 {
                temperature[full]
            } else {
                0.0
            };
        }
        Ok(())
    }

    pub fn mass_temp(&self) -> &[f64] {
        &self.mass_temp
    }

    pub fn interior_mass_temp(&self) -> Vec<f64> {
        self.mass_temp
            .iter()
            .zip(&self.config.mass_coupling)
            .filter_map(|(&t, &g)| (g > 0.0).then_some(t))
            .collect()
    }

    pub fn n_terms(&self) -> usize {
        self.config.applied.cols
    }

    fn validate_field(&self, field: &[f64], name: &str) -> Result<(), String> {
        if field.len() != self.config.full_size || field.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "{name} must contain {} finite values",
                self.config.full_size
            ));
        }
        Ok(())
    }

    fn factorize(&self, h: f64) -> Result<Factorization, String> {
        let cfg = &self.config;
        let n = cfg.solved.len();
        let exposure: Vec<_> = (0..n)
            .map(|i| {
                h * cfg.ambient_horizontal_area[i]
                    / (1.0 + h * 0.5 * cfg.horizontal[i] / cfg.conductivity[i])
                    + h * cfg.ambient_vertical_area[i]
                        / (1.0 + h * 0.5 * cfg.vertical[i] / cfg.conductivity[i])
            })
            .collect();
        let mut entries = cfg.conduction.triplets();
        entries.extend(
            exposure
                .iter()
                .enumerate()
                .map(|(i, &v)| Triplet::new(i, i, v)),
        );
        let matrix = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &entries)
            .map_err(|e| format!("thermal matrix: {e:?}"))?;
        let lu = matrix
            .sp_lu()
            .map_err(|e| format!("thermal factorization: {e:?}"))?;
        let k = cfg.applied.cols;
        let mut sensed_response = Mat::zeros(k, k);
        // Block RHS construction bounds temporary storage on the full-size grids.
        for start in (0..k).step_by(8) {
            let width = 8.min(k - start);
            let mut applied = Mat::zeros(n, width);
            for &(r, c, v) in &cfg.applied.entries {
                if c >= start && c < start + width {
                    applied[(r, c - start)] += v;
                }
            }
            let response = lu.solve(applied);
            for rhs in 0..width {
                for (row, value) in cfg
                    .sensed
                    .transpose_mul(&response, rhs)
                    .into_iter()
                    .enumerate()
                {
                    sensed_response[(row, start + rhs)] = value;
                }
            }
        }
        Ok(Factorization {
            h,
            exposure,
            lu,
            sensed_response,
        })
    }

    #[allow(clippy::too_many_arguments)] // Keep the official solver's timestep inputs together.
    pub fn solve(
        &mut self,
        temperature: &[f64],
        input_q: &[f64],
        ambient: f64,
        h: f64,
        air_g: Option<&[f64]>,
        supply: Option<&[f64]>,
    ) -> Result<Vec<f64>, String> {
        self.validate_field(temperature, "temperature")?;
        self.validate_field(input_q, "input_q")?;
        if !ambient.is_finite() || !h.is_finite() || h < 0.0 {
            return Err(
                "ambient must be finite and convection coefficient finite/nonnegative".into(),
            );
        }
        let k = self.n_terms();
        if let Some(g) = air_g {
            let s = supply.ok_or("air-side conductance requires supply temperatures")?;
            if g.len() != k
                || s.len() != k
                || g.iter().any(|&v| !v.is_finite() || v < 0.0)
                || s.iter().any(|v| !v.is_finite())
            {
                return Err(format!(
                    "air-side arrays must contain {k} finite values; conductance must be nonnegative"
                ));
            }
        } else if supply.is_some() {
            return Err("supply temperatures require air-side conductance".into());
        }
        if self.factorization.as_ref().is_none_or(|f| f.h != h) {
            self.factorization = Some(Arc::new(self.factorize(h)?));
        }
        let f = self
            .factorization
            .as_ref()
            .ok_or("factorization unavailable")?;
        let cfg = &self.config;
        let n = cfg.solved.len();
        let rhs = Mat::from_fn(n, 1, |i, _| {
            cfg.capacitance_rate[i] * temperature[cfg.solved[i]]
                + input_q[cfg.solved[i]]
                + ambient * (cfg.dirichlet_conductance[i] + f.exposure[i])
                + cfg.mass_coupling[i] * self.mass_temp[i]
        });
        let mut solved = f.lu.solve(rhs);
        if let Some(g) = air_g {
            let s = supply.ok_or("supply temperatures missing")?;
            if k > 0 {
                let sensed = cfg.sensed.transpose_mul(&solved, 0);
                let small = Mat::from_fn(k, k, |r, c| {
                    f64::from(r == c) + g[r] * f.sensed_response[(r, c)]
                });
                let power_rhs = Mat::from_fn(k, 1, |r, _| g[r] * (s[r] - sensed[r]));
                let power = small.partial_piv_lu().solve(power_rhs);
                let mut applied = Mat::zeros(n, 1);
                for &(r, c, v) in &cfg.applied.entries {
                    applied[(r, 0)] += v * power[(c, 0)];
                }
                let correction = f.lu.solve(applied);
                for i in 0..n {
                    solved[(i, 0)] += correction[(i, 0)];
                }
            }
        }
        let mut updated = vec![ambient; cfg.full_size];
        let mut mass_temp = Vec::with_capacity(n);
        for (i, &full) in cfg.solved.iter().enumerate() {
            let value = solved[(i, 0)];
            let mass = cfg.mass_retained[i] * self.mass_temp[i] + cfg.mass_gained[i] * value;
            if !value.is_finite() || !mass.is_finite() {
                return Err("thermal solve produced a nonfinite temperature".into());
            }
            updated[full] = value;
            mass_temp.push(mass);
        }
        self.mass_temp = mass_temp;
        Ok(updated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ThermalConfig {
        ThermalConfig {
            full_size: 3,
            solved: vec![0, 1],
            conduction: SparseMatrix {
                rows: 2,
                cols: 2,
                entries: vec![(0, 0, 3.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 3.0)],
            },
            capacitance_rate: vec![1.0; 2],
            dirichlet_conductance: vec![1.0; 2],
            mass_coupling: vec![0.0; 2],
            mass_retained: vec![0.0; 2],
            mass_gained: vec![0.0; 2],
            ambient_horizontal_area: vec![0.0; 2],
            ambient_vertical_area: vec![0.0; 2],
            horizontal: vec![1.0; 2],
            vertical: vec![1.0; 2],
            conductivity: vec![1.0; 2],
            applied: SparseMatrix {
                rows: 2,
                cols: 1,
                entries: vec![(0, 0, 1.0)],
            },
            sensed: SparseMatrix {
                rows: 2,
                cols: 1,
                entries: vec![(1, 0, 1.0)],
            },
        }
    }

    #[test]
    fn two_node_step_and_pinned_ambient() {
        let mut solver = ThermalSolver::new(config()).unwrap();
        let actual = solver
            .solve(&[10.0, 20.0, 99.0], &[2.0, 4.0, 0.0], 0.0, 0.0, None, None)
            .unwrap();
        assert!((actual[0] - 7.5).abs() < 1e-12);
        assert!((actual[1] - 10.5).abs() < 1e-12);
        assert_eq!(actual[2], 0.0);
    }

    #[test]
    fn woodbury_matches_direct_nonsymmetric_system() {
        let mut solver = ThermalSolver::new(config()).unwrap();
        let actual = solver
            .solve(
                &[10.0, 20.0, 0.0],
                &[2.0, 4.0, 0.0],
                0.0,
                0.0,
                Some(&[2.0]),
                Some(&[30.0]),
            )
            .unwrap();
        // (A + P G Q^T) = [[3,1],[-1,3]], RHS = [72,24].
        assert!((actual[0] - 19.2).abs() < 1e-12);
        assert!((actual[1] - 14.4).abs() < 1e-12);
    }

    #[test]
    fn mass_reset_clone_and_variable_convection() {
        let mut cfg = config();
        cfg.conduction.entries[0].2 += 2.0;
        cfg.mass_coupling[0] = 2.0;
        cfg.mass_retained[0] = 0.25;
        cfg.mass_gained[0] = 0.75;
        cfg.ambient_horizontal_area[1] = 1.0;
        let mut solver = ThermalSolver::new(cfg).unwrap();
        solver.reset(&[10.0, 20.0, 30.0]).unwrap();
        let mut fork = solver.clone();
        let first = solver
            .solve(&[10.0, 20.0, 30.0], &[0.0; 3], 0.0, 2.0, None, None)
            .unwrap();
        assert!((solver.mass_temp()[0] - (2.5 + 0.75 * first[0])).abs() < 1e-12);
        assert_eq!(fork.mass_temp(), &[10.0, 0.0]);
        let second = fork
            .solve(&[10.0, 20.0, 30.0], &[0.0; 3], 0.0, 0.0, None, None)
            .unwrap();
        assert!(second[1] > first[1]);
        assert_eq!(fork.interior_mass_temp().len(), 1);
    }

    #[test]
    fn invalid_fields_and_dimensions_fail_before_mutation() {
        let mut cfg = config();
        cfg.solved = vec![1, 1];
        assert!(ThermalSolver::new(cfg).is_err());
        let mut solver = ThermalSolver::new(config()).unwrap();
        assert!(
            solver
                .solve(&[0.0; 3], &[0.0; 3], 0.0, 0.0, Some(&[1.0]), None)
                .is_err()
        );
        assert!(
            solver
                .solve(&[0.0; 2], &[0.0; 3], 0.0, 0.0, None, None)
                .is_err()
        );
        assert!(
            solver
                .solve(&[0.0; 3], &[0.0; 3], 0.0, -1.0, None, None)
                .is_err()
        );
    }

    #[test]
    fn signed_row_sum_roundoff_is_retained_but_negative_conductance_is_rejected() {
        let mut cfg = config();
        cfg.dirichlet_conductance[0] = -f64::EPSILON;
        let solver = ThermalSolver::new(cfg.clone()).unwrap();
        assert_eq!(solver.config.dirichlet_conductance[0], -f64::EPSILON);
        cfg.dirichlet_conductance[0] = -1e-6;
        assert!(ThermalSolver::new(cfg).is_err());
    }
}
