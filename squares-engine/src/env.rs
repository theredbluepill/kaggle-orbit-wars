use serde::{Deserialize, Serialize};

use crate::{evaluate, GeometryReport, GeometryStatus, Pose};

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Action {
    SetPose {
        index: usize,
        pose: Pose,
    },
    Translate {
        indices: Vec<usize>,
        dx: f64,
        dy: f64,
    },
    Rotate {
        index: usize,
        delta: f64,
    },
    ScaleCenters {
        factor: f64,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Observation {
    pub squares: Vec<Pose>,
    pub steps: u64,
    pub max_steps: u64,
    pub done: bool,
    pub geometry: GeometryReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct StepResult {
    pub accepted: bool,
    pub reward: f64,
    pub observation: Observation,
}

/// Deterministic, cloneable environment. No policy or learning framework is required.
#[derive(Clone, Debug)]
pub struct Engine {
    squares: Vec<Pose>,
    geometry: GeometryReport,
    steps: u64,
    max_steps: u64,
}

impl Engine {
    pub fn reset(n: usize, max_steps: u64) -> Result<Self, String> {
        if !(1..=1000).contains(&n) {
            return Err("n must be in 1..=1000".into());
        }
        let width = (n as f64).sqrt().ceil() as usize;
        let squares = (0..n)
            .map(|i| Pose {
                x: (i % width) as f64 + 0.5,
                y: (i / width) as f64 + 0.5,
                angle: 0.0,
            })
            .collect();
        Self::from_poses(squares, max_steps)
    }

    pub fn from_poses(squares: Vec<Pose>, max_steps: u64) -> Result<Self, String> {
        if max_steps == 0 {
            return Err("max_steps must be positive".into());
        }
        let geometry = evaluate(&squares)?;
        if geometry.status == GeometryStatus::Overlapping {
            return Err("initial squares overlap in the floating-point screen".into());
        }
        Ok(Self {
            squares,
            geometry,
            steps: 0,
            max_steps,
        })
    }

    pub fn observe(&self) -> Observation {
        Observation {
            squares: self.squares.clone(),
            steps: self.steps,
            max_steps: self.max_steps,
            done: self.steps == self.max_steps,
            geometry: self.geometry.clone(),
        }
    }

    /// Invalid API calls do not mutate state. Screen-detected collisions consume
    /// one turn and earn -1, preserving the poses. Uncertain contacts remain provisional.
    pub fn step(&mut self, action: Action) -> Result<StepResult, String> {
        if self.steps == self.max_steps {
            return Err("episode is done; reset before stepping".into());
        }
        let mut candidate = self.squares.clone();
        match action {
            Action::SetPose { index, pose } => {
                *candidate
                    .get_mut(index)
                    .ok_or("square index out of range")? = pose;
            },
            Action::Translate { indices, dx, dy } => {
                if !dx.is_finite() || !dy.is_finite() || indices.is_empty() {
                    return Err("translation needs finite offsets and nonempty indices".into());
                }
                let mut seen = vec![false; candidate.len()];
                for index in indices {
                    let used = seen.get_mut(index).ok_or("square index out of range")?;
                    if *used {
                        return Err("translation indices must be unique".into());
                    }
                    *used = true;
                    candidate[index].x += dx;
                    candidate[index].y += dy;
                }
            },
            Action::Rotate { index, delta } => {
                if !delta.is_finite() {
                    return Err("rotation delta must be finite".into());
                }
                candidate
                    .get_mut(index)
                    .ok_or("square index out of range")?
                    .angle += delta;
            },
            Action::ScaleCenters { factor } => {
                if !factor.is_finite() || factor <= 0.0 {
                    return Err("center scale factor must be finite and positive".into());
                }
                let origin = candidate[0];
                for square in &mut candidate {
                    square.x = origin.x + (square.x - origin.x) * factor;
                    square.y = origin.y + (square.y - origin.y) * factor;
                }
            },
        }
        let geometry = evaluate(&candidate)?;
        let accepted = geometry.status != GeometryStatus::Overlapping;
        let reward = if accepted {
            let gain = self.geometry.score_estimate - geometry.score_estimate;
            self.squares = candidate;
            self.geometry = geometry;
            gain
        } else {
            -1.0
        };
        self.steps += 1;
        Ok(StepResult {
            accepted,
            reward,
            observation: self.observe(),
        })
    }
}
