//! CPU packing environment following OWL's typed rules-engine boundary.
//!
//! Geometry is a floating-point screen, not a feasibility certificate. Pass exported
//! decimal poses to the pinned official evaluator before promoting a packing.

mod env;
mod geometry;

pub use env::{Action, Engine, Observation, StepResult};
pub use geometry::{evaluate, GeometryReport, GeometryStatus};

use serde::{Deserialize, Serialize};

/// Unit-square center and angle in radians, matching Kaggle's submission units.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pose {
    pub x: f64,
    pub y: f64,
    pub angle: f64,
}

impl Pose {
    pub(crate) fn validate(self) -> Result<(), String> {
        if !self.x.is_finite() || !self.y.is_finite() || !self.angle.is_finite() {
            return Err("coordinates and angle must be finite".into());
        }
        if self.x.abs() > 1e4 || self.y.abs() > 1e4 {
            return Err("coordinates must have absolute value <= 10000".into());
        }
        if self.angle.abs() > 100.0 {
            return Err("angle must have absolute value <= 100 radians".into());
        }
        Ok(())
    }
}
