use serde::Serialize;

use crate::Pose;

const SAT_TOLERANCE: f64 = 1e-8;
const AABB_PAD: f64 = 1e-7;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryStatus {
    Clear,
    Overlapping,
    Uncertain,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GeometryReport {
    pub n: usize,
    pub side: f64,
    /// Enclosure-only proxy; invalid or unresolved packings do not earn this score.
    pub score_estimate: f64,
    pub status: GeometryStatus,
    pub candidate_pairs: usize,
    pub overlapping_pairs: usize,
    pub uncertain_pairs: usize,
    /// Always false: the official decimal/interval checker owns certification.
    pub certified: bool,
}

#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }
}

struct Square {
    corners: [Point; 4],
    axes: [Point; 2],
    bounds: [f64; 4],
}

impl Square {
    fn new(pose: Pose, origin: Pose) -> Self {
        let (s, c) = pose.angle.rem_euclid(std::f64::consts::TAU).sin_cos();
        let cx = pose.x - origin.x;
        let cy = pose.y - origin.y;
        let corners = [(-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)].map(|(x, y)| Point {
            x: cx + x * c - y * s,
            y: cy + x * s + y * c,
        });
        let bounds = [
            corners.iter().map(|p| p.x).fold(f64::INFINITY, f64::min),
            corners.iter().map(|p| p.y).fold(f64::INFINITY, f64::min),
            corners
                .iter()
                .map(|p| p.x)
                .fold(f64::NEG_INFINITY, f64::max),
            corners
                .iter()
                .map(|p| p.y)
                .fold(f64::NEG_INFINITY, f64::max),
        ];
        Self {
            corners,
            axes: [Point { x: c, y: s }, Point { x: -s, y: c }],
            bounds,
        }
    }

    fn project(&self, axis: Point) -> (f64, f64) {
        self.corners
            .iter()
            .map(|p| p.dot(axis))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), value| {
                (lo.min(value), hi.max(value))
            })
    }

    fn relation(&self, other: &Self) -> GeometryStatus {
        let mut min_overlap = f64::INFINITY;
        for axis in self.axes.iter().chain(other.axes.iter()) {
            let (a_min, a_max) = self.project(*axis);
            let (b_min, b_max) = other.project(*axis);
            let overlap = a_max.min(b_max) - a_min.max(b_min);
            if overlap < -SAT_TOLERANCE {
                return GeometryStatus::Clear;
            }
            min_overlap = min_overlap.min(overlap);
        }
        if min_overlap > SAT_TOLERANCE {
            GeometryStatus::Overlapping
        } else {
            GeometryStatus::Uncertain
        }
    }
}

/// Translation-invariant bounding side and broad-phase/SAT diagnostics.
/// Near contacts are deliberately unresolved, including exact grid contacts.
pub fn evaluate(poses: &[Pose]) -> Result<GeometryReport, String> {
    if poses.is_empty() || poses.len() > 1000 {
        return Err("a puzzle must contain 1..=1000 unit squares".into());
    }
    for pose in poses {
        pose.validate()?;
    }
    let squares: Vec<_> = poses.iter().map(|p| Square::new(*p, poses[0])).collect();
    let min_x = squares
        .iter()
        .map(|s| s.bounds[0])
        .fold(f64::INFINITY, f64::min);
    let min_y = squares
        .iter()
        .map(|s| s.bounds[1])
        .fold(f64::INFINITY, f64::min);
    let max_x = squares
        .iter()
        .map(|s| s.bounds[2])
        .fold(f64::NEG_INFINITY, f64::max);
    let max_y = squares
        .iter()
        .map(|s| s.bounds[3])
        .fold(f64::NEG_INFINITY, f64::max);
    let side = (max_x - min_x).max(max_y - min_y);
    let mut report = GeometryReport {
        n: poses.len(),
        side,
        score_estimate: (side / (poses.len() as f64).sqrt() - 1.0).max(0.0),
        status: GeometryStatus::Clear,
        candidate_pairs: 0,
        overlapping_pairs: 0,
        uncertain_pairs: 0,
        certified: false,
    };
    let mut order: Vec<_> = (0..squares.len()).collect();
    order.sort_by(|a, b| squares[*a].bounds[0].total_cmp(&squares[*b].bounds[0]));
    for (offset, &i) in order.iter().enumerate() {
        let a = &squares[i];
        for &j in &order[offset + 1..] {
            let b = &squares[j];
            if b.bounds[0] > a.bounds[2] + 2.0 * AABB_PAD {
                break;
            }
            if b.bounds[1] > a.bounds[3] + 2.0 * AABB_PAD
                || a.bounds[1] > b.bounds[3] + 2.0 * AABB_PAD
            {
                continue;
            }
            report.candidate_pairs += 1;
            match a.relation(b) {
                GeometryStatus::Clear => {},
                GeometryStatus::Overlapping => report.overlapping_pairs += 1,
                GeometryStatus::Uncertain => report.uncertain_pairs += 1,
            }
        }
    }
    report.status = if report.overlapping_pairs > 0 {
        GeometryStatus::Overlapping
    } else if report.uncertain_pairs > 0 {
        GeometryStatus::Uncertain
    } else {
        GeometryStatus::Clear
    };
    Ok(report)
}
