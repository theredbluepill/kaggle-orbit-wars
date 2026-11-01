use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, SQRT_2};

use squares_engine::{evaluate, Action, Engine, GeometryStatus, Observation, Pose};

fn pose(x: f64, y: f64, angle: f64) -> Pose {
    Pose { x, y, angle }
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-10,
        "actual={actual:.16}, expected={expected:.16}"
    );
}

fn same_state(actual: &Observation, expected: &Observation) {
    assert_eq!(actual.squares, expected.squares);
    assert_eq!(actual.steps, expected.steps);
    assert_eq!(actual.max_steps, expected.max_steps);
    assert_eq!(actual.done, expected.done);
    assert_eq!(actual.geometry.status, expected.geometry.status);
    close(actual.geometry.side, expected.geometry.side);
    close(
        actual.geometry.score_estimate,
        expected.geometry.score_estimate,
    );
    assert_eq!(actual.geometry.certified, expected.geometry.certified);
}

#[test]
fn grid_reset_covers_the_competition_size_boundary() {
    for (n, side) in [(1, 1.0), (2, 2.0), (4, 2.0), (5, 3.0), (1000, 32.0)] {
        let env = Engine::reset(n, 10).unwrap();
        let state = env.observe();
        assert_eq!(state.squares.len(), n);
        assert_eq!(state.geometry.n, n);
        assert_eq!(state.steps, 0);
        assert!(!state.done);
        assert!(!state.geometry.certified);
        close(state.geometry.side, side);
        close(
            state.geometry.score_estimate,
            side / (n as f64).sqrt() - 1.0,
        );
    }
    assert!(Engine::reset(0, 10).is_err());
    assert!(Engine::reset(1001, 10).is_err());
    assert!(Engine::reset(1, 0).is_err());
}

#[test]
fn rotated_single_square_has_analytic_enclosing_side() {
    let report = evaluate(&[pose(123.0, -89.0, FRAC_PI_4)]).unwrap();
    assert_eq!(report.status, GeometryStatus::Clear);
    assert_eq!(report.candidate_pairs, 0);
    assert!(!report.certified);
    close(report.side, SQRT_2);
    close(report.score_estimate, SQRT_2 - 1.0);
}

#[test]
fn touching_and_sub_tolerance_contacts_remain_uncertified() {
    for x in [1.0, 1.0 - 1e-9, 1.0 + 1e-9] {
        let report = evaluate(&[pose(0.0, 0.0, 0.0), pose(x, 0.0, 0.0)]).unwrap();
        assert_eq!(report.status, GeometryStatus::Uncertain, "x={x}");
        assert_eq!(report.uncertain_pairs, 1);
        assert_eq!(report.overlapping_pairs, 0);
        assert!(!report.certified);
    }
}

#[test]
fn definite_axis_aligned_gap_and_overlap_are_distinguished() {
    let separated = evaluate(&[pose(0.0, 0.0, 0.0), pose(1.0 + 1e-6, 0.0, 0.0)]).unwrap();
    assert_eq!(separated.status, GeometryStatus::Clear);
    assert_eq!(separated.overlapping_pairs, 0);
    let overlapping = evaluate(&[pose(0.0, 0.0, 0.0), pose(1.0 - 1e-6, 0.0, 0.0)]).unwrap();
    assert_eq!(overlapping.status, GeometryStatus::Overlapping);
    assert_eq!(overlapping.overlapping_pairs, 1);
}

#[test]
fn sat_separates_rotated_squares_even_when_their_aabbs_overlap() {
    let report = evaluate(&[pose(0.0, 0.0, FRAC_PI_4), pose(1.0, 1.0, FRAC_PI_4)]).unwrap();
    assert_eq!(report.candidate_pairs, 1);
    assert_eq!(report.status, GeometryStatus::Clear);
    assert_eq!(report.uncertain_pairs, 0);
    close(report.side, 1.0 + SQRT_2);
}

#[test]
fn sat_detects_rotated_overlap_and_keeps_rotated_contacts_uncertain() {
    let overlap = evaluate(&[pose(0.0, 0.0, FRAC_PI_4), pose(1.0, 0.0, FRAC_PI_4)]).unwrap();
    assert_eq!(overlap.status, GeometryStatus::Overlapping);
    let contact = evaluate(&[
        pose(0.0, 0.0, FRAC_PI_4),
        pose(FRAC_PI_4.cos(), FRAC_PI_4.sin(), FRAC_PI_4),
    ])
    .unwrap();
    assert_eq!(contact.status, GeometryStatus::Uncertain);
    assert!(!contact.certified);
}

#[test]
fn invalid_geometry_is_rejected_at_the_public_boundary() {
    assert!(evaluate(&[]).is_err());
    assert!(evaluate(&vec![pose(0.0, 0.0, 0.0); 1001]).is_err());
    for invalid in [
        pose(f64::NAN, 0.0, 0.0),
        pose(0.0, f64::INFINITY, 0.0),
        pose(0.0, 0.0, f64::NEG_INFINITY),
        pose(10_000.1, 0.0, 0.0),
        pose(0.0, -10_000.1, 0.0),
        pose(0.0, 0.0, 100.1),
    ] {
        assert!(evaluate(&[invalid]).is_err(), "accepted {invalid:?}");
    }
    assert!(evaluate(&[pose(10_000.0, -10_000.0, 100.0)]).is_ok());
}

#[test]
fn load_rejects_proven_overlap_but_retains_contact_uncertainty() {
    assert!(Engine::from_poses(vec![pose(0.0, 0.0, 0.0), pose(0.5, 0.0, 0.0)], 2).is_err());
    let env = Engine::from_poses(vec![pose(0.0, 0.0, 0.0), pose(1.0, 0.0, 0.0)], 2).unwrap();
    assert_eq!(env.observe().geometry.status, GeometryStatus::Uncertain);
    assert!(!env.observe().geometry.certified);
    assert!(Engine::from_poses(vec![pose(0.0, 0.0, 0.0)], 0).is_err());
}

#[test]
fn reward_tracks_enclosing_score_change_and_collision_consumes_a_turn() {
    let mut env = Engine::from_poses(vec![pose(0.0, 0.0, 0.0), pose(2.0, 0.0, 0.0)], 2).unwrap();
    let moved = env
        .step(Action::Translate {
            indices: vec![1],
            dx: -0.5,
            dy: 0.0,
        })
        .unwrap();
    assert!(moved.accepted);
    close(moved.observation.geometry.side, 2.5);
    close(moved.reward, 0.5 / SQRT_2);
    assert_eq!(moved.observation.steps, 1);
    assert!(!moved.observation.done);

    let rejected = env
        .step(Action::SetPose {
            index: 1,
            pose: pose(0.2, 0.0, 0.0),
        })
        .unwrap();
    assert!(!rejected.accepted);
    assert_eq!(rejected.reward, -1.0);
    assert_eq!(rejected.observation.squares, moved.observation.squares);
    assert_eq!(rejected.observation.steps, 2);
    assert!(rejected.observation.done);

    let terminal = env.observe();
    assert!(env
        .step(Action::Rotate {
            index: 0,
            delta: 0.1
        })
        .is_err());
    same_state(&env.observe(), &terminal);
}

#[test]
fn invalid_actions_roll_back_every_field_and_do_not_consume_turns() {
    let mut env = Engine::from_poses(vec![pose(0.0, 0.0, 0.0), pose(2.0, 0.0, 0.0)], 50).unwrap();
    let invalid = [
        Action::SetPose {
            index: 2,
            pose: pose(3.0, 0.0, 0.0),
        },
        Action::SetPose {
            index: 0,
            pose: pose(f64::NAN, 0.0, 0.0),
        },
        Action::Translate {
            indices: vec![],
            dx: 1.0,
            dy: 0.0,
        },
        Action::Translate {
            indices: vec![0, 0],
            dx: 1.0,
            dy: 0.0,
        },
        Action::Translate {
            indices: vec![0, 2],
            dx: 1.0,
            dy: 0.0,
        },
        Action::Translate {
            indices: vec![0],
            dx: f64::INFINITY,
            dy: 0.0,
        },
        Action::Translate {
            indices: vec![0],
            dx: 10_001.0,
            dy: 0.0,
        },
        Action::Rotate {
            index: 2,
            delta: 0.0,
        },
        Action::Rotate {
            index: 0,
            delta: f64::NAN,
        },
        Action::Rotate {
            index: 0,
            delta: 101.0,
        },
        Action::ScaleCenters { factor: 0.0 },
        Action::ScaleCenters { factor: -1.0 },
        Action::ScaleCenters {
            factor: f64::INFINITY,
        },
    ];
    for action in invalid {
        let before = env.observe();
        assert!(env.step(action).is_err());
        same_state(&env.observe(), &before);
    }
}

#[test]
fn group_translation_preserves_geometry_and_yields_zero_reward() {
    let mut env = Engine::from_poses(vec![pose(0.0, 0.0, 0.2), pose(3.0, 0.5, -0.3)], 3).unwrap();
    let before = env.observe();
    let moved = env
        .step(Action::Translate {
            indices: vec![0, 1],
            dx: 7.25,
            dy: -3.5,
        })
        .unwrap();
    assert!(moved.accepted);
    close(moved.reward, 0.0);
    close(moved.observation.geometry.side, before.geometry.side);
    for (old, new) in before.squares.iter().zip(&moved.observation.squares) {
        close(new.x, old.x + 7.25);
        close(new.y, old.y - 3.5);
        assert_eq!(new.angle, old.angle);
    }
}

#[test]
fn rotate_is_about_the_selected_center_and_uses_radians() {
    let mut env = Engine::from_poses(vec![pose(4.0, -2.0, 0.0)], 2).unwrap();
    let result = env
        .step(Action::Rotate {
            index: 0,
            delta: FRAC_PI_4,
        })
        .unwrap();
    assert!(result.accepted);
    assert_eq!(result.observation.squares, vec![pose(4.0, -2.0, FRAC_PI_4)]);
    close(result.observation.geometry.side, SQRT_2);
    close(result.reward, 1.0 - SQRT_2);
}

#[test]
fn scaling_changes_centers_around_the_first_pose_but_not_square_sizes() {
    let mut env = Engine::from_poses(vec![pose(5.0, 7.0, 0.0), pose(7.0, 7.0, 0.0)], 3).unwrap();
    let result = env.step(Action::ScaleCenters { factor: 0.75 }).unwrap();
    assert!(result.accepted);
    assert_eq!(
        result.observation.squares,
        vec![pose(5.0, 7.0, 0.0), pose(6.5, 7.0, 0.0)]
    );
    close(result.observation.geometry.side, 2.5);
    close(result.reward, 0.5 / SQRT_2);

    let collision = env.step(Action::ScaleCenters { factor: 0.1 }).unwrap();
    assert!(!collision.accepted);
    assert_eq!(collision.observation.squares, result.observation.squares);

    let mut single = Engine::from_poses(vec![pose(5.0, 7.0, 0.0)], 1).unwrap();
    let result = single.step(Action::ScaleCenters { factor: 0.1 }).unwrap();
    assert!(result.accepted);
    close(result.observation.geometry.side, 1.0);
}

#[test]
fn cloned_states_are_independent_and_identical_action_replays_are_deterministic() {
    let original = Engine::from_poses(vec![pose(0.0, 0.0, 0.0), pose(3.0, 0.0, 0.0)], 20).unwrap();
    let mut left = original.clone();
    let mut right = original.clone();
    for dx in [-0.25, 0.5, -0.125, -0.25] {
        let a = left
            .step(Action::Translate {
                indices: vec![1],
                dx,
                dy: 0.0,
            })
            .unwrap();
        let b = right
            .step(Action::Translate {
                indices: vec![1],
                dx,
                dy: 0.0,
            })
            .unwrap();
        assert_eq!(a.accepted, b.accepted);
        assert_eq!(a.reward, b.reward);
        same_state(&a.observation, &b.observation);
    }
    let initial = original.observe();
    assert_eq!(initial.steps, 0);
    assert_eq!(initial.squares[1], pose(3.0, 0.0, 0.0));
    assert_ne!(left.observe().squares, initial.squares);
}

#[test]
fn generated_packings_preserve_score_under_rigid_symmetries_and_permutation() {
    let mut seed = 0x622d_78a5_15f2_0771_u64;
    let mut random = || {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (seed >> 11) as f64 / ((1_u64 << 53) as f64)
    };
    for _ in 0..50 {
        let original: Vec<_> = (0..8)
            .map(|i| {
                pose(
                    (i % 4) as f64 * 3.0,
                    (i / 4) as f64 * 3.0 + random() * 0.1,
                    random(),
                )
            })
            .collect();
        let report = evaluate(&original).unwrap();
        assert_eq!(report.status, GeometryStatus::Clear);
        let mut transformed: Vec<_> = original
            .iter()
            .map(|sq| pose(5000.0 - sq.y, -4000.0 + sq.x, sq.angle + FRAC_PI_2))
            .collect();
        transformed.reverse();
        let changed = evaluate(&transformed).unwrap();
        assert_eq!(changed.status, report.status);
        close(changed.side, report.side);
        close(changed.score_estimate, report.score_estimate);
        assert!(!changed.certified);
    }
}
