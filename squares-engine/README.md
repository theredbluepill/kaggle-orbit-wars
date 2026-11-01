# Squares in Squares CPU engine

A standalone deterministic packing environment in the OWL fork. It follows the
original `src/rules_engine/` separation of typed state, actions, reset and step.
It has no dependency on the Orbit Wars simulator, PyO3, PyTorch, CUDA or a policy.

## Build and check

From this fork's root, explicitly select stable Rust instead of OWL's training toolchain:

```sh
cargo +stable build --release --locked --manifest-path squares-engine/Cargo.toml
cargo +stable test --locked --manifest-path squares-engine/Cargo.toml
cargo +stable clippy --all-targets --locked --manifest-path squares-engine/Cargo.toml -- -D warnings
cargo +stable fmt --manifest-path squares-engine/Cargo.toml -- --check
```

The parent [Squares-in-Squares repository](https://github.com/theredbluepill/squares-in-squares)
pins this fork as a submodule and supplies `scripts/verify_engine.py`, a retained official
metric snapshot, deterministic differential fixtures and a verification receipt.
That check builds this crate alone and needs only CPU Python packages for the oracle.

## Rust API

`Pose { x, y, angle }` describes a unit square's center and rotation **in radians**.
`Engine::reset(n, max_steps)` builds the trivial grid. `Engine::from_poses` loads
a supplied arrangement. `observe` returns poses, step count, horizon, terminal flag
and geometry diagnostics. `Clone` creates an independent deterministic branch.

`step(Action)` supports:

- `SetPose { index, pose }`: replace one pose.
- `Translate { indices, dx, dy }`: move a nonempty set of distinct squares.
- `Rotate { index, delta }`: rotate one square around its center, in radians.
- `ScaleCenters { factor }`: scale centers around square zero; square side lengths stay one.

Every square and action is checked for finite values and valid indices. Supported
puzzles have 1–1,000 squares, coordinates with absolute value at most 10,000, and
angles with absolute value at most 100 radians, matching the pinned metric's limits.
The horizon must be positive. Invalid requests return an error without changing
poses or the step count. Loading a new state is transactional too.

A move with a definite float-screen overlap is rejected: it consumes one step,
keeps the previous poses and earns reward `-1`. Other moves are provisionally accepted
and earn the previous enclosure score minus the new one. All moves consume a step;
after `max_steps`, another step errors until reset. There is no artificial opponent
or hidden state, and no training is performed.

## Geometry and certification boundary

The enclosing side is translation invariant: the larger of the x/y corner spans.
The provisional enclosure score is `max(0, L / sqrt(n) - 1)`.
An x-axis sweep with padded bounding boxes finds potential neighbors; separating-axis
tests on both squares' axes classify them as `clear`, `overlapping`, or `uncertain`.
The broad-phase padding is `1e-7`, and the float SAT tolerance is `1e-8`.

**Every report has `certified: false`.** These statuses describe floating-point
screening. Exact grid contacts and tiny gaps/overlaps can be `uncertain`.
Such states may enter search, but may not be promoted as valid submissions without
the independent official decimal/interval evaluator. `score_estimate` is an
enclosure-only proxy; an invalid arrangement does not earn it on Kaggle. Preserve
original decimal strings for that evaluator: binary64 can merge distinct decimals.

## JSONL process interface

Run `squares-engine/target/release/squares-engine` and send one JSON object per line:

```json
{"command":"reset","n":4,"max_steps":10}
{"command":"step","action":{"type":"translate","indices":[3],"dx":1.0,"dy":0.0}}
{"command":"observe"}
{"command":"evaluate","squares":[{"x":0.5,"y":0.5,"angle":0.0}]}
{"command":"load","max_steps":10,"squares":[{"x":0.5,"y":0.5,"angle":0.0}]}
```

Replies are `{"ok":true,"result":...}` or `{"ok":false,"error":"..."}`.
Unknown fields, malformed numbers and invalid commands error without changing the
environment. The process continues after a bad request. `evaluate` needs no active
episode and never modifies one. JSON geometry output includes side, score estimate,
pair counts and its explicit certification flag.

## Attribution and limits

Architecture adapted from Isaiah Pressman's MIT-licensed
[kaggle-orbit-wars](https://github.com/IsaiahPressman/kaggle-orbit-wars), pinned at
`32b3ec900ad406eedd965f53a1a0f4490d31c589` before this addition. The upstream
MIT notice is retained in this crate and at the fork root. This crate implements
packing-specific geometry and episode semantics; it does not port the game's physics.
Metric definitions and screening thresholds follow the public
[Kaggle metric](https://www.kaggle.com/code/metric/squares-in-squares/output).

This engine supplies repeatable CPU search transitions. It does not yet implement
an optimizer, a policy learner or a Kaggle submission client. Passing its finite tests
does not prove universal geometry parity or a competition score.
