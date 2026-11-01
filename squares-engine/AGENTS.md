# Standalone CPU packing engine

This crate adds Squares in Squares to the OWL fork. Its numerical code, typed
reset/step API and JSONL transport do not depend on OWL's Python extension,
training libraries, nightly toolchain or GPU code.

For changes scoped to this crate, use its independent preparation workflow from
the parent Squares repository (so rustup does not select OWL's training toolchain):

```sh
cargo fmt --manifest-path vendor/orbit-wars/squares-engine/Cargo.toml -- --check
cargo clippy --manifest-path vendor/orbit-wars/squares-engine/Cargo.toml --all-targets --locked -- -D warnings
cargo test --manifest-path vendor/orbit-wars/squares-engine/Cargo.toml --locked
.venv/bin/python scripts/verify_engine.py
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Within this fork alone, use the same Cargo commands with manifest path
`squares-engine/Cargo.toml` from a directory running a stable Rust toolchain.
The official-oracle differential suite lives in the parent Squares repository.
This scope-specific workflow replaces the root training package's `just prepare`
for standalone packing changes. Changes to Orbit Wars itself still follow the
root workflow and its parity documentation.

Keep error rollback, finite/bounded input validation, radians, deterministic
replay, and the explicit distinction between floating screening and official
decimal/interval verification. Never mark uncertain contacts certified.
Update this crate's README and tests with behavior changes; retain Cargo.lock.
