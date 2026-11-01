# Standalone Smart Buildings engine

This crate ports the pinned Smart Buildings simulator's timestep to native Rust.
The parent project retains the official Python oracle. Python prepares geometry
and exogenous inputs; Rust advances temperatures, interior mass and HVAC state.
Preserve the original OWL package and its training interfaces.

For changes confined to this crate, the standalone checks replace OWL's root
`just rs-prepare` / `just prepare`, which require its unrelated training stack:

```sh
cargo fmt --manifest-path Cargo.toml -- --check
cargo clippy --manifest-path Cargo.toml --all-targets --locked -- -D warnings
cargo test --manifest-path Cargo.toml --locked
```

Also run the parent project's independent Python differential verifier for
behavior changes. Report its exact coverage and numerical tolerance; a local
check does not establish a Kaggle score. Keep dependencies/lockfiles managed by
Cargo. Simulator-derived code retains Google's Apache-2.0 attribution.
