# Smart Buildings native physics engine

Standalone CPU port of the timestep in smart-buildings-2026/starter-kit at
`95cea5b1c57b5291f44baf0283e0a081d53f5b16`. Derived simulator code is Apache-2.0;
see LICENSE and NOTICE. The surrounding OWL package is unchanged and retains MIT.

`cargo +stable build --release --locked` builds the library and JSONL executable.
The parent repository supplies the official scenario/Gym adapter and differential
verifier; run its `scripts/verify_engine.py` after behavior changes.

The library exposes `dynamics::{Config,Action,Forcing,Simulation,StepOutput}` and
`thermal::{ThermalConfig,ThermalSolver,SparseMatrix}`. Initialize from exported
geometry/matrix coefficients, then step with physical actions and action-independent
weather, calendar, occupancy and ordered convection swaps. `Simulation` is cloneable;
clones share immutable thermal factorization and keep their own dynamic state.

The executable reads one request and writes one reply per line:

* `{"command":"init","config":...}` initializes or replaces a simulator.
* `{"command":"step","action":...,"forcing":...}` advances it and returns StepOutput.
* `{"command":"reset"}` restores the initial configuration and dynamic state.

Typed serde structs in the library define each payload. Invalid requests return
`{"error":"..."}` and do not terminate the transport. Reset uses its configured
thermostat snapshot; the Python adapter re-exports actual official reset state
to preserve that simulator's retained thermostat history.

Python-free geometry, weather/occupancy RNG, observations/rewards, rendering,
training and batched collection are not implemented by this crate. No throughput
or learned-policy result is claimed.
