use serde::Deserialize;
use serde_json::{Value, json};
use smart_buildings_engine::dynamics::{Action, Config, Forcing, Simulation};
use std::io::{self, BufRead, Write};

#[derive(Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
enum Request {
    Init { config: Box<Config> },
    Step { action: Action, forcing: Forcing },
    Reset,
}

fn dispatch(request: Request, simulation: &mut Option<Simulation>) -> Result<Value, String> {
    match request {
        Request::Init { config } => {
            *simulation = Some(Simulation::new(*config)?);
            Ok(json!({"ok": true}))
        },
        Request::Step { action, forcing } => {
            let sim = simulation
                .as_mut()
                .ok_or("initialize the simulator first")?;
            serde_json::to_value(sim.step(&action, &forcing)?).map_err(|e| e.to_string())
        },
        Request::Reset => {
            simulation
                .as_mut()
                .ok_or("initialize the simulator first")?
                .reset()?;
            Ok(json!({"ok": true}))
        },
    }
}

fn main() -> io::Result<()> {
    let mut simulation = None;
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let response = serde_json::from_str(&line?)
            .map_err(|e| e.to_string())
            .and_then(|request| dispatch(request, &mut simulation))
            .unwrap_or_else(|error| json!({"error": error}));
        serde_json::to_writer(&mut stdout, &response)?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}
