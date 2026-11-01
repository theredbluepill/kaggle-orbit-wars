use std::io::{self, BufRead, Write};

use serde::Deserialize;
use serde_json::{json, Value};
use squares_engine::{evaluate, Action, Engine, Pose};

#[derive(Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Reset { n: usize, max_steps: u64 },
    Load { squares: Vec<Pose>, max_steps: u64 },
    Observe {},
    Step { action: Action },
    Evaluate { squares: Vec<Pose> },
}

fn handle(request: Request, state: &mut Option<Engine>) -> Result<Value, String> {
    match request {
        Request::Reset { n, max_steps } => {
            let engine = Engine::reset(n, max_steps)?;
            let observation = engine.observe();
            *state = Some(engine);
            Ok(json!(observation))
        },
        Request::Load { squares, max_steps } => {
            let engine = Engine::from_poses(squares, max_steps)?;
            let observation = engine.observe();
            *state = Some(engine);
            Ok(json!(observation))
        },
        Request::Observe {} => Ok(json!(state
            .as_ref()
            .ok_or("reset or load first")?
            .observe())),
        Request::Step { action } => Ok(json!(state
            .as_mut()
            .ok_or("reset or load first")?
            .step(action)?)),
        Request::Evaluate { squares } => Ok(json!(evaluate(&squares)?)),
    }
}

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    let mut state = None;
    for line in stdin.lock().lines() {
        let line = line?;
        let response = serde_json::from_str::<Request>(&line)
            .map_err(|error| error.to_string())
            .and_then(|request| handle(request, &mut state));
        let output = match response {
            Ok(result) => json!({"ok": true, "result": result}),
            Err(error) => json!({"ok": false, "error": error}),
        };
        writeln!(stdout, "{output}")?;
        stdout.flush()?;
    }
    Ok(())
}
