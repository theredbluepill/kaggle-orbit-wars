//! Native transition for the pinned official flexible-floor-plan simulator.
//! Scenario generation, calendar/occupancy inputs and convection RNG are supplied by the adapter.

use crate::thermal::{ThermalConfig, ThermalSolver};
use serde::{Deserialize, Serialize};

const AIR_CAPACITY: f64 = 1006.0 * 1.2;
const WATER_CAPACITY: f64 = 4180.0 * 1000.0;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZoneConfig {
    pub cells: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AhuConfig {
    pub fan_efficiency: f64,
    pub max_air_flow_rate: f64,
    pub min_static_pressure: f64,
    pub max_recirculation: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VavConfig {
    pub ahu: usize,
    pub zones: Vec<usize>,
    pub thermal_column: Option<usize>,
    pub max_air_flow_rate: f64,
    pub max_air_flow_static_pressure: f64,
    pub reheat_max_water_flow_factor: f64,
    pub min_damper_setting: f64,
    pub max_discharge_air_temperature: f64,
    pub initial_thermostat_mode: u8,
    pub initial_previous_comfort: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaterConfig {
    pub header_resistance: f64,
    pub heating_rate: f64,
    pub cooling_rate: f64,
    pub has_tank: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub thermal: ThermalConfig,
    pub initial_temperature: Vec<f64>,
    pub initial_input_q: Vec<f64>,
    pub initial_return_water_temperature: f64,
    pub zones: Vec<ZoneConfig>,
    pub ahus: Vec<AhuConfig>,
    pub vavs: Vec<VavConfig>,
    pub water: WaterConfig,
    pub cv_area_m2: f64,
    pub internal_gain_w_per_m2: f64,
    pub internal_gain_standby_fraction: f64,
    pub occupant_sensible_heat_w: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AhuAction {
    pub run_command: bool,
    pub supply_air_temperature_setpoint: f64,
    pub fan_static_pressure: f64,
    pub recirculation: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VavAction {
    pub damper_command: Option<f64>,
    pub reheat_valve_command: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaterAction {
    pub run_command: bool,
    pub supply_water_temperature_setpoint: f64,
    pub differential_pressure_setpoint: f64,
    // Python's observation getter advances the boiler. Import its current state at the boundary.
    pub current_temperature: f64,
    pub step_tank_temperature_change: f64,
    pub sensor_elapsed_seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Action {
    pub ahus: Vec<AhuAction>,
    pub vavs: Vec<VavAction>,
    pub water: WaterAction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Forcing {
    pub ambient_temperature: f64,
    pub convection_coefficient: f64,
    pub comfort_mode: bool,
    pub heating_setpoint: f64,
    pub cooling_setpoint: f64,
    pub occupancy: Vec<f64>,
    // Sequential swaps from the official action-independent random stream.
    pub convection_swaps: Vec<[usize; 2]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VavState {
    pub zone_air_temperature: f64,
    pub heating_setpoint: f64,
    pub cooling_setpoint: f64,
    pub damper_setting: f64,
    pub reheat_valve_setting: f64,
    pub supply_air_flow_ratio: f64,
    pub flow_rate_demand: f64,
    pub thermostat_mode: u8,
    pub previous_comfort: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AhuState {
    // Raw accumulator, including the 1e-5 minimum VAV demand when the fan is off.
    pub air_flow_rate: f64,
    pub cooling_request_count: usize,
    pub recirculation: f64,
    pub supply_air_temperature_setpoint: f64,
    pub fan_static_pressure: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaterState {
    pub current_temperature: f64,
    pub step_tank_temperature_change: f64,
    pub last_step_duration_seconds: f64,
    pub return_water_temperature: f64,
    pub flow_factor_sum: f64,
    pub heating_request_count: usize,
    pub total_flow_rate: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepOutput {
    pub temperature: Vec<f64>,
    pub input_q: Vec<f64>,
    pub interior_mass_temperature: Vec<f64>,
    pub vavs: Vec<VavState>,
    pub ahus: Vec<AhuState>,
    pub water: WaterState,
    pub clipped: bool,
}

#[derive(Clone)]
pub struct Simulation {
    config: Config,
    solver: ThermalSolver,
    temperature: Vec<f64>,
    input_q: Vec<f64>,
    vavs: Vec<VavState>,
    return_water_temperature: f64,
    areas: Vec<f64>,
}

fn finite(values: impl IntoIterator<Item = f64>) -> bool {
    values.into_iter().all(f64::is_finite)
}

impl Simulation {
    pub fn new(config: Config) -> Result<Self, String> {
        let n = config.initial_temperature.len();
        if n == 0
            || config.initial_input_q.len() != n
            || !finite(config.initial_temperature.iter().copied())
            || !finite(config.initial_input_q.iter().copied())
            || !finite([
                config.initial_return_water_temperature,
                config.cv_area_m2,
                config.internal_gain_w_per_m2,
                config.internal_gain_standby_fraction,
                config.occupant_sensible_heat_w,
                config.water.header_resistance,
                config.water.heating_rate,
                config.water.cooling_rate,
            ])
            || config.cv_area_m2 <= 0.0
            || config.water.header_resistance < 0.0
        {
            return Err("invalid initial field or gain/water configuration".into());
        }
        let mut occupied = vec![false; n];
        for zone in &config.zones {
            if zone.cells.is_empty() {
                return Err("physical zone must contain cells".into());
            }
            for &cell in &zone.cells {
                if cell >= n || occupied[cell] {
                    return Err("zone cells must be in range and disjoint".into());
                }
                occupied[cell] = true;
            }
        }
        for ahu in &config.ahus {
            if !finite([
                ahu.fan_efficiency,
                ahu.max_air_flow_rate,
                ahu.min_static_pressure,
                ahu.max_recirculation,
            ]) || ahu.fan_efficiency <= 0.0
                || ahu.max_air_flow_rate <= 0.0
                || ahu.min_static_pressure < 0.0
                || !(0.0..=1.0).contains(&ahu.max_recirculation)
            {
                return Err("invalid AHU configuration".into());
            }
        }
        let mut solver = ThermalSolver::new(config.thermal.clone())?;
        solver.reset(&config.initial_temperature)?;
        let mut columns = vec![false; solver.n_terms()];
        let mut areas = Vec::new();
        for vav in &config.vavs {
            if vav.ahu >= config.ahus.len()
                || vav.zones.iter().any(|&z| z >= config.zones.len())
                || !finite([
                    vav.max_air_flow_rate,
                    vav.max_air_flow_static_pressure,
                    vav.reheat_max_water_flow_factor,
                    vav.min_damper_setting,
                    vav.max_discharge_air_temperature,
                ])
                || vav.max_air_flow_rate < 0.0
                || vav.max_air_flow_static_pressure <= 0.0
                || vav.reheat_max_water_flow_factor < 0.0
                || !(0.0..=1.0).contains(&vav.min_damper_setting)
                || vav.initial_thermostat_mode > 3
            {
                return Err("invalid VAV configuration".into());
            }
            let area = vav
                .zones
                .iter()
                .map(|&z| config.zones[z].cells.len() as f64)
                .sum();
            if let Some(column) = vav.thermal_column {
                if column >= columns.len() || columns[column] || vav.zones.is_empty() {
                    return Err(
                        "VAV thermal column must be unique and service physical zones".into(),
                    );
                }
                columns[column] = true;
            } else if !vav.zones.is_empty() {
                return Err("a VAV serving physical zones requires a thermal column".into());
            }
            areas.push(area);
        }
        if columns.iter().any(|&seen| !seen) {
            return Err("every thermal column requires one VAV".into());
        }
        let mut simulation = Self {
            temperature: config.initial_temperature.clone(),
            input_q: config.initial_input_q.clone(),
            return_water_temperature: config.initial_return_water_temperature,
            config,
            solver,
            vavs: Vec::new(),
            areas,
        };
        simulation.reset()?;
        Ok(simulation)
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.temperature
            .clone_from(&self.config.initial_temperature);
        self.input_q.clone_from(&self.config.initial_input_q);
        self.solver.reset(&self.temperature)?;
        self.return_water_temperature = self.config.initial_return_water_temperature;
        self.vavs = self
            .config
            .vavs
            .iter()
            .enumerate()
            .map(|(i, vav)| VavState {
                zone_air_temperature: if self.areas[i] > 0.0 {
                    mean_zones(&self.temperature, &self.config.zones, &vav.zones)
                } else {
                    0.0
                },
                heating_setpoint: 294.0,
                cooling_setpoint: 297.0,
                damper_setting: vav.min_damper_setting,
                reheat_valve_setting: 0.0,
                supply_air_flow_ratio: 1.0,
                flow_rate_demand: 0.0,
                thermostat_mode: vav.initial_thermostat_mode,
                previous_comfort: vav.initial_previous_comfort,
            })
            .collect();
        Ok(())
    }

    fn validate_step(&self, action: &Action, forcing: &Forcing) -> Result<(), String> {
        if action.ahus.len() != self.config.ahus.len()
            || action.vavs.len() != self.config.vavs.len()
            || forcing.occupancy.len() != self.config.zones.len()
        {
            return Err("action/occupancy dimensions must match the building".into());
        }
        if !finite([
            forcing.ambient_temperature,
            forcing.convection_coefficient,
            forcing.heating_setpoint,
            forcing.cooling_setpoint,
            action.water.supply_water_temperature_setpoint,
            action.water.differential_pressure_setpoint,
            action.water.current_temperature,
            action.water.step_tank_temperature_change,
            action.water.sensor_elapsed_seconds,
        ]) || forcing.convection_coefficient < 0.0
            || forcing.heating_setpoint > forcing.cooling_setpoint
            || action.water.differential_pressure_setpoint < 0.0
            || forcing.occupancy.iter().any(|&x| !x.is_finite() || x < 0.0)
            || forcing
                .convection_swaps
                .iter()
                .flatten()
                .any(|&i| i >= self.temperature.len())
        {
            return Err("invalid forcing or water state".into());
        }
        for ahu in &action.ahus {
            if !finite([
                ahu.supply_air_temperature_setpoint,
                ahu.fan_static_pressure,
                ahu.recirculation,
            ]) || ahu.fan_static_pressure < 0.0
            {
                return Err("invalid AHU action".into());
            }
        }
        for vav in &action.vavs {
            if !finite(
                vav.damper_command
                    .into_iter()
                    .chain(vav.reheat_valve_command),
            ) {
                return Err("VAV commands must be finite".into());
            }
        }
        Ok(())
    }

    pub fn step(&mut self, action: &Action, forcing: &Forcing) -> Result<StepOutput, String> {
        self.validate_step(action, forcing)?;
        let config = &self.config;
        let mut vavs = self.vavs.clone();
        let mut ahus: Vec<AhuState> = action
            .ahus
            .iter()
            .zip(&config.ahus)
            .map(|(a, c)| AhuState {
                air_flow_rate: 0.0,
                cooling_request_count: 0,
                recirculation: a.recirculation.clamp(0.0, c.max_recirculation),
                supply_air_temperature_setpoint: a.supply_air_temperature_setpoint,
                fan_static_pressure: a.fan_static_pressure,
            })
            .collect();
        let mut recirculation_sums = vec![0.0; ahus.len()];
        let mut recirculation_areas = vec![0.0; ahus.len()];
        for (i, ((vav, c), a)) in vavs
            .iter_mut()
            .zip(&config.vavs)
            .zip(&action.vavs)
            .enumerate()
        {
            if self.areas[i] == 0.0 {
                continue;
            }
            vav.zone_air_temperature = mean_zones(&self.temperature, &config.zones, &c.zones);
            update_thermostat(vav, forcing);
            vav.heating_setpoint = forcing.heating_setpoint;
            vav.cooling_setpoint = forcing.cooling_setpoint;
            let (damper, reheat) = match vav.thermostat_mode {
                1 => (1.0, 1.0),
                2 => (1.0, 0.0),
                _ => (c.min_damper_setting, 0.0),
            };
            vav.damper_setting = a
                .damper_command
                .unwrap_or(damper)
                .clamp(0.0, 1.0)
                .max(c.min_damper_setting);
            vav.reheat_valve_setting = a.reheat_valve_command.unwrap_or(reheat).clamp(0.0, 1.0);
            recirculation_sums[c.ahu] += vav.zone_air_temperature * self.areas[i];
            recirculation_areas[c.ahu] += self.areas[i];
        }
        let fallback = self.temperature.iter().sum::<f64>() / self.temperature.len() as f64;
        let supply: Vec<f64> = ahus
            .iter()
            .enumerate()
            .map(|(i, ahu)| {
                let recirculation_temp = if recirculation_areas[i] > 0.0 {
                    recirculation_sums[i] / recirculation_areas[i]
                } else {
                    fallback
                };
                let mixed = ahu.recirculation * recirculation_temp
                    + (1.0 - ahu.recirculation) * forcing.ambient_temperature;
                if action.ahus[i].run_command && mixed > ahu.supply_air_temperature_setpoint {
                    ahu.supply_air_temperature_setpoint
                } else {
                    mixed
                }
            })
            .collect();

        let flow_factor_sum = vavs
            .iter()
            .zip(&config.vavs)
            .map(|(v, c)| v.reheat_valve_setting * c.reheat_max_water_flow_factor)
            .sum::<f64>();
        let pressure = if action.water.run_command {
            action.water.differential_pressure_setpoint
        } else {
            0.0
        };
        let water_flow = flow_factor_sum
            * (pressure / (1.0 + flow_factor_sum.powi(2) * config.water.header_resistance)).sqrt()
            / 3600.0;

        let mut demands = vec![0.0; ahus.len()];
        for (vav, c) in vavs.iter_mut().zip(&config.vavs) {
            let pressure = if action.ahus[c.ahu].run_command {
                ahus[c.ahu]
                    .fan_static_pressure
                    .max(config.ahus[c.ahu].min_static_pressure)
            } else {
                0.0
            };
            vav.flow_rate_demand = (vav.damper_setting
                * c.max_air_flow_rate
                * (pressure / c.max_air_flow_static_pressure).sqrt())
            .max(0.00001);
            demands[c.ahu] += vav.flow_rate_demand;
        }
        for (vav, c) in vavs.iter_mut().zip(&config.vavs) {
            vav.supply_air_flow_ratio =
                (config.ahus[c.ahu].max_air_flow_rate / demands[c.ahu]).min(1.0);
            vav.flow_rate_demand *= vav.supply_air_flow_ratio;
        }

        let mut water = WaterState {
            current_temperature: action.water.current_temperature,
            step_tank_temperature_change: action.water.step_tank_temperature_change,
            last_step_duration_seconds: action.water.sensor_elapsed_seconds,
            return_water_temperature: self.return_water_temperature,
            flow_factor_sum,
            heating_request_count: vavs.len(),
            total_flow_rate: water_flow,
        };
        let mut zone_q = vec![0.0; config.zones.len()];
        let mut zone_written = vec![false; zone_q.len()];
        let mut air_g = vec![0.0; self.solver.n_terms()];
        let mut air_supply = vec![0.0; self.solver.n_terms()];
        let mut total_reheat = 0.0;
        let mut entering_air = Vec::new();
        // Official coupling columns are registered in VAV iteration order.
        for (i, (vav, c)) in vavs.iter().zip(&config.vavs).enumerate() {
            let Some(column) = c.thermal_column else {
                continue;
            };
            let supply_temp = supply[c.ahu];
            air_supply[column] = supply_temp;
            read_water_sensor(&mut water, &action.water, &config.water);
            let reheat_flow = if flow_factor_sum == 0.0 {
                0.0
            } else {
                water_flow
                    * (vav.reheat_valve_setting * c.reheat_max_water_flow_factor / flow_factor_sum)
            };
            let reheat = reheat_power(
                vav.flow_rate_demand,
                reheat_flow,
                supply_temp,
                water.current_temperature,
                c.max_discharge_air_temperature,
            );
            total_reheat += reheat;
            entering_air.push(supply_temp);
            if vav.damper_setting > 0.0 && c.max_air_flow_rate > 0.0 {
                air_g[column] = vav.flow_rate_demand * AIR_CAPACITY;
                for &zone in &c.zones {
                    zone_q[zone] += reheat * config.zones[zone].cells.len() as f64 / self.areas[i];
                    zone_written[zone] = true;
                }
            }
            if vav.flow_rate_demand > 0.0 {
                ahus[c.ahu].air_flow_rate = (ahus[c.ahu].air_flow_rate + vav.flow_rate_demand)
                    .min(config.ahus[c.ahu].max_air_flow_rate);
                ahus[c.ahu].cooling_request_count += 1;
            }
        }
        if config.internal_gain_w_per_m2 > 0.0 {
            let fraction = if forcing.comfort_mode {
                1.0
            } else {
                config.internal_gain_standby_fraction
            };
            let per_cell = config.internal_gain_w_per_m2 * config.cv_area_m2 * fraction;
            for (i, zone) in config.zones.iter().enumerate() {
                zone_q[i] += per_cell * zone.cells.len() as f64;
                zone_written[i] = true;
            }
        }
        if config.occupant_sensible_heat_w > 0.0 {
            for (i, &occupants) in forcing.occupancy.iter().enumerate() {
                zone_q[i] += occupants * config.occupant_sensible_heat_w;
                zone_written[i] = true;
            }
        }
        let mut input_q = self.input_q.clone();
        assign_zone_q(&mut input_q, &config.zones, &zone_q, &zone_written);
        let mut temperature = self.solver.solve(
            &self.temperature,
            &input_q,
            forcing.ambient_temperature,
            forcing.convection_coefficient,
            Some(&air_g),
            Some(&air_supply),
        )?;
        for (i, c) in config.vavs.iter().enumerate() {
            if let Some(column) = c.thermal_column {
                let sensed = mean_zones(&temperature, &config.zones, &c.zones);
                let air_q = air_g[column] * (air_supply[column] - sensed);
                for &zone in &c.zones {
                    zone_q[zone] += air_q * config.zones[zone].cells.len() as f64 / self.areas[i];
                    zone_written[zone] = true;
                }
            }
        }
        assign_zone_q(&mut input_q, &config.zones, &zone_q, &zone_written);
        for &[a, b] in &forcing.convection_swaps {
            temperature.swap(a, b);
        }
        if water_flow * WATER_CAPACITY > 0.0 && !entering_air.is_empty() {
            read_water_sensor(&mut water, &action.water, &config.water);
            let return_temp =
                water.current_temperature - total_reheat / (WATER_CAPACITY * water_flow);
            let air_extreme = if total_reheat >= 0.0 {
                entering_air.iter().copied().fold(f64::INFINITY, f64::min)
            } else {
                entering_air
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
            };
            water.return_water_temperature = return_temp.clamp(
                water.current_temperature.min(air_extreme),
                water.current_temperature.max(air_extreme),
            );
        }
        let clipped = temperature.iter().any(|&x| !(150.0..=500.0).contains(&x));
        if clipped {
            for t in &mut temperature {
                *t = t.clamp(150.0, 500.0);
            }
        }
        self.temperature = temperature.clone();
        self.input_q = input_q.clone();
        self.vavs = vavs.clone();
        self.return_water_temperature = water.return_water_temperature;
        Ok(StepOutput {
            temperature,
            input_q,
            interior_mass_temperature: self.solver.mass_temp().to_vec(),
            vavs,
            ahus,
            water,
            clipped,
        })
    }
}

fn mean_zones(temperature: &[f64], zones: &[ZoneConfig], assigned: &[usize]) -> f64 {
    let mut total = 0.0;
    let mut count = 0;
    for &zone in assigned {
        let cells = &zones[zone].cells;
        // Preserve the official per-zone mean followed by area-weighted aggregation.
        let mean = cells.iter().map(|&cell| temperature[cell]).sum::<f64>() / cells.len() as f64;
        total += mean * cells.len() as f64;
        count += cells.len();
    }
    total / count as f64
}

fn assign_zone_q(input_q: &mut [f64], zones: &[ZoneConfig], q: &[f64], written: &[bool]) {
    for (i, zone) in zones.iter().enumerate() {
        if written[i] {
            let per_cell = q[i] / zone.cells.len() as f64;
            for &cell in &zone.cells {
                input_q[cell] = per_cell;
            }
        }
    }
}

fn update_thermostat(state: &mut VavState, forcing: &Forcing) {
    let t = state.zone_air_temperature;
    let low = forcing.heating_setpoint;
    let high = forcing.cooling_setpoint;
    if !forcing.comfort_mode && state.previous_comfort == Some(true) {
        state.thermostat_mode = 3;
    } else if !(!forcing.comfort_mode && state.thermostat_mode == 3 && t > low && t < high) {
        let mid = 0.5 * (high - low) + low;
        state.thermostat_mode = if t < low {
            1
        } else if t > high {
            2
        } else if (t < mid && state.thermostat_mode == 1) || (t > mid && state.thermostat_mode == 2)
        {
            state.thermostat_mode
        } else {
            0
        };
    }
    state.previous_comfort = Some(forcing.comfort_mode);
}

fn read_water_sensor(state: &mut WaterState, action: &WaterAction, config: &WaterConfig) {
    if config.has_tank && config.heating_rate > 0.0 && config.cooling_rate > 0.0 {
        let previous = state.current_temperature;
        state.current_temperature = if action.supply_water_temperature_setpoint > previous {
            (previous + config.heating_rate * action.sensor_elapsed_seconds / 60.0)
                .min(action.supply_water_temperature_setpoint)
        } else if action.supply_water_temperature_setpoint < previous {
            (previous - config.cooling_rate * action.sensor_elapsed_seconds / 60.0)
                .max(action.supply_water_temperature_setpoint)
        } else {
            action.supply_water_temperature_setpoint
        };
        state.step_tank_temperature_change = state.current_temperature - previous;
    } else {
        state.current_temperature = action.supply_water_temperature_setpoint;
    }
}

fn reheat_power(
    air_flow: f64,
    water_flow: f64,
    air_temp: f64,
    water_temp: f64,
    max_discharge: f64,
) -> f64 {
    let air_capacity = AIR_CAPACITY * air_flow;
    if air_capacity <= 0.0 {
        return 0.0;
    }
    let mut effectiveness = (WATER_CAPACITY * water_flow / air_capacity).min(1.0);
    let gap = water_temp - air_temp;
    if gap > 0.0 {
        let headroom = max_discharge - air_temp;
        if headroom <= 0.0 {
            return 0.0;
        }
        effectiveness = effectiveness.min(headroom / gap);
    }
    effectiveness * air_capacity * gap
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thermal::SparseMatrix;

    fn two_zone_config() -> Config {
        let terms = SparseMatrix {
            rows: 2,
            cols: 2,
            entries: vec![(0, 0, 1.0), (1, 1, 1.0)],
        };
        Config {
            thermal: ThermalConfig {
                full_size: 2,
                solved: vec![0, 1],
                conduction: SparseMatrix {
                    rows: 2,
                    cols: 2,
                    entries: vec![(0, 0, 10.0), (1, 1, 10.0)],
                },
                capacitance_rate: vec![10.0; 2],
                dirichlet_conductance: vec![0.0; 2],
                mass_coupling: vec![0.0; 2],
                mass_retained: vec![0.0; 2],
                mass_gained: vec![0.0; 2],
                ambient_horizontal_area: vec![0.0; 2],
                ambient_vertical_area: vec![0.0; 2],
                horizontal: vec![1.0; 2],
                vertical: vec![1.0; 2],
                conductivity: vec![1.0; 2],
                applied: terms.clone(),
                sensed: terms,
            },
            initial_temperature: vec![295.0, 298.0],
            initial_input_q: vec![0.0; 2],
            initial_return_water_temperature: 295.0,
            zones: vec![ZoneConfig { cells: vec![0] }, ZoneConfig { cells: vec![1] }],
            ahus: vec![AhuConfig {
                fan_efficiency: 0.9,
                max_air_flow_rate: 1.0,
                min_static_pressure: 200.0,
                max_recirculation: 0.7,
            }],
            vavs: (0..2)
                .map(|zone| VavConfig {
                    ahu: 0,
                    zones: vec![zone],
                    thermal_column: Some(zone),
                    max_air_flow_rate: 2.0,
                    max_air_flow_static_pressure: 2000.0,
                    reheat_max_water_flow_factor: 0.108,
                    min_damper_setting: 0.1,
                    max_discharge_air_temperature: 308.15,
                    initial_thermostat_mode: 0,
                    initial_previous_comfort: None,
                })
                .collect(),
            water: WaterConfig {
                header_resistance: 0.0,
                heating_rate: 0.5,
                cooling_rate: 0.1,
                has_tank: true,
            },
            cv_area_m2: 1.0,
            internal_gain_w_per_m2: 0.0,
            internal_gain_standby_fraction: 0.2,
            occupant_sensible_heat_w: 0.0,
        }
    }

    fn action_and_forcing() -> (Action, Forcing) {
        (
            Action {
                ahus: vec![AhuAction {
                    run_command: true,
                    supply_air_temperature_setpoint: 290.0,
                    fan_static_pressure: 2000.0,
                    recirculation: 0.0,
                }],
                vavs: vec![
                    VavAction {
                        damper_command: Some(1.0),
                        reheat_valve_command: Some(0.0)
                    };
                    2
                ],
                water: WaterAction {
                    run_command: true,
                    supply_water_temperature_setpoint: 340.0,
                    differential_pressure_setpoint: 0.5,
                    current_temperature: 340.0,
                    step_tank_temperature_change: 0.0,
                    sensor_elapsed_seconds: 0.0,
                },
            },
            Forcing {
                ambient_temperature: 300.0,
                convection_coefficient: 0.0,
                comfort_mode: true,
                heating_setpoint: 294.0,
                cooling_setpoint: 297.0,
                occupancy: vec![0.0; 2],
                convection_swaps: vec![[0, 1]],
            },
        )
    }

    #[test]
    fn coupled_transition_caps_flow_preserves_sensors_and_swaps_exactly() {
        let mut simulation = Simulation::new(two_zone_config()).unwrap();
        let (action, forcing) = action_and_forcing();
        let out = simulation.step(&action, &forcing).unwrap();
        let g = 0.5 * AIR_CAPACITY;
        for (i, old) in [298.0, 295.0].into_iter().enumerate() {
            assert!((out.temperature[i] - (10.0 * old + g * 290.0) / (10.0 + g)).abs() < 1e-10);
            assert_eq!(out.vavs[i].flow_rate_demand, 0.5);
            assert_eq!(out.vavs[i].supply_air_flow_ratio, 0.25);
        }
        assert_eq!(out.vavs[0].zone_air_temperature, 295.0);
        assert_eq!(out.vavs[1].zone_air_temperature, 298.0);
        assert_eq!(out.ahus[0].air_flow_rate, 1.0);
        assert_eq!(out.ahus[0].cooling_request_count, 2);
        assert_eq!(out.water.total_flow_rate, 0.0);
        assert_eq!(out.water.return_water_temperature, 295.0);
        assert!(!out.clipped);
    }

    #[test]
    fn reset_clone_and_invalid_input_do_not_corrupt_next_step() {
        let mut simulation = Simulation::new(two_zone_config()).unwrap();
        let mut clone = simulation.clone();
        let (action, forcing) = action_and_forcing();
        let expected = simulation.step(&action, &forcing).unwrap();
        let mut invalid = forcing.clone();
        invalid.convection_swaps = vec![[0, 2]];
        assert!(clone.step(&action, &invalid).is_err());
        let actual = clone.step(&action, &forcing).unwrap();
        assert_eq!(expected.temperature, actual.temperature);
        simulation.reset().unwrap();
        let repeated = simulation.step(&action, &forcing).unwrap();
        assert_eq!(expected.temperature, repeated.temperature);
        assert_eq!(expected.input_q, repeated.input_q);
    }

    #[test]
    fn reheat_respects_water_and_discharge_energy_limits() {
        let q = reheat_power(1.0, 0.1, 290.0, 340.0, 308.15);
        assert!((q - AIR_CAPACITY * 18.15).abs() < 1e-8);
        assert_eq!(reheat_power(1.0, 0.0, 290.0, 340.0, 308.15), 0.0);
        assert_eq!(reheat_power(1.0, 0.1, 310.0, 340.0, 308.15), 0.0);
        assert!(reheat_power(1.0, 0.00001, 310.0, 290.0, 308.15) < 0.0);
    }

    #[test]
    fn boiler_sensor_preserves_signed_elapsed_time_and_repeated_reads() {
        let config = WaterConfig {
            header_resistance: 0.0,
            heating_rate: 0.5,
            cooling_rate: 0.1,
            has_tank: true,
        };
        let mut action = WaterAction {
            run_command: true,
            supply_water_temperature_setpoint: 350.0,
            differential_pressure_setpoint: 0.5,
            current_temperature: 340.0,
            step_tank_temperature_change: 0.0,
            sensor_elapsed_seconds: -300.0,
        };
        let mut state = WaterState {
            current_temperature: 340.0,
            step_tank_temperature_change: 0.0,
            last_step_duration_seconds: -300.0,
            return_water_temperature: 295.0,
            flow_factor_sum: 0.0,
            heating_request_count: 0,
            total_flow_rate: 0.0,
        };
        read_water_sensor(&mut state, &action, &config);
        assert_eq!(state.current_temperature, 337.5);
        action.sensor_elapsed_seconds = 300.0;
        read_water_sensor(&mut state, &action, &config);
        read_water_sensor(&mut state, &action, &config);
        assert_eq!(state.current_temperature, 342.5);
    }
}
