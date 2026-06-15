# Model Predictive Contouring Control for Autonomous Drone Racing

> **Time-optimal-ish path following for a racing quadrotor. Track the geometry, optimise the progress.**

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)
![acados](https://img.shields.io/badge/acados-SQP%20%2F%20HPIPM-00897b)
![CasADi](https://img.shields.io/badge/CasADi-symbolic%20dynamics-ff6f00)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)
![Course](https://img.shields.io/badge/TUM-Autonomous%20Drone%20Racing-0065BD)

> **Note — this is a fork.** This repository is a fork of the
> [LSY Drone Racing](https://github.com/learnsyslab/lsy_drone_racing) course framework
> for the **Autonomous Drone Racing Project Course** at the Technical University of Munich
> (Prof. Dr. Angela P. Schoellig). The upstream framework (simulator, environments,
> difficulty levels, deployment tooling) is unchanged. **My contribution is the single file
> [`lsy_drone_racing/control/mpcc_controller.py`](lsy_drone_racing/control/mpcc_controller.py)**
> — the MPCC controller, its augmented acados model, and the warm-started B-spline planner it
> queries. Everything else in this repo belongs to the upstream project and its authors.

---

## Demo

<div align="center">
  <img width="800" src="docs/img/race.gif"/>
  <br/>
  <sub><sup style="font-size: 0.8em;">Racing environment — powered by <a href="https://github.com/learnsyslab/crazyflow">Crazyflow</a></sup></sub>
</div>

---

## Overview

Drone racing rewards flying *fast through gates*, not *tracking a clock*. A classic
reference-tracking MPC chases a time-parameterised trajectory `r(t)`: if the drone falls
behind — fighting a disturbance, or simply asked to go faster than it can — the reference
"runs away" and the tracking error explodes, which forces a reference governor / nearest-tick
machinery to keep things stable.

**Model Predictive Contouring Control (MPCC)** removes the clock. The controller follows the
*geometric path* and is free to choose how fast to advance along it. The model is augmented
with a **progress state** `θ` (arc length along the path) and its speed `v_θ`; the cost
penalises the **contouring error** (perpendicular distance to the path) and the **lag error**
(longitudinal), while **rewarding progress** at a target speed. The optimiser trades these off
on its own: hold the line tightly near gates, and push `v_θ` on the straights.

**Why MPCC for racing:**

| | Reference-tracking MPC | **MPCC (this work)** |
|---|---|---|
| Reference | Time-parameterised `r(t)` | Geometry only `p(θ)` |
| "Reference runs away" | Possible → needs a governor | **Impossible** — no clock |
| Speed | Fixed by the trajectory | **Chosen by the optimiser** (`v_θ` reward) |
| Behaviour near gates | Tracks whatever `r(t)` says | Naturally slows to hold the contour |

The path itself comes from a lightweight **B-spline planner** that is **re-planned in the
background** whenever a gate or obstacle pose is revealed, so the controller works across the
course's progressive difficulty levels (randomised inertia, moving gates/obstacles).

---

## Project Status

The **MPCC controller** is fully implemented and runs in the Crazyflow simulator. A learned
weight-adaptation layer (RL) is the planned next step.

| Component | Status |
|---|---|
| Augmented acados model (drone + progress double-integrator `[θ, v_θ]`) | ✅ Done |
| NONLINEAR_LS contouring / lag / progress cost | ✅ Done |
| Warm-started B-spline `SimplePlanner` (geometric-path API by arc length) | ✅ Done |
| Background re-planning on revealed gate / obstacle poses | ✅ Done |
| Convex per-stage keep-out half-planes (obstacles + non-target gates) | ✅ Done |
| Receding-horizon primal warm start (shifted previous solution) | ✅ Done |
| Non-uniform shooting grid (look further ahead at ~no extra cost) | ✅ Done |
| Jump-free path projection (local, forward-biased `θ` window) | ✅ Done |
| Live visualisation overlay (path, gates, obstacles, progress marker) | ✅ Done |
| **RL layer that adapts the MPCC cost weights in-flight** | 🚧 Planned |
| Quantitative evaluation (lap time vs. tracking-MPC baseline) | 🚧 Planned |

---

## Method

### Conceptual Pipeline

```
   Gate / obstacle poses (revealed online)
                    |
                    v
        +---------------------------+
        |  SimplePlanner            |   B-spline through gate waypoints,
        |  (warm-started, replans)  |   shaped by scipy.minimize to avoid
        +---------------------------+   obstacles & gate frames
                    |
                    |  geometric path  p(θ), tangent t(θ)   (queried by arc length)
                    v
        +---------------------------+
        |  MPCC  (acados SQP)       |   state augmented with [θ, v_θ];
        |  contouring + lag + prog. |   per-stage path point/tangent as params
        +---------------------------+
                    |
                    v
        Collective-thrust + body-rate command  ->  drone
```

### Augmented Model

The physical drone (12 states / 4 inputs, from `drone_models.so_rpy`) is augmented with a
**progress double-integrator**:

```
   θ̇   = v_θ            θ      : arc-length progress along the path
   v̇_θ = a_θ            v_θ    : progress speed       (state)
                        a_θ    : progress accel.      (extra input)
```

So the optimiser controls *how fast it moves along the path* as a first-class decision
variable. State `x = [pos, rpy, vel, drpy, θ, v_θ]` (14), input `u = [rpy_cmd, thrust, a_θ]` (5).

### Cost (NONLINEAR_LS, Gauss-Newton friendly)

At each shooting node the *frozen* per-stage path point `p_ref` and unit tangent `t_ref`
define the error split:

```
   d   = pos - p_ref
   e_l = t_ref · d                 lag  (longitudinal, signed)
   e_c = d - e_l · t_ref           contouring (perpendicular, 3-vector)
   e_v = v_θ - v_target            progress-speed tracking  (the "reward")
```

| Cost term | Weight | Role |
|---|---|---|
| **Contouring** `e_c` | `q_c = 50` | Stay *on* the path (dominant — keeps the racing line) |
| **Lag** `e_l` | `q_l = 5` | Don't fall behind the progress point |
| **Progress** `e_v` | `q_v = 20` | Push `v_θ` toward the target speed (advance eagerly) |
| **Attitude / rates** | `q_att, q_dr` | Smooth, flyable orientation |
| **Input** `rpy_cmd, thrust, a_θ` | `r_*` | Regularise effort |

`q_c ≫ q_l` keeps it contouring; the `q_l / q_v` balance sets *how eagerly* it advances.

### Collision Avoidance

Obstacles and non-target gates are handled as **convex per-stage keep-out half-planes**
`aₓ·pₓ + a_y·p_y − b ≥ 0`, re-linearised every tick (SCP): the normal `a` points from the
keep-out centre to the *predicted* drone position and `b` makes the line tangent to the
keep-out circle. Linear ⇒ the QP stays convex and solves fast. The **target gate is disabled**
each tick (we fly through it on the path); all other gate frames stay active — so even a U-turn
back toward a frame is caught. Constraints are **soft** (slack-penalised) so a transient never
makes the QP infeasible.

---

## Technical Features

- **Progress-augmented acados OCP** — `[θ, v_θ]` states + `a_θ` input; CasADi symbolic drone
  dynamics, `FULL_CONDENSING_HPIPM` + Gauss-Newton SQP.
- **Geometric-path planner API** — `SimplePlanner.path_point_tangent(θ)` maps arc length to
  `(point, unit-tangent)` via a cached arc-length LUT; `project_to_theta` snaps the drone onto
  the path.
- **Jump-free projection** — the projection is restricted to a *local, forward-biased* `θ`
  window around last tick's value, so where the path passes near itself (gate-and-back, U-turns)
  it can never teleport onto the wrong branch and "shortcut".
- **Receding-horizon primal warm start** — the previous solution is shifted one node forward and
  fed back in, so SQP converges in a couple of iterations instead of rebuilding the trajectory.
- **Non-uniform shooting grid** — intervals grow toward the horizon end, reaching ~1.1 s ahead
  with the same `N` nodes; cost-scaling fixed so no single node dominates.
- **Background re-planning** — a `ThreadPoolExecutor` rebuilds the full-track path off the
  control thread when a gate/obstacle moves, so the 50 Hz loop never stalls.
- **Warm-started B-spline planning** — gate waypoints + freely-shifted intermediates optimised
  by L-BFGS-B (obstacle + gate-frame + deviation cost), the previous solution seeding the next.
- **Graceful degradation** — failed solves hold the last command for a few ticks, then brake to
  hover, rather than crashing.
- **Live overlay** — planned path (green), gates (yellow + cyan/white frames), obstacle poles
  (orange), drone progress marker `θ` (red), and the far end of the horizon (blue).

---

## Planned: RL-Tuned MPCC Weights

The MPCC cost weights (`q_c, q_l, q_v, …`) are currently **static**. The planned next layer is a
**reinforcement-learning policy that adapts these weights in-flight, per tick**, from the drone's
state — learning *when* and *how* to retune to push racing performance (e.g. tighten contouring
near gates, favour progress on straights).

<div align="center">
  <img width="640" src="figures/rl_blockdiagram.png"/>
</div>

**Design intent** (not yet implemented):

- **Hook point:** in `compute_control`, just before `solve()`: `features → policy → new W →
  cost_set(stage, "W", W_new) → solve`. acados updates `W` without a rebuild; the planner is
  untouched (clean split: planner = global, MPCC = local, RL = tuning).
- **Action space:** bounded *multipliers* on the baseline weights (not absolute weights — too
  fragile), kept in range by an activation at the policy output.
- **Observation:** tracking/contouring error, speed, distance & angle to the next gate,
  remaining thrust margin, plus angle & distance to the nearest obstacle.
- **Key risk:** solver robustness — extreme weights stiffen the QP. Bounded actions are
  near-mandatory, since failed solves fall back to the last command and poison the learning signal.
- **Training:** offline in simulation (reward: gate-pass time, crashes, tracking error), policy
  frozen at deployment. Lit. pointers: RL-tuned MPC, Gros & Zanon, differentiable MPC (Amos).

> Status: **planned, not started.** Listed here for completeness of the project roadmap.

---

## Repository Structure

> Only `mpcc_controller.py` is my own work; the rest is the upstream LSY framework.

```
lsy_drone_racing/
|
+-- lsy_drone_racing/
|   +-- control/
|   |   +-- mpcc_controller.py        # ★ THIS WORK — MPCC controller + SimplePlanner
|   |   +-- mpc_planner_controller.py # (upstream-adjacent) reference-tracking MPC + helpers reused here
|   |   +-- controller.py             # base Controller interface (upstream)
|   +-- ...                           # envs, wrappers, utils (upstream)
|
+-- config/                           # level0..3 + sim2real task definitions (upstream)
+-- scripts/                          # sim.py / deploy.py entry points (upstream)
+-- figures/                          # diagrams (path planning, MPC/RL block diagrams, speed profile)
+-- docs/                             # course documentation (upstream)
+-- acados/  c_generated_code/        # acados install + generated solver code
+-- pyproject.toml  pixi.lock
+-- README.md
```

The MPCC controller imports a few helpers (`_Cylinder`, `_GateFrame`, slack constants) from
`mpc_planner_controller.py` to share the obstacle/gate-frame geometry, but is otherwise
self-contained: its own augmented model, OCP, planner, and visualisation.

---

## Installation

This fork keeps the upstream environment. Use the project's `pixi` environment:

```bash
git clone https://github.com/mangam-02/lsy_drone_racing.git
cd lsy_drone_racing
pixi shell --frozen
```

See the upstream
[official documentation](https://lsy-drone-racing.readthedocs.io/en/latest/getting_started/general.html)
for full setup, including the `acados` build. On Apple Silicon the repo's bundled Linux `acados`
must be rebuilt for arm64 — see the project notes.

---

## Usage

### Simulation

Run the MPCC controller in the simulator on any difficulty level:

```bash
python scripts/sim.py --controller mpcc_controller.py --config level2.toml
```

The competition environment uses **level 2** (randomised inertia + moving gates/obstacles,
re-planning required), which is what the background-replanning path planner is built for.

### Real hardware

```bash
python scripts/deploy.py --controller mpcc_controller.py --config level0.toml
```

---

## Implementation Details

| Component | Choice | Rationale |
|---|---|---|
| Path following | MPCC (contouring + lag + progress) | No clock → reference can't run away; speed chosen by optimiser |
| Progress model | `θ`, `v_θ` double-integrator with input `a_θ` | Makes "how fast along the path" a decision variable |
| Cost type | `NONLINEAR_LS`, Gauss-Newton | Residual is linear in the vars given frozen path params → fast, reliable |
| QP solver | `FULL_CONDENSING_HPIPM`, SQP | Small dense QP, warm-startable |
| Obstacle / gate avoidance | Convex per-stage half-planes (SCP), soft | Convex QP, re-linearised each tick; transient overspeed never infeasible |
| Warm start | Shifted previous solution (primal) + per-stage `θ` seed | SQP converges in a couple of iterations |
| Shooting grid | Non-uniform (growing intervals), dt-weighted cost | Look ~1.1 s ahead at ~no extra cost; no node dominates |
| Path planner | Weighted cubic B-spline + L-BFGS-B waypoint shift | Smooth, fly-through gates, obstacle/frame-aware, warm-started |
| Projection | Local forward-biased arc-length window | `θ` advances continuously, never snaps onto an old branch |
| Re-planning | Background thread on revealed pose change | 50 Hz control loop never stalls |
| Framework | acados + CasADi + scipy | Symbolic dynamics, fast embedded NLP, lightweight planning |

---

## Future Work

- **RL-tuned MPCC weights** (see above) — the immediate next milestone.
- **Quantitative evaluation** — lap time, gate-pass rate, and solve time vs. the
  reference-tracking MPC baseline across levels 0–3.
- **Time-optimal `v_target`** — schedule the progress target by path curvature instead of a
  constant cruise speed.
- **sim2real transfer** — deployment on the physical Crazyflie with the level-2/sim2real configs.

---

## Acknowledgements & Credits

- **Course framework:** [LSY Drone Racing](https://github.com/learnsyslab/lsy_drone_racing) and
  [Crazyflow](https://github.com/learnsyslab/crazyflow), by the
  [Learning Systems Lab (LSY)](https://www.ce.cit.tum.de/lsy/home/), TUM. All upstream code,
  simulator, environments, and tooling are theirs.
- **My contribution:** the MPCC controller and its planner in
  [`lsy_drone_racing/control/mpcc_controller.py`](lsy_drone_racing/control/mpcc_controller.py).

## References

1. Lam, D., Manzie, C., Good, M. **Model predictive contouring control.** CDC, 2010.
2. Liniger, A., Domahidi, A., Morari, M. **Optimization-based autonomous racing of 1:43 scale RC cars.** Optimal Control Applications and Methods, 2015.
3. Romero, A. et al. **Model Predictive Contouring Control for Time-Optimal Quadrotor Flight.** IEEE T-RO, 2022.
4. Verschueren, R. et al. **acados — a modular open-source framework for fast embedded optimal control.** Mathematical Programming Computation, 2022.
5. Gros, S., Zanon, M. **Data-driven economic NMPC using reinforcement learning.** IEEE TAC, 2020.

---

*Project for the **Autonomous Drone Racing Project Course** (SS26), Prof. Dr. Angela P. Schoellig, Technical University of Munich (TUM). Built on the LSY Drone Racing framework. This is a fork; the MPCC controller is the author's own contribution.*
