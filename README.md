# WALL-E EVE Signal Mission Simulator

A probabilistic grid-world simulator in which WALL-E searches for EVE using
Bayesian belief updates and utility-based decision making.  The agent
navigates a configurable 2-D grid, interprets noisy sensor readings as
Bayesian evidence, manages battery resources safely, and produces a full
mission trace, annotated map, and Markdown report.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Environment Design](#3-environment-design)
4. [Config Format](#4-config-format)
5. [Belief Update Algorithm](#5-belief-update-algorithm)
6. [Agent Decision Algorithm](#6-agent-decision-algorithm)
7. [How to Run](#7-how-to-run)
8. [Sample Output](#8-sample-output)
9. [Advanced Features](#9-advanced-features)
10. [Testing](#10-testing)

---

## 1. Project Overview

WALL-E received a last-known directional signal from EVE before contact was
lost.  The mission: autonomously search the grid, accumulate sensor evidence,
and locate EVE before the battery runs out or the time limit expires.

**What the simulator does:**

- Models the search as a partially observable grid world where EVE's position
  is hidden from the agent.
- Maintains a probabilistic belief map (2-D probability distribution) over
  all possible EVE locations.
- Updates beliefs after every action using Bayesian inference driven by
  directional scan results and physical clue discoveries.
- Selects actions (move N/S/E/W, scan, recharge, end mission) by scoring
  utility and enforcing a hard battery-safety constraint.
- Writes a JSON step-by-step trace, an annotated ASCII map, and a human-
  readable Markdown mission report to the `outputs/` directory.

**Why it exists:**

The project demonstrates core AI concepts — probabilistic reasoning,
Bayesian updates, utility-based planning, and resource-constrained search —
in a self-contained, testable Python codebase that requires no external
simulation framework.

---

## 2. Architecture

```
+-----------------------------------------------------------+
|  run.py  (CLI entrypoint)                                 |
|    |                                                      |
|    +-- load_config()  -->  Config (validated dataclass)   |
|    |                                                      |
|    +-- Environment(config)                                |
|    |      Holds ground truth: EVE pos, clues, obstacles   |
|    |      Exposes: move(), scan(), recharge(), end_mission()|
|    |                                                      |
|    +-- BeliefGrid(config)                                 |
|    |      2-D numpy probability map                       |
|    |      Exposes: update_from_scan(), update_from_clue() |
|    |               update_negative(), top_cells(),        |
|    |               entropy(), normalize()                 |
|    |                                                      |
|    +-- Agent(config, environment, belief, logger)         |
|    |      _decide_action()  ->  utility scoring           |
|    |      _execute_action() ->  calls env methods         |
|    |      _update_belief()  ->  calls belief methods      |
|    |      _check_replan()   ->  replanning trigger        |
|    |      _safety_reserve() ->  battery safety rule       |
|    |                                                      |
|    +-- JSONLogger  -->  outputs/<name>_trace.json         |
|    +-- ReportGenerator --> outputs/<name>_report.md       |
|    +-- Visualizer  -->  outputs/<name>_map.txt            |
+-----------------------------------------------------------+
```

**Module responsibilities:**

| Module | Responsibility |
|---|---|
| `src/config.py` | Parse and validate YAML/JSON config into typed dataclasses |
| `src/environment.py` | Hidden ground truth; noisy scan and move responses |
| `src/belief.py` | Bayesian probability grid; all update and query methods |
| `src/agent.py` | Utility scoring, safety constraint, replanning logic |
| `src/logger.py` | Append each step as a JSON record to the trace file |
| `src/report.py` | Render final Markdown mission report from trace + belief |
| `src/visualize.py` | Render ASCII grid map with belief intensity overlay |
| `run.py` | Wire everything together; CLI argument parsing |
| `benchmark.py` | Batch runner across N seeds; aggregated JSON statistics |

---

## 3. Environment Design

### Grid World

The environment is a rectangular 2-D grid of configurable width and height.
Each cell can be:

- **Empty** — the agent can move here freely.
- **Obstacle** — impassable; the agent cannot enter.
- **Recharge station** — the agent can spend one action here to restore
  battery.
- **Clue** — a planted piece of physical evidence that boosts belief when the
  agent visits the cell (discovered via scanning nearby).
- **EVE's position** — the mission goal, hidden from the agent.
- **Decoy** (hard config) — a false signal source that can mislead scanning.

The agent always starts at cell `(0, 0)`.  The grid uses (x, y) coordinates
where `x` increases East and `y` increases South.

### Actions

| Action | Effect | Battery cost |
|---|---|---|
| `N` / `S` / `E` / `W` | Move one cell in that direction | `move_cost` |
| `scan` | Emit a sensor ping; returns clue direction + strength within radius 2 | `scan_cost` |
| `recharge` | Restore battery by `recharge_amount` (only on a station) | `recharge_cost` |
| `end_mission` | Terminate the episode immediately | 0 |

### Observations

After every action the environment returns an `Observation` dict:

```python
{
  "agent_pos":      [x, y],          # current position
  "battery":        int,             # remaining battery
  "time_remaining": int,             # steps left before time limit
  "steps_taken":    int,             # total steps so far
  "scan_result": {
    "clue_strength":  float,         # 0.0 if nothing detected
    "clue_direction": str | None,    # compass direction to nearest clue
    "eve_detected":   bool,          # True if EVE is within scan_radius
  },
  "event":          str,             # human-readable description
}
```

`scan_result` is populated with non-zero values only when the action was
`scan`.  All scan readings are corrupted by Gaussian noise scaled by the
`sensor_noise` parameter.

### Terminal Conditions

The episode ends when **any** of the following is true:

| Condition | How it triggers |
|---|---|
| EVE found | Agent moves onto EVE's cell |
| Battery depleted | `battery <= 0` after an action |
| Time limit reached | `steps_taken >= time_limit` |
| `end_mission` called | Agent explicitly ends the episode |

---

## 4. Config Format

Full annotated YAML with every field explained:

```yaml
# ------------------------------------------------------------------
# map: defines the simulation grid
# ------------------------------------------------------------------
map:
  width: 10          # Number of columns (integer >= 1)
  height: 10         # Number of rows    (integer >= 1)
  seed: 42           # RNG seed for reproducibility (integer)
  obstacles:         # List of [x, y] cells that block movement
    - [2, 3]         # Each entry is [column, row]
    - [2, 4]
    - [5, 5]
    - [6, 5]

# ------------------------------------------------------------------
# eve: EVE's true hidden position [x, y]
# ------------------------------------------------------------------
eve: [7, 8]

# ------------------------------------------------------------------
# initial_signal: the directional hint WALL-E received before the
# mission started.  Used to bias the initial belief prior.
# ------------------------------------------------------------------
initial_signal:
  direction: 'NE'    # Compass string: N S E W NE NW SE SW
                     # OR a coordinate pair [x, y] for a known origin
  strength: 0.8      # Signal confidence in [0.0, 1.0]
  timestamp: 0       # Mission timestamp of the original signal

# ------------------------------------------------------------------
# resources: battery and time budget
# ------------------------------------------------------------------
resources:
  battery: 150         # Starting battery level (integer)
  time_limit: 80       # Max steps before forced termination
  move_cost: 2         # Battery drained per move action
  scan_cost: 5         # Battery drained per scan action
  recharge_cost: 1     # Battery drained by the recharge action itself
                       # (set to 0 for a free recharge)
  recharge_amount: 50  # Battery restored per recharge action
  recharge_stations:   # List of [x, y] positions with chargers
    - [1, 1]
    - [8, 2]
    - [5, 9]

# ------------------------------------------------------------------
# clues: list of planted evidence objects, OR an integer seed to
# auto-generate clues procedurally.
# ------------------------------------------------------------------
clues:
  - position: [6, 7]      # [x, y] cell where this clue is located
    type: 'energy_trace'  # Flavour string (does not affect mechanics)
    strength: 0.7         # Evidence weight in [0.0, 1.0]; higher = stronger
                          # Bayesian boost when discovered
  - position: [7, 6]
    type: 'sound'
    strength: 0.5
  - position: [8, 8]
    type: 'heat_signature'
    strength: 0.8
  - position: [6, 9]
    type: 'movement'
    strength: 0.6

# clues can also be an integer — the environment then generates that
# many clues procedurally using map.seed:
# clues: 99

# ------------------------------------------------------------------
# sensor_noise: fraction of noise added to all scan readings.
# 0.0 = perfect sensor; 1.0 = completely random output.
# ------------------------------------------------------------------
sensor_noise: 0.05

# ------------------------------------------------------------------
# decoys: false signal sources that can mislead scanning.
# Each decoy has a position and a false-signal strength.
# Omit or leave empty ([]) for no decoys.
# ------------------------------------------------------------------
decoys: []
# decoys:
#   - position: [5, 5]
#     strength: 0.6
#   - position: [10, 3]
#     strength: 0.5
```

The hard config (`sample_hard.yaml`) uses:
- 15 x 15 grid with 14 obstacles
- Procedural clue generation (`clues: 99`)
- Two active decoys
- Higher sensor noise (`0.15`) and tighter battery (`120`)

---

## 5. Belief Update Algorithm

### What the belief grid represents

`BeliefGrid` maintains a 2-D numpy array `P[y, x]` where each entry is the
agent's current probability estimate that EVE is located at grid cell `(x, y)`.
The array is always normalised so all values sum to 1.

### Initialisation

The grid starts as a uniform prior (all cells equally likely).  The
`initial_signal` then biases the prior:

- **Direction string** (e.g. `'NE'`): for each cell `(x, y)`, the dot product
  of the cell's unit vector from the agent start and the signal direction
  vector is computed.  Cells in the signal's half-plane receive up to 3x
  extra weight:

  ```
  weight(x, y) = 1  +  max(0,  dot( unit(x-0, y-0),  dir_vec ))  *  2
  ```

- **Coordinate pair** `[sx, sy]`: a Gaussian centred on `(sx, sy)` is added
  to the uniform prior:

  ```
  weight(x, y) = 1  +  2 * exp( -dist((x,y),(sx,sy))^2 / (2 * sigma^2) )
  ```

  where `sigma = max(width, height) / 4`.

The grid is normalised after initialisation.

### Scan update (Bayesian likelihood)

When the agent scans and receives a clue signal of strength `s` pointing in
direction `d`:

For every cell `(x, y)`, the likelihood `L(x, y)` of receiving that
observation given EVE is at `(x, y)` is:

```
dot(x, y)    =  inner_product( unit(x - ax, y - ay),  dir_vec(d) )

L(x, y)  =  1  +  s * max(0, dot)  *  (1 - sensor_noise)    if dot > 0
L(x, y)  =  1                                                otherwise
```

The posterior is the prior multiplied by the likelihood, then renormalised:

```
P_new[y, x]  =  P_old[y, x]  *  L(x, y)
P_new        =  P_new / sum(P_new)
```

When the scan returns **nothing** (strength 0), probability in the scanned
area is gently reduced (multiplied by 0.6) as a negative update.

When `eve_detected = True`, all cells within `scan_radius` are multiplied by
3.0 before renormalisation.

### Clue discovery update

When the agent physically passes through a clue cell (detected via scanning),
a Gaussian kernel centred on the clue position is multiplied in:

```
boost(x, y)  =  1  +  s * exp( -dist((x,y), clue_pos)^2 / (2 * sigma_c^2) )

sigma_c  =  2 + s * 2     (stronger clues spread the boost wider)

P_new[y, x]  =  P_old[y, x]  *  boost(x, y)
P_new        =  P_new / sum(P_new)
```

### Negative (movement) update

Each time the agent moves to a new cell without finding EVE, probability in
the immediate neighbourhood (radius 1) is strongly suppressed:

```
P_new[y, x]  =  P_old[y, x]  *  0.3     for (x, y) within radius 1 of agent
P_new        =  P_new / sum(P_new)
```

This models the evidence "EVE is not in the area I just walked through."

### Entropy

Uncertainty is tracked via Shannon entropy (in nats):

```
H  =  -sum_i  P_i * ln(P_i)
```

Lower entropy = higher certainty about EVE's location.

---

## 6. Agent Decision Algorithm

At each step the agent calls `_decide_action()`, which scores every
candidate action and returns the highest-scoring safe action.

### Utility scoring

**Move utility** (for each of N, S, E, W):

```
dist_before  =  Manhattan( agent_pos, target_cell )
dist_after   =  Manhattan( next_pos,  target_cell )

belief_gain       =  (dist_before - dist_after) * P[target_cell]
resource_penalty  =  move_cost / battery
revisit_penalty   =  visit_count(next_pos) * 0.05

U(move)  =  belief_gain - resource_penalty - revisit_penalty
```

`target_cell` is the highest-probability cell in the current belief grid.

**Scan utility:**

```
H            =  entropy of current belief
expected_dH  =  H * 0.15      (heuristic: each scan reduces entropy ~15%)
scan_cost_penalty  =  scan_cost / battery

U(scan)  =  expected_dH - scan_cost_penalty
```

**Recharge utility:**

```
battery_fraction  =  battery / max_battery

U(recharge)  =  100.0     if on_station AND battery_fraction < 0.4
             =  -inf       otherwise
```

The final action is `argmax(U)`.

### Safety constraint (hard rule)

Before any action is scored, the agent computes a `safety_reserve`:

```
safety_reserve  =  Manhattan( agent_pos, nearest_station ) * move_cost
```

An action is only **eligible** if:

```
battery - action_cost  >=  safety_reserve
```

This guarantees WALL-E can always walk back to a charger after any action
it commits to.

### Replanning trigger

After each step, the agent checks whether the top belief cell has shifted:

```
dist  =  Manhattan( new_top_cell, last_top_cell )

if dist > 2:
    # halve all visit counts to release grip on current path
    visited_counts = { k: v // 2 for k, v in visited_counts.items() }
    last_top_cell  = new_top_cell
```

This prevents the agent from fixating on a stale target when new evidence
points elsewhere.

---

## 7. How to Run

### Installation

Python 3.10+ is required.

```bash
# Clone the repository
git clone <repo-url>
cd wall-e-eve-signal

# Install dependencies (exact versions pinned)
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy>=1.24.0
pyyaml>=6.0
matplotlib>=3.7.0
pytest>=7.4.0
```

### Run the easy mission (10 x 10 grid)

```bash
python run.py --config configs/sample_easy.yaml
```

Outputs written to `outputs/`:
- `sample_easy_trace.json` — full step-by-step JSON log
- `sample_easy_report.md` — Markdown mission report
- `sample_easy_map.txt` — ASCII belief-intensity map

### Run the hard mission (15 x 15 grid, decoys, high noise)

```bash
python run.py --config configs/sample_hard.yaml
```

Outputs written to `outputs/`:
- `sample_hard_trace.json`
- `sample_hard_report.md`
- `sample_hard_map.txt`

### Optional flags

```bash
python run.py --config configs/sample_easy.yaml \
              --output-dir results \   # write outputs to a different folder
              --verbose                # print intermediate belief snapshots
```

### Batch benchmark

Run N independent missions (varying the RNG seed each time) and collect
aggregate statistics:

```bash
# 20 runs on the easy config
python benchmark.py --config configs/sample_easy.yaml --runs 20

# 50 runs on the hard config, custom output path
python benchmark.py --config configs/sample_hard.yaml \
                    --runs 50 \
                    --output results/bench_hard.json
```

The output JSON contains per-run results (`seed`, `found_eve`, `steps`,
`battery_remaining`, `confidence`) plus aggregate statistics
(`success_rate`, `mean_steps`, `mean_battery_remaining`).

---

## 8. Sample Output

### Trace excerpt — first 3 steps (`sample_easy_trace.json`)

```json
[
  {
    "step": 1,
    "action": "scan",
    "observation": {
      "agent_pos": [0, 0],
      "battery": 145,
      "time_remaining": 79,
      "steps_taken": 1,
      "scan_result": {
        "clue_strength": 0.0,
        "clue_direction": null,
        "eve_detected": false
      },
      "event": "scan"
    },
    "battery": 145,
    "time_remaining": 79,
    "belief_summary": {
      "top_cells": [
        {"x": 8, "y": 0, "prob": 0.020598},
        {"x": 6, "y": 0, "prob": 0.020598},
        {"x": 9, "y": 0, "prob": 0.020598},
        {"x": 4, "y": 0, "prob": 0.020598},
        {"x": 5, "y": 0, "prob": 0.020598}
      ],
      "entropy": 4.5021
    },
    "decision": "scan",
    "reasoning": "Scanning to reduce entropy (H=4.508). Expected reduction ~= 0.676, resource cost = 0.0333."
  },
  {
    "step": 2,
    "action": "scan",
    "observation": {
      "agent_pos": [0, 0],
      "battery": 140,
      "time_remaining": 78,
      "steps_taken": 2,
      "scan_result": {
        "clue_strength": 0.0,
        "clue_direction": null,
        "eve_detected": false
      },
      "event": "scan"
    },
    "battery": 140,
    "time_remaining": 78,
    "belief_summary": {
      "top_cells": [
        {"x": 8, "y": 0, "prob": 0.021081},
        {"x": 6, "y": 0, "prob": 0.021081},
        {"x": 9, "y": 0, "prob": 0.021081},
        {"x": 4, "y": 0, "prob": 0.021081},
        {"x": 5, "y": 0, "prob": 0.021081}
      ],
      "entropy": 4.4869
    },
    "decision": "scan",
    "reasoning": "Scanning to reduce entropy (H=4.502). Expected reduction ~= 0.675, resource cost = 0.0345."
  },
  {
    "step": 3,
    "action": "scan",
    "observation": {
      "agent_pos": [0, 0],
      "battery": 135,
      "time_remaining": 77,
      "steps_taken": 3,
      "scan_result": {
        "clue_strength": 0.0,
        "clue_direction": null,
        "eve_detected": false
      },
      "event": "scan"
    },
    "battery": 135,
    "time_remaining": 77,
    "belief_summary": {
      "top_cells": [
        {"x": 8, "y": 0, "prob": 0.021381},
        {"x": 6, "y": 0, "prob": 0.021381},
        {"x": 9, "y": 0, "prob": 0.021381},
        {"x": 4, "y": 0, "prob": 0.021381},
        {"x": 5, "y": 0, "prob": 0.021381}
      ],
      "entropy": 4.4704
    },
    "decision": "scan",
    "reasoning": "Scanning to reduce entropy (H=4.487). Expected reduction ~= 0.673, resource cost = 0.0357."
  }
]
```

**Observation:** The initial signal pointed `NE`, so the belief concentrates
on the north-east edge cells.  The first three scans from `(0,0)` return no
signal (EVE is at `(7,8)`, well outside scan radius), so entropy decreases
only slightly via the negative scan update.

### Final ASCII map (`sample_easy_map.txt`)

```
   0123456789
   ----------
 0|@  @@@@@@@
 1| R #%%%%@@
 2|   +*###R%
 3|--#-=+**##
 4|--#--=++**
 5|-----##=++
 6|-------C=+
 7|------C--=
 8|--------C-
 9|-----RC---

Legend: @=Agent  *=Visited  C=Clue  R=Recharge  #=Obstacle  E=EVE(found)  .=Empty
Belief intensity (low->high): .  :  -  =  +  *  #  %  @
```

### Mission report snippet (`sample_easy_report.md`)

```
## 1. Mission Summary

| Field             | Value             |
|-------------------|-------------------|
| Config Name       | sample_easy       |
| Grid Size         | 10 x 10           |
| Seed              | 42                |
| Total Steps       | 30                |
| Mission Result    | [NOT FOUND]       |
| Terminal Event    | mission_ended     |

## 3. Inferred EVE Location and Status

| Field               | Value    |
|---------------------|----------|
| Most Likely Cell    | (8, 0)   |
| Cell Probability    | 2.2%     |
| Belief Entropy      | 4.4172   |
| EVE Status Estimate | unknown  |

## 4. Resources Used vs Remaining

| Resource     | Used | Remaining | Limit |
|--------------|------|-----------|-------|
| Battery      | 145  | 5         | 150   |
| Steps / Time | 29   | 51        | 80    |
```

---

## 9. Advanced Features

### Noisy sensors

Set `sensor_noise` in the config to a value between `0.0` and `1.0`:

```yaml
sensor_noise: 0.15   # 15% noise — used in sample_hard.yaml
```

The likelihood multiplier for each Bayesian update is scaled by
`(1 - sensor_noise)`, so higher noise makes each scan less informative.
With `sensor_noise: 0.0` the sensor is perfect; at `1.0` all scans are
ignored entirely by the belief update.

### Decoys

Decoys are false signal sources placed in the grid.  They emit scan
responses with a configurable false strength, causing the agent to
temporarily shift its belief toward the wrong area:

```yaml
decoys:
  - position: [5, 5]
    strength: 0.6    # false signal strength in [0, 1]
  - position: [10, 3]
    strength: 0.5
```

The agent has no way to distinguish a decoy from a real clue without
physically visiting the area and receiving subsequent negative evidence.
Decoys are defined at the config level and handled inside `Environment`.

### Procedural clue generation

Instead of listing clues manually, pass an integer as the `clues` field.
The environment generates that many clues pseudo-randomly using `map.seed`:

```yaml
clues: 99   # generate 99 clues procedurally
```

Useful for large grids or automated benchmark generation.

### Batch benchmark

`benchmark.py` runs N missions, varying the map seed on each run:

```bash
python benchmark.py --config configs/sample_hard.yaml --runs 50
```

Each run produces a result record:

```json
{
  "seed": 7,
  "found_eve": false,
  "steps": 58,
  "battery_remaining": 3,
  "confidence": 0.008312
}
```

Aggregate statistics in the output JSON include success rate, mean steps,
and mean battery remaining — useful for comparing config difficulty.

---

## 10. Testing

The test suite uses `pytest` and covers all four core modules.

### Run all tests

```bash
python -m pytest tests/ -v
```

Expected output:

```
tests/test_agent.py       ... passed
tests/test_belief.py      ... passed
tests/test_config.py      ... passed
tests/test_environment.py ... passed

95 passed in 0.69s
```

### Test files

| File | What it covers |
|---|---|
| `tests/test_environment.py` | Grid initialisation, move/scan/recharge mechanics, obstacle collision, terminal conditions, battery depletion |
| `tests/test_belief.py` | Prior initialisation, scan updates, clue updates, negative updates, normalisation, entropy calculation |
| `tests/test_agent.py` | Action selection, safety reserve, replanning trigger, full episode runs |
| `tests/test_config.py` | YAML parsing, field validation, procedural clue generation, decoy loading |

### Run a specific test file

```bash
python -m pytest tests/test_belief.py -v
```

### Run tests matching a keyword

```bash
python -m pytest tests/ -v -k "recharge"
```
