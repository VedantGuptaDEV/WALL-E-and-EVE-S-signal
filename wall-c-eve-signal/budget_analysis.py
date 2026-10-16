"""Budget analysis for hard config — shows why scans never fire."""
import sys
sys.path.insert(0, ".")

from src.config import load_config
from src.environment import Environment
from src.belief import BeliefGrid
from src.agent import Agent
from src.logger import JSONLogger
import json

config = load_config("configs/sample_hard.yaml")

print("=== BUDGET ANALYSIS ===")
print(f"battery       : {config.resources.battery}")
print(f"move_cost     : {config.resources.move_cost}")
print(f"scan_cost     : {config.resources.scan_cost}")
print(f"recharge_amt  : {config.resources.recharge_amount}")
print(f"recharge_cost : {config.resources.recharge_cost}")
print(f"time_limit    : {config.resources.time_limit}")
print(f"grid size     : {config.map.width}x{config.map.height} = {config.map.width*config.map.height} cells")
print(f"max pure moves: {config.resources.battery // config.resources.move_cost}")
print(f"EVE at        : {config.eve}")
print(f"stations      : {config.resources.recharge_stations}")
print()

# Simulate with trace to capture safety_reserve at each step
env = Environment(config)
belief = BeliefGrid(
    width=config.map.width, height=config.map.height,
    initial_signal=config.initial_signal.direction,
    seed=config.map.seed, signal_strength=config.initial_signal.strength,
)
logger = JSONLogger("diag_trace2.json")
agent = Agent(config=config, environment=env, belief=belief, logger=logger)

# Monkey-patch _decide_action to log safety_reserve vs scan check
original_decide = agent._decide_action
scan_blocked_reasons = []

def patched_decide():
    battery = env.battery
    scan_cost = config.resources.scan_cost
    move_cost = config.resources.move_cost
    ax, ay = env.agent_pos
    safety_reserve = agent._safety_reserve()
    scan_reserve = safety_reserve + move_cost  # the new check
    scans_here = agent._total_scans_at.get((ax, ay), 0)
    
    can_scan = (battery - scan_cost >= scan_reserve) and (scans_here < 2)
    scan_blocked_reasons.append({
        "step": env.steps_taken + 1,
        "battery": battery,
        "safety_reserve": safety_reserve,
        "scan_reserve": scan_reserve,
        "scan_cost": scan_cost,
        "after_scan": battery - scan_cost,
        "can_scan": can_scan,
        "scans_here": scans_here,
        "pos": list(env.agent_pos),
    })
    return original_decide()

agent._decide_action = patched_decide
result = agent.run()

print("=== SCAN ELIGIBILITY EACH STEP ===")
print(f"{'Step':>4} {'Pos':>10} {'Batt':>5} {'Reserve':>8} {'ScanRes':>8} {'AfterScan':>9} {'CanScan':>8}")
for r in scan_blocked_reasons:
    flag = "YES" if r["can_scan"] else "NO "
    print(f"  {r['step']:3d}  {str(r['pos']):>10}  {r['battery']:4d}  {r['safety_reserve']:7d}  {r['scan_reserve']:7d}  {r['after_scan']:8d}  {flag}")

print()
print(f"Total steps: {result['steps']}, found_eve: {result['found_eve']}")
print()
print("=== BUDGET FIX RECOMMENDATION ===")
# EVE is at [12,11], start at [0,0]. 
# Path to EVE: ~23 moves minimum. Scanning 5 times = 40 battery.
# Path back to station from EVE: ~2 moves = 6 battery.
# Total needed: 23*3 + 5*8 + 2*3 = 69 + 40 + 6 = 115 just for a direct run.
# But agent starts near obstacles and needs to explore first.
# Recommended: battery=220, time_limit=100
eve_x, eve_y = config.eve
direct_dist = eve_x + eve_y  # Manhattan from [0,0] to [12,11] = 23
direct_cost = direct_dist * config.resources.move_cost
scan_budget = 6 * config.resources.scan_cost  # 6 scans while searching
return_dist = 2  # from EVE to nearest station
return_cost = return_dist * config.resources.move_cost
exploration = 15 * config.resources.move_cost  # extra for exploration detours
total_needed = direct_cost + scan_budget + return_cost + exploration
print(f"Min battery for reliable solve: ~{total_needed}")
print(f"  direct walk:  {direct_dist} moves × {config.resources.move_cost} = {direct_cost}")
print(f"  6 scans:      6 × {config.resources.scan_cost} = {scan_budget}")
print(f"  return:       {return_dist} moves × {config.resources.move_cost} = {return_cost}")
print(f"  exploration:  15 moves × {config.resources.move_cost} = {exploration}")
print(f"Recommended config values:")
print(f"  battery: 220")
print(f"  time_limit: 100")
