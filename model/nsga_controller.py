# ── Standard library ──────────────────────────────────────────────────────────
import time
from typing import Tuple
import csv
import os

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

import traci
import random


def save_results(scenario, algorithm,  delay, queue, co2, throughput):
    file = "results.csv"
    file_exists = os.path.isfile(file)

    with open(file, "a", newline="") as f:
        writer = csv.writer(f)

        # write header only once
        if not file_exists:
            writer.writerow(["Scenario", "Algorithm", "Delay", "Queue", "CO2", "Throughput"])

        writer.writerow([scenario, algorithm, delay, queue, co2, throughput])
def run_sumo(green_times):
    g1, g2, g3, g4 = green_times

    traci.start(["sumo", "-c", "config.sumocfg"])

    total_wait = 0
    total_queue = 0
    total_co2 = 0
    total_vehicles = 0

    for step in range(1000):
        traci.simulationStep()

        cycle = int(g1 + g2 + g3 + g4)

        phase_time = step % cycle

        if phase_time < g1:
          traci.trafficlight.setPhase("center", 0)
        elif phase_time < g1 + g2:
          traci.trafficlight.setPhase("center", 1)
        elif phase_time < g1 + g2 + g3:
         traci.trafficlight.setPhase("center", 2)
        else:
          traci.trafficlight.setPhase("center", 3)
        # Collect metrics
        edges = ["n2c", "s2c", "e2c", "w2c"]

        for e in edges:
          total_wait += traci.edge.getWaitingTime(e)
          total_queue += traci.edge.getLastStepVehicleNumber(e)
          total_vehicles += traci.simulation.getArrivedNumber()
          total_co2 += traci.edge.getCO2Emission(e)

    traci.close()

    avg_delay = total_wait / (total_vehicles + 1)
    avg_queue = total_queue / 1000
    co2 = total_co2  # approx (or use emission API)
    throughput = total_vehicles
    print("Running simulation with:", green_times)
   



    return avg_delay, avg_queue, co2, throughput
    
   

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Problem definition
# ─────────────────────────────────────────────────────────────────────────────

YELLOW_TIME   = 3          # seconds (fixed, not optimized)
GREEN_MIN     = 10.0       # seconds
GREEN_MAX     = 60.0       # seconds
N_VARIABLES   = 4          # g1, g2, g3, g4
N_OBJECTIVES  = 4          # delay, queue, co2, -throughput
PHASE_LABELS  = ["NS_straight", "EW_straight", "NS_left", "EW_left"]


class TrafficSignalProblem(ElementwiseProblem):
  

    def __init__(self, **kwargs):
        super().__init__(
            n_var=N_VARIABLES,
            n_obj=N_OBJECTIVES,
            n_ieq_constr=0,           # no inequality constraints
            xl=np.full(N_VARIABLES, GREEN_MIN),
            xu=np.full(N_VARIABLES, GREEN_MAX),
            **kwargs,
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
      
        delay, queue_length, co2, throughput = run_sumo(x)

        out["F"] = np.array([
            delay,
            queue_length,
            co2,
            -throughput,          # maximise throughput → minimise negative
        ], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  NSGA-II setup
# ─────────────────────────────────────────────────────────────────────────────

def build_nsga2(pop_size: int = 60) -> NSGA2:
  
    return NSGA2(
        pop_size=pop_size,
        # Simulated Binary Crossover — good for real-valued problems
        crossover=SBX(prob=0.9, eta=15),
        # Polynomial Mutation
        mutation=PM(prob=1.0 / N_VARIABLES, eta=20),
        # Latin-Hypercube-style uniform sampling in the feasible box
        sampling=FloatRandomSampling(),
        eliminate_duplicates=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Optimisation execution
# ─────────────────────────────────────────────────────────────────────────────

def run_optimization(
    pop_size: int   = 60,
    n_gen: int      = 50,
    seed: int       = 42,
    verbose: bool   = True,
) -> object:
    
    problem     = TrafficSignalProblem()
    algorithm   = build_nsga2(pop_size=pop_size)
    termination = get_termination("n_gen", n_gen)

    print(f"\n{'═'*60}")
    print("  NSGA-II · Traffic Signal Timing Optimisation")
    print(f"{'═'*60}")
    print(f"  Population size : {pop_size}")
    print(f"  Generations     : {n_gen}")
    print(f"  Evaluations     : {pop_size * n_gen:,}  (≈)")
    print(f"  Decision vars   : {N_VARIABLES}  ({', '.join(PHASE_LABELS)})")
    print(f"  Objectives      : {N_OBJECTIVES}  (delay, queue, CO2, −throughput)")
    print(f"{'─'*60}\n")

    t0 = time.perf_counter()

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=True,
        save_history=False,
    )

    elapsed = time.perf_counter() - t0
    print(f"\n{'─'*60}")
    print(f"  Optimisation complete in {elapsed:.1f} s")
    print(f"  Pareto-front size : {len(result.X)} solutions")
    print(f"{'═'*60}\n")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Result printing
# ─────────────────────────────────────────────────────────────────────────────

def print_results(result) -> None:
   

    X = result.X          # shape (n_pareto, 4)
    F = result.F          # shape (n_pareto, 4)

    header = (
        f"{'Sol':>4}  "
        f"{'g1':>6}  {'g2':>6}  {'g3':>6}  {'g4':>6}  "
        f"{'Delay':>8}  {'Queue':>7}  {'CO2':>10}  {'Thru':>7}"
    )
    sep = "─" * len(header)

    print("  Pareto-Optimal Signal Timings")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")

    # Sort by average delay for readability
    order = np.argsort(F[:, 0])

    for rank, idx in enumerate(order, start=1):
        g1, g2, g3, g4 = X[idx]
        delay, queue, co2, neg_thru = F[idx]
        throughput = -neg_thru

        print(
            f"  {rank:>4}  "
            f"{g1:>6.1f}  {g2:>6.1f}  {g3:>6.1f}  {g4:>6.1f}  "
            f"{delay:>8.2f}  {queue:>7.2f}  {co2:>10.1f}  {throughput:>7.3f}"
        )

    print(f"  {sep}")
    print(f"\n  Objective units  →  Delay: s/veh | Queue: veh | CO2: mg | Thru: veh")

    # ── Summary statistics across the front ──────────────────────────────────
    print("\n  Pareto-front statistics")
    print(f"  {'Objective':<14}  {'Min':>10}  {'Mean':>10}  {'Max':>10}")
    print(f"  {'─'*46}")
    labels = ["Delay (s/veh)", "Queue (veh)", "CO2 (mg)", "Thru (veh)"]
    signs  = [1, 1, 1, -1]     # last column is stored negated
    for i, (lbl, sign) in enumerate(zip(labels, signs)):
        col = F[:, i] * sign
        print(f"  {lbl:<14}  {col.min():>10.2f}  {col.mean():>10.2f}  {col.max():>10.2f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_optimization(
        pop_size=50,
        n_gen=30,
        seed=42,
        verbose=True,
    )
    print_results(result)
    best_idx = np.argmin(result.F[:,0])  # best delay

    best_delay, best_queue, best_co2, neg_thru = result.F[best_idx]
    best_throughput = -neg_thru

    save_results("low", "NSGA-II", best_delay, best_queue, best_co2, best_throughput)
    print(result.X)
    print(result.F)
