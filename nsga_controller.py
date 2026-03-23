"""
traffic_signal_nsga2.py
=======================
Uses NSGA-II (via pymoo) to minimize average delay, queue length, CO2 emissions,
and maximize throughput (modelled as minimization of negative throughput).

Simulation is delegated to an external `run_sumo()` stub — replace the body
of that function with your real SUMO TraCI / libsumo call.

Requirements
------------
    pip install pymoo numpy

Usage
-----
    python traffic_signal_nsga2.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import time
from typing import Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SUMO interface (stub — replace with real TraCI / libsumo calls)
# ─────────────────────────────────────────────────────────────────────────────

def run_sumo(green_times: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Evaluate one candidate signal-timing plan inside SUMO.

    Parameters
    ----------
    green_times : array-like, shape (4,)
        [g1_NS_straight, g2_EW_straight, g3_NS_left, g4_EW_left]
        All values are in seconds (floats, already within [10, 60]).

    Returns
    -------
    delay       : float  — average vehicle delay  (seconds/vehicle)
    queue_length: float  — mean queue length       (vehicles)
    co2         : float  — total CO2 emitted       (mg)
    throughput  : float  — vehicles that completed their trip

    """
    g1, g2, g3, g4 = green_times          # unpack for clarity

    # ── Placeholder physics (linear proxy of a real simulation) ──────────────
    # These formulas are illustrative only and carry no real physical meaning.
    total_green = g1 + g2 + g3 + g4
    cycle_time  = total_green + 4 * 3     # 3-s yellow × 4 phases

    delay        = 30 * (60 / (g1 + g2 + 1)) + 0.5 * cycle_time + np.random.normal(0, 0.5)
    queue_length = 5  * (30 / (g1 + g3 + 1)) + 0.2 * cycle_time + np.random.normal(0, 0.2)
    co2          = 200 * delay + 50 * queue_length               + np.random.normal(0, 10)
    throughput   = 3600 / cycle_time * (total_green / cycle_time) * 10 + np.random.normal(0, 1)
    # ─────────────────────────────────────────────────────────────────────────

    return float(delay), float(queue_length), float(co2), float(throughput)


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
    """
    4-variable, 4-objective optimisation problem.

    Decision variables
    ------------------
    x[0] = g1  NS straight green time  [10, 60] s
    x[1] = g2  EW straight green time  [10, 60] s
    x[2] = g3  NS left     green time  [10, 60] s
    x[3] = g4  EW left     green time  [10, 60] s

    Objectives (all minimised)
    --------------------------
    f0 = average delay          (s/veh)        ↓
    f1 = mean queue length      (veh)          ↓
    f2 = total CO2 emissions    (mg)           ↓
    f3 = −throughput            (veh/s)        ↓  (negate to convert max→min)
    """

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
        """Called once per individual (ElementwiseProblem)."""
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
    """
    Construct an NSGA-II algorithm instance with tuned operators.

    Parameters
    ----------
    pop_size : int
        Population size (40–100 recommended for this problem).

    Returns
    -------
    NSGA2 algorithm object ready for pymoo's `minimize()`.
    """
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
    """
    Run the full NSGA-II optimisation loop.

    Parameters
    ----------
    pop_size : Population size.
    n_gen    : Number of generations.
    seed     : Random seed for reproducibility.
    verbose  : Print pymoo's per-generation progress.

    Returns
    -------
    pymoo Result object  (result.X = decision vars, result.F = objectives).
    """
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
        verbose=verbose,
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
    """
    Pretty-print the Pareto-optimal solutions.

    Columns
    -------
    Sol  : solution index (1-based)
    g1…g4: green-time decision variables (seconds)
    Delay: average delay       (s/veh)
    Queue: mean queue length   (veh)
    CO2  : CO2 emissions       (mg)
    Thru : throughput          (veh — positive, recovered from −f3)
    """
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
        pop_size=60,
        n_gen=50,
        seed=42,
        verbose=True,
    )
    print_results(result)
