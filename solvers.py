"""
solvers.py — Aggregate Planning Strategy Solvers
All six strategies return a standardised results dict consumed by app.py.

Common result structure:
{
    "periods":      list[str],
    "demand":       list[float],
    "production":   list[float],
    "workforce":    list[float],
    "hired":        list[float],
    "fired":        list[float],
    "inventory":    list[float],
    "overtime":     list[float],
    "subcontract":  list[float],
    "cost_rt":      list[float],   # regular-time labour cost per period
    "cost_hire":    list[float],
    "cost_fire":    list[float],
    "cost_hold":    list[float],
    "cost_back":    list[float],
    "cost_ot":      list[float],
    "cost_sub":     list[float],
    "cost_total":   list[float],   # sum of all costs per period
    "grand_total":  float,
    "feasible":     bool,
    "message":      str,
    "shadow_prices": dict | None,  # LP only
    "transport_tableau": dict | None,  # Transportation only
}
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linprog, milp, LinearConstraint, Bounds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_result(periods, demand):
    T = len(periods)
    z = [0.0] * T
    return dict(
        periods=periods, demand=demand,
        production=z[:], workforce=z[:], hired=z[:], fired=z[:],
        inventory=z[:], overtime=z[:], subcontract=z[:], lost_sales=z[:],
        cost_rt=z[:], cost_hire=z[:], cost_fire=z[:],
        cost_hold=z[:], cost_back=z[:], cost_ot=z[:], cost_sub=z[:], cost_ls=z[:],
        cost_total=z[:], grand_total=0.0,
        feasible=True, message="", shadow_prices=None, transport_tableau=None,
    )


def _compute_period_costs(res: dict, costs: dict) -> dict:
    T = len(res["periods"])
    for t in range(T):
        rt  = res["production"][t]  * costs["regular_time"]
        hi  = res["hired"][t]       * costs["hiring"]
        fi  = res["fired"][t]       * costs["firing"]
        hld = max(res["inventory"][t], 0) * costs["holding"]
        bk  = max(-res["inventory"][t], 0) * costs["backorder"]
        ot  = res["overtime"][t]    * costs["overtime"]
        sub = res["subcontract"][t] * costs["subcontracting"]
        ls  = res["lost_sales"][t]  * costs.get("lost_sales", 0.0)
        res["cost_rt"][t]   = rt
        res["cost_hire"][t] = hi
        res["cost_fire"][t] = fi
        res["cost_hold"][t] = hld
        res["cost_back"][t] = bk
        res["cost_ot"][t]   = ot
        res["cost_sub"][t]  = sub
        res["cost_ls"][t]   = ls
        res["cost_total"][t] = rt + hi + fi + hld + bk + ot + sub + ls
    res["grand_total"] = sum(res["cost_total"])
    return res


# ---------------------------------------------------------------------------
# 1. Chase Demand Strategy
# ---------------------------------------------------------------------------

def solve_chase(periods, demand, costs, capacity, initial_workforce,
                initial_inventory, productivity,
                integer_workforce=False, integer_production=False,
                shortage_policy="backorders"):
    """
    Production = Demand each period.
    Workforce adjusted to exactly meet production; no overtime/subcontract.
    """
    T = len(periods)
    res = _empty_result(periods, demand)

    inv = float(initial_inventory)
    wf  = float(initial_workforce)

    for t in range(T):
        prod = float(demand[t])
        new_wf = prod / productivity  # workers needed

        hired = max(0.0, new_wf - wf)
        fired = max(0.0, wf - new_wf)

        inv = inv + prod - demand[t]  # always 0 when prod == demand

        res["production"][t] = float(round(prod))   if integer_production else prod
        res["workforce"][t]  = float(round(new_wf)) if integer_workforce  else new_wf
        res["hired"][t]      = float(round(hired))  if integer_workforce  else hired
        res["fired"][t]      = float(round(fired))  if integer_workforce  else fired
        res["inventory"][t]  = inv
        res["lost_sales"][t] = 0.0
        wf = res["workforce"][t]

    return _compute_period_costs(res, costs)


# ---------------------------------------------------------------------------
# 2. Level Production Strategy
# ---------------------------------------------------------------------------

def solve_level(periods, demand, costs, capacity, initial_workforce,
                initial_inventory, productivity,
                integer_workforce=False, integer_production=False,
                shortage_policy="backorders"):
    """
    Constant production rate = average demand.
    Inventory / backorders absorb variability.
    """
    T = len(periods)
    res = _empty_result(periods, demand)

    avg_prod = sum(demand) / T
    wf       = avg_prod / productivity
    if integer_workforce:
        wf = float(round(wf))
    if integer_production:
        avg_prod = float(round(avg_prod))
    inv      = float(initial_inventory)

    for t in range(T):
        inv_raw = inv + avg_prod - demand[t]
        if shortage_policy == "lost_sales" and inv_raw < 0:
            res["lost_sales"][t] = -inv_raw
            inv = 0.0
        elif shortage_policy == "no_shortages" and inv_raw < 0:
            res["lost_sales"][t] = 0.0
            inv = inv_raw   # flag infeasibility but proceed
        else:
            res["lost_sales"][t] = 0.0
            inv = inv_raw

        res["production"][t] = avg_prod
        res["workforce"][t]  = wf
        res["hired"][t]      = max(0.0, wf - (initial_workforce if t == 0 else wf))
        res["fired"][t]      = max(0.0, (initial_workforce if t == 0 else wf) - wf)
        res["inventory"][t]  = inv

    # One-time hire/fire on first period
    res["hired"][0]  = max(0.0, wf - initial_workforce)
    res["fired"][0]  = max(0.0, initial_workforce - wf)

    if shortage_policy == "no_shortages" and min(res["inventory"]) < 0:
        res["feasible"] = False
        res["message"]  = "No-shortages policy violated: demand exceeds production capacity in at least one period."

    return _compute_period_costs(res, costs)


# ---------------------------------------------------------------------------
# 3. Mixed / Hybrid Strategy
# ---------------------------------------------------------------------------

def solve_mixed(periods, demand, costs, capacity, initial_workforce,
                initial_inventory, productivity,
                target_workforce=None, allow_overtime=True,
                allow_subcontract=True,
                integer_workforce=False, integer_production=False,
                shortage_policy="backorders"):
    """
    Partial workforce adjustment.  Gaps filled with overtime then subcontracting.
    target_workforce: fixed workforce level to use (defaults to average).
    """
    T = len(periods)
    res = _empty_result(periods, demand)

    avg_wf   = sum(demand) / T / productivity
    wf_fixed = float(target_workforce) if target_workforce else avg_wf
    wf_fixed = min(wf_fixed, capacity["max_workforce"])
    inv      = float(initial_inventory)
    wf       = float(initial_workforce)
    max_ot_frac = capacity["max_overtime_fraction"]
    max_sub     = capacity["max_subcontract_per_period"]

    for t in range(T):
        new_wf = wf_fixed
        hired  = max(0.0, new_wf - wf)
        fired  = max(0.0, wf - new_wf)

        rt_cap  = new_wf * productivity
        ot_cap  = rt_cap * max_ot_frac if allow_overtime else 0.0
        gap     = max(0.0, demand[t] - inv - rt_cap)
        ot_used  = min(gap, ot_cap) if allow_overtime else 0.0
        gap2    = max(0.0, gap - ot_used)
        sub_used = min(gap2, max_sub) if allow_subcontract else 0.0
        prod    = rt_cap + ot_used + sub_used
        inv_raw = inv + prod - demand[t]

        if shortage_policy == "lost_sales" and inv_raw < 0:
            res["lost_sales"][t] = -inv_raw
            inv = 0.0
        elif shortage_policy == "no_shortages" and inv_raw < 0:
            res["lost_sales"][t] = 0.0
            inv = inv_raw   # unchanged; infeasibility flagged below
        else:
            res["lost_sales"][t] = 0.0
            inv = inv_raw

        res["production"][t]  = float(round(prod))    if integer_production else prod
        res["workforce"][t]   = float(round(new_wf)) if integer_workforce  else new_wf
        res["hired"][t]       = float(round(hired))  if integer_workforce  else hired
        res["fired"][t]       = float(round(fired))  if integer_workforce  else fired
        res["inventory"][t]   = inv
        res["overtime"][t]    = float(round(ot_used))  if integer_production else ot_used
        res["subcontract"][t] = float(round(sub_used)) if integer_production else sub_used
        wf = res["workforce"][t]

    if shortage_policy == "no_shortages" and min(res["inventory"]) < 0:
        res["feasible"] = False
        res["message"]  = "No-shortages policy violated: demand exceeds production + overtime + subcontract capacity."

    return _compute_period_costs(res, costs)

def solve_lp(periods, demand, costs, capacity, initial_workforce,
             initial_inventory, productivity,
             integer_workforce=False, integer_production=False,
             shortage_policy="backorders"):
    """
    Decision variables per period t (0-indexed, T periods):
      x[t]   = production (regular-time units)
      h[t]   = hired workers
      f[t]   = fired workers
      w[t]   = workforce level
      inv[t] = inventory (can go negative = backorder)
      ot[t]  = overtime units
      sub[t] = subcontracted units

    Variable order in vector v: [x_0..x_{T-1}, h_0..h_{T-1}, f, w, inv, ot, sub]
    """
    T = len(periods)

    # Index helpers
    ix  = slice(0,   T)         # production (RT)
    ih  = slice(T,   2*T)       # hired
    iff = slice(2*T, 3*T)       # fired
    iw  = slice(3*T, 4*T)       # workforce
    ii  = slice(4*T, 5*T)       # inventory (unbounded below)
    iot = slice(5*T, 6*T)       # overtime units
    isb = slice(6*T, 7*T)       # subcontract
    # Variable layout: [x, h, f, w, ip, im, ot, sub, ls]
    # im = backorder (inventory-), ls = lost sales; policy controls which are active
    ix  = slice(0,    T)
    ih  = slice(T,    2*T)
    iff = slice(2*T,  3*T)
    iw  = slice(3*T,  4*T)
    ip  = slice(4*T,  5*T)   # inventory+
    im  = slice(5*T,  6*T)   # inventory- (backorder)
    iot = slice(6*T,  7*T)
    isb = slice(7*T,  8*T)
    ils = slice(8*T,  9*T)   # lost sales
    N   = 9 * T

    c_obj = np.zeros(N)
    c_obj[ix]  = costs["regular_time"] / productivity
    c_obj[ih]  = costs["hiring"]
    c_obj[iff] = costs["firing"]
    c_obj[ip]  = costs["holding"]
    c_obj[im]  = costs["backorder"]
    c_obj[iot] = costs["overtime"]
    c_obj[isb] = costs["subcontracting"]
    c_obj[ils] = costs.get("lost_sales", 0.0)

    # ---- Equality constraints (A_eq @ v == b_eq) ----
    # (a) Inventory balance with optional lost sales:
    #   x[t] + ot[t] + sub[t] + ip[t-1] - im[t-1] - ip[t] + im[t] + ls[t] = demand[t]
    # (b) Workforce balance: w[t] - h[t] + f[t] - w[t-1] = 0

    n_eq = 2 * T
    A_eq = np.zeros((n_eq, N))
    b_eq = np.zeros(n_eq)

    for t in range(T):
        row_inv = t
        row_wf  = T + t

        # Inventory balance
        A_eq[row_inv, ix.start + t]  = 1   # x[t]
        A_eq[row_inv, iot.start + t] = 1   # ot[t]
        A_eq[row_inv, isb.start + t] = 1   # sub[t]
        A_eq[row_inv, ip.start + t]  = -1  # -ip[t]
        A_eq[row_inv, im.start + t]  =  1  # +im[t]
        A_eq[row_inv, ils.start + t] =  1  # +ls[t]
        if t == 0:
            b_eq[row_inv] = demand[t] - initial_inventory
        else:
            A_eq[row_inv, ip.start + t - 1] = 1   # +ip[t-1]
            A_eq[row_inv, im.start + t - 1] = -1  # -im[t-1]
            b_eq[row_inv] = demand[t]

        # Workforce balance
        A_eq[row_wf, iw.start + t]  = 1
        A_eq[row_wf, ih.start + t]  = -1
        A_eq[row_wf, iff.start + t] = 1
        if t == 0:
            b_eq[row_wf] = initial_workforce
        else:
            A_eq[row_wf, iw.start + t - 1] = -1
            b_eq[row_wf] = 0.0

    # ---- Inequality constraints (A_ub @ v <= b_ub) ----
    n_ub = 4 * T
    A_ub = np.zeros((n_ub, N))
    b_ub = np.zeros(n_ub)
    max_ot = capacity["max_overtime_fraction"]
    max_wf = capacity["max_workforce"]
    max_sub_pp = capacity["max_subcontract_per_period"]

    for t in range(T):
        A_ub[t,       ix.start + t]  =  1
        A_ub[t,       iw.start + t]  = -productivity
        A_ub[T + t,   iot.start + t] =  1
        A_ub[T + t,   iw.start + t]  = -max_ot * productivity
        A_ub[2*T + t, isb.start + t] = 1
        b_ub[2*T + t] = max_sub_pp
        A_ub[3*T + t, iw.start + t]  = 1
        b_ub[3*T + t] = max_wf

    # ---- Bounds — policy controls im and ls ----
    bounds = [(0, None)] * N
    if shortage_policy == "backorders":
        # im free >= 0, ls = 0
        for t in range(T):
            bounds[ils.start + t] = (0, 0)
    elif shortage_policy == "no_shortages":
        # im = 0, ls = 0
        for t in range(T):
            bounds[im.start + t]  = (0, 0)
            bounds[ils.start + t] = (0, 0)
    else:  # lost_sales
        # im = 0, ls free >= 0
        for t in range(T):
            bounds[im.start + t] = (0, 0)

    if integer_workforce or integer_production:
        integrality = np.zeros(N)
        if integer_workforce:
            integrality[ih]  = 1
            integrality[iff] = 1
            integrality[iw]  = 1
        if integer_production:
            integrality[ix]  = 1
            integrality[iot] = 1
            integrality[isb] = 1
        A_all = np.vstack([A_eq, A_ub])
        lb_bnd = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
        ub_bnd = np.array([b[1] if b[1] is not None else  np.inf for b in bounds])
        lb_all = np.concatenate([b_eq, np.full(len(b_ub), -np.inf)])
        ub_all = np.concatenate([b_eq, b_ub])
        constraints = LinearConstraint(A_all, lb_all, ub_all)
        var_bounds = Bounds(lb=lb_bnd, ub=ub_bnd)
        result = milp(c_obj, constraints=constraints,
                      integrality=integrality, bounds=var_bounds)
        use_milp = True
    else:
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                         A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                         method="highs")
        use_milp = False

    res = _empty_result(periods, demand)
    if result.status != 0:
        res["feasible"] = False
        res["message"] = f"LP infeasible or unbounded: {result.message}"
        return res

    v = result.x
    inv_arr = v[ip.start:ip.stop] - v[im.start:im.stop]

    for t in range(T):
        res["production"][t]  = v[ix.start + t] + v[iot.start + t] + v[isb.start + t]
        res["workforce"][t]   = v[iw.start + t]
        res["hired"][t]       = v[ih.start + t]
        res["fired"][t]       = v[iff.start + t]
        res["inventory"][t]   = inv_arr[t]
        res["overtime"][t]    = v[iot.start + t]
        res["subcontract"][t] = v[isb.start + t]
        res["lost_sales"][t]  = v[ils.start + t]

    res = _compute_period_costs(res, costs)

    # Shadow prices — LP (continuous) only
    if not use_milp:
        if result.ineqlin is not None or hasattr(result, "eqlin"):
            duals = getattr(result, "eqlin", None)
            if duals is not None and hasattr(duals, "marginals"):
                sp_inv = duals.marginals[:T]
                sp_wf  = duals.marginals[T:]
                res["shadow_prices"] = {
                    "Inventory Balance": [round(float(v), 4) for v in sp_inv],
                    "Workforce Balance": [round(float(v), 4) for v in sp_wf],
                }
        res["message"] = f"Optimal — grand total cost: ${res['grand_total']:,.0f}"
    else:
        res["message"] = f"Optimal (MILP) — grand total cost: ${res['grand_total']:,.0f}"
    return res


# ---------------------------------------------------------------------------
# 5. Transportation Method
# ---------------------------------------------------------------------------

def solve_transportation(periods, demand, costs, capacity, initial_workforce,
                         initial_inventory, productivity,
                         integer_workforce=False, integer_production=False,
                         shortage_policy="backorders"):
    """
    Build and solve a transportation tableau:
      Sources: initial inventory + (regular-time, overtime, subcontract) for each period
      Destinations: demand in each period
      Cell cost includes holding penalty for units produced early.
    """
    T = len(periods)

    c_rt  = costs["regular_time"] / productivity
    c_ot  = costs["overtime"]
    c_sub = costs["subcontracting"]
    c_hld = costs["holding"]
    c_bk  = costs["backorder"]

    # Average workforce → RT capacity per period
    avg_wf   = sum(demand) / T / productivity
    rt_cap   = min(avg_wf, capacity["max_workforce"]) * productivity
    ot_cap   = rt_cap * capacity["max_overtime_fraction"]
    sub_cap  = capacity["max_subcontract_per_period"]

    # Build source list
    # Each source: (name, capacity, base_cost, period_produced)
    sources = []
    if initial_inventory > 0:
        sources.append(("Initial Inv", float(initial_inventory), 0.0, -1))
    for t in range(T):
        sources.append((f"RT {periods[t]}",  rt_cap,  c_rt,  t))
        sources.append((f"OT {periods[t]}",  ot_cap,  c_ot,  t))
        sources.append((f"SC {periods[t]}",  sub_cap, c_sub, t))

    S = len(sources)
    D = list(float(d) for d in demand)

    # Cost matrix S × T (no backorders: ship only to current or future periods)
    BIG = 1e9
    cost_matrix = np.full((S, T), BIG)
    for i, (name, cap, base, tp) in enumerate(sources):
        for j in range(T):
            if tp <= j:            # produced in period tp, consumed in period j
                holding_periods = j - tp  if tp >= 0 else j + 1
                cost_matrix[i, j] = base + c_hld * holding_periods
            # backorders not modelled (cost = BIG effectively blocks them)

    # Solve as LP: min c·x  s.t. sum_j x_ij <= cap_i, sum_i x_ij = D_j, x>=0
    caps = np.array([s[1] for s in sources], dtype=float)

    # Variable layout: x[i*T + j]
    n_vars = S * T
    c_lp   = cost_matrix.flatten()

    # Equality: demand satisfaction for each period j: sum_i x[i*T+j] = D[j]
    A_eq = np.zeros((T, n_vars))
    b_eq = np.array(D)
    for j in range(T):
        for i in range(S):
            A_eq[j, i*T + j] = 1

    # Inequality: supply capacity for each source i: sum_j x[i*T+j] <= cap_i
    A_ub = np.zeros((S, n_vars))
    b_ub = caps.copy()
    for i in range(S):
        for j in range(T):
            A_ub[i, i*T + j] = 1

    bounds = [(0, None)] * n_vars
    if integer_production:
        A_all = np.vstack([A_eq, A_ub])
        lb_all = np.concatenate([b_eq, np.full(len(b_ub), -np.inf)])
        ub_all = np.concatenate([b_eq, b_ub])
        constraints = LinearConstraint(A_all, lb_all, ub_all)
        var_bounds = Bounds(lb=np.zeros(n_vars), ub=np.full(n_vars, np.inf))
        result = milp(c_lp, constraints=constraints,
                      integrality=np.ones(n_vars), bounds=var_bounds)
    else:
        result = linprog(c_lp, A_ub=A_ub, b_ub=b_ub,
                         A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    res = _empty_result(periods, demand)
    if result.status != 0:
        res["feasible"] = False
        res["message"] = f"Transportation LP infeasible: {result.message}"
        return res

    x_mat = result.x.reshape(S, T)

    # Aggregate back to period-level output variables
    rt_prod  = np.zeros(T)
    ot_prod  = np.zeros(T)
    sub_prod = np.zeros(T)
    # Map each source row to type
    for i, (name, cap, base, tp) in enumerate(sources):
        if tp < 0:
            continue  # initial inventory row
        if name.startswith("RT"):
            rt_prod[tp]  += x_mat[i].sum()
        elif name.startswith("OT"):
            ot_prod[tp]  += x_mat[i].sum()
        else:
            sub_prod[tp] += x_mat[i].sum()

    inv = float(initial_inventory)
    avg_wf_level = avg_wf
    wf = float(initial_workforce)
    for t in range(T):
        prod = rt_prod[t] + ot_prod[t] + sub_prod[t]
        inv_raw = inv + prod - demand[t]
        if shortage_policy == "lost_sales" and inv_raw < 0:
            ls = -inv_raw
            inv = 0.0
        else:
            ls = 0.0
            inv = inv_raw
        new_wf = avg_wf_level
        res["production"][t]  = prod
        res["workforce"][t]   = new_wf
        res["hired"][t]       = max(0.0, new_wf - wf)
        res["fired"][t]       = max(0.0, wf - new_wf)
        res["inventory"][t]   = inv
        res["overtime"][t]    = ot_prod[t]
        res["subcontract"][t] = sub_prod[t]
        res["lost_sales"][t]  = ls
        wf = new_wf

    res = _compute_period_costs(res, costs)

    # Build tableau dict for display
    tableau_data = {
        "sources": [s[0] for s in sources],
        "capacities": [s[1] for s in sources],
        "allocations": x_mat.tolist(),
        "cell_costs":  cost_matrix.tolist(),
    }
    res["transport_tableau"] = tableau_data
    res["message"] = f"Optimal — grand total cost: ${res['grand_total']:,.0f}"
    return res


# ---------------------------------------------------------------------------
# 6. Trial-and-Error (user-defined plan)
# ---------------------------------------------------------------------------

def solve_trial(periods, demand, costs, capacity, initial_workforce,
                initial_inventory, productivity,
                user_production=None, user_workforce=None,
                user_overtime=None, user_subcontract=None,
                integer_workforce=False, integer_production=False,
                shortage_policy="backorders"):
    """
    User manually specifies per-period values. Solver validates feasibility
    and computes costs without optimising.
    """
    T = len(periods)
    res = _empty_result(periods, demand)

    prod  = [float(p) for p in (user_production   or demand)]
    wf_in = [float(w) for w in (user_workforce    or [initial_workforce]*T)]
    ot_in = [float(o) for o in (user_overtime     or [0.0]*T)]
    sb_in = [float(s) for s in (user_subcontract  or [0.0]*T)]

    inv = float(initial_inventory)
    wf  = float(initial_workforce)
    warnings = []

    for t in range(T):
        new_wf = wf_in[t]
        hired  = max(0.0, new_wf - wf)
        fired  = max(0.0, wf - new_wf)
        rt_prod = prod[t] - ot_in[t] - sb_in[t]
        if rt_prod < 0:
            warnings.append(f"Period {periods[t]}: OT+Sub exceeds total production.")
        inv_raw = inv + prod[t] - demand[t]
        if shortage_policy == "lost_sales" and inv_raw < 0:
            ls = -inv_raw
            inv = 0.0
        else:
            ls = 0.0
            inv = inv_raw

        res["production"][t]  = prod[t]
        res["workforce"][t]   = new_wf
        res["hired"][t]       = hired
        res["fired"][t]       = fired
        res["inventory"][t]   = inv
        res["overtime"][t]    = ot_in[t]
        res["subcontract"][t] = sb_in[t]
        res["lost_sales"][t]  = ls
        wf = new_wf

    res = _compute_period_costs(res, costs)
    if warnings:
        res["feasible"] = False
        res["message"]  = " | ".join(warnings)
    else:
        res["message"] = f"User plan evaluated — grand total: ${res['grand_total']:,.0f}"
    return res


# ---------------------------------------------------------------------------
# Utility: run all strategies and return comparison table
# ---------------------------------------------------------------------------

def compare_all(periods, demand, costs, capacity, initial_workforce,
                initial_inventory, productivity,
                integer_workforce=False, integer_production=False,
                shortage_policy="backorders"):
    strategies = {
        "Chase":          solve_chase,
        "Level":          solve_level,
        "Mixed":          solve_mixed,
        "LP (Optimal)":   solve_lp,
        "Transportation": solve_transportation,
    }
    rows = []
    for name, fn in strategies.items():
        r = fn(periods, demand, costs, capacity,
               initial_workforce, initial_inventory, productivity,
               integer_workforce=integer_workforce,
               integer_production=integer_production,
               shortage_policy=shortage_policy)
        rows.append({
            "Strategy":       name,
            "Total Cost ($)": round(r["grand_total"], 0),
            "Total Hired":    round(sum(r["hired"]), 1),
            "Total Fired":    round(sum(r["fired"]), 1),
            "Avg Inventory":  round(sum(r["inventory"]) / len(periods), 1),
            "Total OT Units": round(sum(r["overtime"]), 1),
            "Total Sub Units":round(sum(r["subcontract"]), 1),
            "Lost Sales":     round(sum(r["lost_sales"]), 1),
            "Feasible":       r["feasible"],
        })
    return pd.DataFrame(rows)
