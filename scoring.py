"""
scoring.py  —  Candidate implementation with deep comments

Purpose (plain English):
------------------------
This file implements the *lexicographic convex program* you designed:

  Phase 1 (primary goal):      Maximize routed volume T
  Phase 2 (secondary goal):    Minimize payout cost C while keeping T ≈ T1

…under these constraints:
  - Budget cap                         (sum payouts <= Budget)
  - Payout-to-volume cap (kappa_bar)   (sum payouts <= kappa_bar * T)
  - Diversity cap per miner            (miner_i share <= rho_cap * T)
  - Eligibility                        (only miners above ROI_min & V_min can get x > 0)
  - Ramp (smoothness)                  (x - x_prev bounded each epoch)
  - Bounds                             (0 <= x <= 1)

Key modeling choices:
---------------------
- Decision variable x[i]  ∈ [0,1] is the fraction of miner i’s "epoch" we fund this epoch.
- We approximate miner i’s *epoch* by their last-epoch qualified volume v_prev[i].
- So the routed volume contributed by miner i this epoch ≈ v_prev[i] * x[i].
- Total routed volume T = sum_i v_prev[i] * x[i].

- Payout cost vector c[i] = v_prev[i] * max(roi_prev[i], 0.0).
  *Why this is OK:* We want higher ROI miners to cost more per funded dollar so that
  Phase 2 has a meaningful “minimize cost” tie-break. Eligibility already prevents
  negative or tiny ROI miners from flowing through; clamping to >= 0 keeps the LP simple.

- Diversity (rho_cap): Each miner’s share of volume is limited:
      v_prev[i] * x[i] <= rho_cap * T
  This prevents a single miner from dominating the flow.

- Ramp (ramp): Limits how much each x[i] can move relative to last epoch:
      -ramp <= x[i] - x_prev[i] <= ramp
  This keeps flow changes smooth. We set a single global ramp scalar from HHI.

- HHI-based ramp: ramp = sum(share^2), where share = All_volumes / sum(All_volumes).
  Intuition: If the field is concentrated (few big miners), ramp is higher (system can
  pivot faster). If broad (many similar miners), ramp is lower (slower changes, smoother).

Data & defaults (for demo):
---------------------------
- Budget, Volume_prev, Total_volume, All_volumes, ROI_min, V_min, x_prev, roi_prev, v_prev.
- You can wire these to real epoch data in production.
- ECOS solver is used (install: pip install ecos). If missing, use SCS.

Outputs:
--------
- Phase 1: T1*, x1*, and duals (shadow prices for each constraint).
- Phase 2: C*, x2*, T2, and duals.
- Payouts P[i] = c[i] * x_opt[i] (opt from Phase 2, else Phase 1).
- Printed dual “scoreboard” table with Greek symbols, names, and values.

Reading the duals (dashboard idea):
-----------------------------------
- λ_B   (lambda_B):     how tight the budget is.
- λ_κ   (lambda_k):     how tight the payout-to-volume ratio is.
- λ_i   (lambda_i[i]):  which miner i’s diversity cap is binding.
- μ     (mu):           whether eligibility is biting (x <= eligible).
- ρ⁺/ρ⁻ (rho_plus/minus): how much ramp (smoothness) is costing us (upper/lower).
- ν⁺/ν⁻ (nu_plus/minus): whether x is stuck against [0,1] bounds.
- α     (alpha):        link constraint T = v·x (technical dual).
- η     (eta, Phase 2): cost of forcing T ≥ (1−ε) T1 (locking max volume).

IMPORTANT UNITS / INTERPRETATION:
---------------------------------
- roi_prev is a *fraction*, e.g., 0.05 means 5% ROI. Do NOT pass 5 or 100 for 5%.
- v_prev is in dollars (or any unit of qualified volume).
- c has units of payout (token) per unit x (since x is a fraction of the epoch);
  c @ x gives total payout tokens.
- T is in the same units as volume (v_prev @ x).

"""

import numpy as np
import cvxpy as cp
from collections import defaultdict
from typing import Dict, Any, List
from datetime import datetime, timedelta, timezone
from constants import (
  ROLLING_WINDOW_IN_DAYS,
  ROI_MIN,
  VOLUME_MIN,
  VOLUME_FEE,
  RAMP,
  RHO_CAP,
  GENERAL_POOL_WEIGHT_PERCENTAGE,
  MINER_WEIGHT_PERCENTAGE,
)

def score_miners(
    all_uids: List[int],
    all_hotkeys: List[str],
    trading_history: Dict[str, Any],
    verbose: bool = True,
    target_epoch_idx: int = None
):
    """
    Score the miners based on the trading history.
    
    Creates epoch-based numpy matrices (similar to simulate_epochs.py) where:
    - Rows = epochs (days), indexed 0 to 29 for last 30 days
    - Columns = entities (miner_ids for miners, profile_ids for general pool)
    - Separates miners from general pool users
    """

    # Convert the trading history to match the format expected by the scoring function
    """
    {
        "trade_id":2206,
        "profile_id":"0x1234567890abcdef",
        "miner_id":44,
        "miner_hotkey":"5F12345",
        "is_general_pool":true,
        "market_id":"nba_celtics_knicks_2025",
        "date_created":"2025-07-22",
        "volume":859,
        "pnl":621.858553239,
        "is_correct":true,
        "is_settled":true,
        "date_settled":"2025-07-25",
        "trade_type":"buy",
        "price":0.5
    }
    """

    # Build epoch-based data structures similar to simulate_epochs.py
    miner_history = build_epoch_history(
        trading_history=trading_history,
        all_uids=all_uids,
        all_hotkeys=all_hotkeys,
        is_miner_pool=True,
        target_epoch_idx=target_epoch_idx
    )
    
    general_pool_history = build_epoch_history(
        trading_history=trading_history,
        all_uids=all_uids,
        all_hotkeys=all_hotkeys,
        is_miner_pool=False,
        target_epoch_idx=target_epoch_idx
    )
    
    # Calculate budget from fees collected for the target epoch
    # If target_epoch_idx is None, use the last epoch (current behavior)
    epoch_idx = target_epoch_idx if target_epoch_idx is not None else -1
    miner_fees = np.sum(miner_history["fees_prev"][epoch_idx]) if miner_history["n_entities"] > 0 else 0.0
    gp_fees = np.sum(general_pool_history["fees_prev"][epoch_idx]) if general_pool_history["n_entities"] > 0 else 0.0
    current_epoch_budget = miner_fees + gp_fees
    
    # Calculate the budget for each pool based on our constants
    miner_pool_epoch_budget = current_epoch_budget * MINER_WEIGHT_PERCENTAGE
    general_pool_epoch_budget = current_epoch_budget * GENERAL_POOL_WEIGHT_PERCENTAGE

    # Calculate the miner pool scores using epoch-based history
    miners_scores = score_with_epochs(
        epoch_history=miner_history,
        Budget=miner_pool_epoch_budget,
        ROI_min=ROI_MIN,
        V_min=VOLUME_MIN,
        verbose=verbose
    )
    
    # Calculate the general pool scores using epoch-based history
    general_pool_scores = score_with_epochs(
        epoch_history=general_pool_history,
        Budget=general_pool_epoch_budget,
        ROI_min=ROI_MIN,
        V_min=VOLUME_MIN,
        verbose=verbose
    )

    return miner_history, general_pool_history, miners_scores, general_pool_scores, miner_pool_epoch_budget, general_pool_epoch_budget


def build_epoch_history(
    trading_history: List[Dict[str, Any]],
    all_uids: List[int],
    all_hotkeys: List[str],
    is_miner_pool: bool,
    target_epoch_idx: int = None
):
    """
    Build epoch-based numpy matrices similar to simulate_epochs.py.
    
    Returns a dictionary with:
    - v_prev: (n_epochs, n_entities) - qualified volume per epoch
    - profit_prev: (n_epochs, n_entities) - profit per epoch
    - fees_prev: (n_epochs, n_entities) - fees per epoch
    - unqualified_prev: (n_epochs, n_entities) - losing volume per epoch
    - entity_ids: list of miner_ids or profile_ids
    - entity_map: dict mapping entity_id -> column index
    - epoch_dates: list of date strings for each epoch (index 0 = oldest)
    """
    # Determine date range and number of epochs
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = today - timedelta(days=ROLLING_WINDOW_IN_DAYS)
    
    # Determine number of epochs based on target_epoch_idx
    if target_epoch_idx is not None:
        # For historical simulation: create epochs 0 through target_epoch_idx
        n_epochs = target_epoch_idx + 1
    else:
        # For current epoch: use full rolling window (default behavior)
        n_epochs = ROLLING_WINDOW_IN_DAYS
    
    # Create list of epoch dates (0 = oldest, n_epochs-1 = most recent)
    epoch_dates = [(start_date + timedelta(days=i)).date() for i in range(n_epochs)]
    
    # First pass: collect all entity IDs and organize trades by epoch
    entity_set = set()
    epoch_trades = defaultdict(list)  # epoch_idx -> list of trades
    
    for trade in trading_history:
        if not trade["is_settled"]:
            continue
            
        # Parse date
        date_settled = trade["date_settled"]
        if isinstance(date_settled, str):
            date_settled = datetime.fromisoformat(date_settled.replace('Z', '+00:00'))
        trade_date = date_settled.date()
        
        # Find epoch index (0-29)
        if trade_date < epoch_dates[0] or trade_date >= today.date():
            continue  # outside our window
        epoch_idx = (trade_date - epoch_dates[0]).days
        
        # Filter by pool type
        if is_miner_pool:
            # Miner pool: validate miner
            if trade["is_general_pool"]:
                continue
            miner_id = trade.get("miner_id")
            miner_hotkey = trade.get("miner_hotkey")
            if miner_id is None or miner_hotkey is None:
                continue
            # Validate: miner_id exists, has a registered hotkey, and hotkey matches
            if miner_id not in all_uids or miner_id not in all_hotkeys or all_hotkeys[miner_id] != miner_hotkey:
                continue
            entity_id = miner_id
        else:
            # General pool
            if not trade["is_general_pool"]:
                continue
            entity_id = trade["profile_id"]
            
        entity_set.add(entity_id)
        epoch_trades[epoch_idx].append((entity_id, trade))
    
    # Create entity mapping
    entity_ids = sorted(list(entity_set))
    entity_map = {eid: idx for idx, eid in enumerate(entity_ids)}
    n_entities = len(entity_ids)
    # n_epochs was already calculated above based on target_epoch_idx
    
    # Initialize matrices (like simulate_epochs.py)
    v_prev = np.zeros((n_epochs, n_entities))             # qualified volume
    profit_prev = np.zeros((n_epochs, n_entities))        # profit
    fees_prev = np.zeros((n_epochs, n_entities))          # fees collected
    unqualified_prev = np.zeros((n_epochs, n_entities))   # losing volume
    trade_counts = np.zeros((n_epochs, n_entities))       # number of trades
    
    # Second pass: populate matrices
    for epoch_idx in range(n_epochs):
        if epoch_idx not in epoch_trades:
            continue
            
        for entity_id, trade in epoch_trades[epoch_idx]:
            col_idx = entity_map[entity_id]
            
            volume = trade["volume"]
            pnl = trade["pnl"]
            is_correct = trade["is_correct"]
            
            # Calculate metrics (matching simulate_epochs.py logic)
            fee = volume * VOLUME_FEE
            
            if is_correct:
                # Winning trade: qualified volume (after fee deduction)
                qualified = volume * (1.0 - VOLUME_FEE)
                v_prev[epoch_idx, col_idx] += qualified
                profit_prev[epoch_idx, col_idx] += pnl
            else:
                # Losing trade: unqualified volume
                unqualified_prev[epoch_idx, col_idx] += volume
                profit_prev[epoch_idx, col_idx] += pnl  # negative profit
            
            fees_prev[epoch_idx, col_idx] += fee
            trade_counts[epoch_idx, col_idx] += 1  # Count each trade
    
    return {
        "v_prev": v_prev,
        "profit_prev": profit_prev,
        "fees_prev": fees_prev,
        "unqualified_prev": unqualified_prev,
        "trade_counts": trade_counts,
        "entity_ids": entity_ids,
        "entity_map": entity_map,
        "epoch_dates": [str(d) for d in epoch_dates],
        "n_epochs": n_epochs,
        "n_entities": n_entities
    }

def score_with_epochs(
    epoch_history: Dict[str, Any],
    Budget: float,
    ROI_min: float,
    V_min: float,
    verbose: bool = True
):
    """
    Score entities using epoch-based history (similar to simulate_epochs.py).
    
    Takes the epoch history matrices and calculates:
    1. Trailing aggregates (total volume, total profit, ROI)
    2. Latest epoch data (for v_prev, roi_prev)
    3. Runs Phase 1 and Phase 2 optimization
    4. Returns scores/payouts for each entity
    """
    v_prev_matrix = epoch_history["v_prev"]           # (n_epochs, n_entities)
    profit_prev_matrix = epoch_history["profit_prev"] # (n_epochs, n_entities)
    entity_ids = epoch_history["entity_ids"]
    n_entities = epoch_history["n_entities"]
    
    # If no entities, return empty
    if n_entities == 0:
        return {
            "entity_ids": [],
            "scores": np.array([]),
            "x_opt": np.array([]),
            "sol1": None,
            "sol2": None
        }
    
    # Calculate trailing aggregates (sum across all epochs, axis=0)
    total_volume = np.sum(v_prev_matrix, axis=0)      # shape: (n_entities,)
    total_profit = np.sum(profit_prev_matrix, axis=0)  # shape: (n_entities,)
    
    # Calculate trailing ROI for each entity
    roi_trailing = np.divide(
        total_profit,
        np.maximum(total_volume, 1e-12),
        out=np.zeros_like(total_profit),
        where=total_volume > 0
    )
    
    # Get latest epoch data (last row = most recent epoch, index -1)
    current_epoch_v = v_prev_matrix[-1] if v_prev_matrix.shape[0] > 0 else np.zeros(n_entities)
    
    # Qualified mask: entities with positive trailing ROI
    qualified_mask = roi_trailing > 0
    v_qualified = current_epoch_v * qualified_mask       # qualified volume for this epoch
    roi_qualified = roi_trailing * qualified_mask
    
    # Initialize x_prev (gates from previous epoch)
    # For now, start at 0 - in production you'd load from persistent storage
    x_prev = np.zeros(n_entities)
    
    # Calculate kappa_bar using the epoch history
    kappa_bar = compute_joint_kappa_from_history(epoch_history)
    
    # Calculate HHI-based ramp
    #ramp = calculate_hhi_ramp(total_volume)
    #ramp = 0.1
    ramp = RAMP
    
    # Load parameters for optimization
    p_dict = load_params(
        Budget      = Budget,
        Volume_prev = np.sum(v_qualified),       # total qualified vol this epoch
        Total_volume= np.sum(total_volume),      # trailing total
        All_volumes = total_volume.copy(),
        ROI_min     = ROI_min,
        V_min       = V_min,
        x_prev      = x_prev,
        roi_prev    = roi_qualified,
        v_prev      = v_qualified,
        kappa_bar   = kappa_bar,
        ramp        = ramp,
        rho_cap     = RHO_CAP,
        v_trailing  = total_volume
    )
    
    """
    Solver Phase 1: Maximize routed qualified volume
    """
    sol1 = solve_phase1(p_dict, verbose=verbose)
    x_star = sol1.get("x_star")
    
    """
    Solver Phase 2: Minimize cost given T*
    """
    sol2 = None
    if sol1["status"] in ("optimal", "optimal_inaccurate") and sol1.get("T_star", 0) > 0:
        sol2 = solve_phase2(p_dict, T1=sol1["T_star"], verbose=verbose)
        if sol2.get("x_star") is not None:
            x_star = sol2["x_star"]
    
    # Use the optimal x from Phase 2 if available, else Phase 1
    x_opt = np.clip(x_star, 0.0, 1.0) if x_star is not None else np.zeros(n_entities)
    
    # Calculate payouts (scores)
    c = v_qualified * np.maximum(roi_qualified, 0.0)
    scores = c * x_opt
    
    return {
        "entity_ids": entity_ids,
        "scores": scores,
        "x_opt": x_opt,
        "sol1": sol1,
        "sol2": sol2,
        "total_volume": total_volume,
        "total_profit": total_profit,
        "roi_trailing": roi_trailing
    }


def calculate_hhi_ramp(volumes: np.ndarray) -> float:
    """
    Calculate HHI-based ramp from volume distribution.
    
    HHI = sum(share^2) where share = volume_i / total_volume
    Higher HHI (concentrated) -> higher ramp (faster adjustments)
    Lower HHI (distributed) -> lower ramp (slower, smoother adjustments)
    """
    total = np.sum(volumes)
    if total < 1e-12:
        return 0.1  # default
    
    shares = volumes / total
    hhi = np.sum(shares ** 2)
    
    # Scale HHI to reasonable ramp range (e.g., 0.05 to 0.2)
    ramp = np.clip(hhi * 0.5, 0.05, 0.2)
    return float(ramp)


def compute_joint_kappa_from_history(epoch_history: Dict[str, Any], gamma=1.5, lookback=50, smooth=0.3) -> float:
    """
    Compute kappa_bar from epoch history matrices.
    
    Similar to compute_joint_kappa in simulate_epochs.py but adapted for
    the epoch_history dictionary structure.
    """
    v_prev = epoch_history["v_prev"]
    profit_prev = epoch_history["profit_prev"]
    
    # Bootstrap if too little history
    if v_prev.shape[0] < 5:
        return 0.05
    
    # Slice last 'lookback' epochs
    V_hist = v_prev[-lookback:] if v_prev.shape[0] >= lookback else v_prev
    P_hist = profit_prev[-lookback:] if profit_prev.shape[0] >= lookback else profit_prev
    
    # Per-epoch totals and ROIs
    vols = np.sum(V_hist, axis=1)                                    # shape: (L,)
    rois = np.sum(P_hist, axis=1) / np.maximum(vols, 1e-12)          # epoch ROI
    
    # Align with optimizer's cost definition (c uses max(roi, 0.0))
    rois_pos = np.maximum(rois, 0.0)
    
    # Volume normalization centered at 1: penalize only *above-average* volume
    V_bar = np.mean(vols)
    ratio = vols / np.maximum(V_bar, 1e-12)
    penalty = 1.0 + gamma * np.maximum(ratio - 1.0, 0.0)
    
    # Adjusted per-epoch ROI with volume penalty, then volume-weighted average
    adj = rois_pos / penalty
    w = vols / np.maximum(np.sum(vols), 1e-12)
    J_t = float(np.dot(adj, w))
    
    # For first call, just return J_t
    # In production, you'd store and smooth this value across calls
    kappa_bar = max(J_t, 1e-12)
    
    return kappa_bar


"""
Function:  load_params
Purpose:   loads all of the required information into a useful dictionary.
Return:    <Dict>
"""
def load_params(Budget, Volume_prev, Total_volume, All_volumes, ROI_min, V_min, x_prev, roi_prev, v_prev, kappa_bar, ramp, rho_cap, v_trailing): 
    #print({"ramp": ramp, "k": kappa_bar, "rho_cap": rho_cap, "ROI_min": ROI_min, "V_min": V_min})
    epsilon   = 1e-4
    return {
        "Budget": float(Budget),
        "kappa_bar": float(kappa_bar),
        "ROI_min": float(ROI_min),
        "V_min": float(V_min),
        "rho_cap": float(rho_cap),
        "ramp": float(ramp),
        "epsilon": float(epsilon),
        "x_prev": np.clip(np.asarray(x_prev, dtype=float).ravel(), 0.0, 1.0),
        "roi_prev": np.asarray(roi_prev, dtype=float).ravel(),
        "v_prev": np.asarray(v_prev, dtype=float).ravel(),
        "ramp": float(ramp),
        "v_trailing": v_trailing
    }

def solve_phase1(p, verbose=True):
    """
    Phase 1: Maximize routed qualified volume given budget & payout constraints. 
    """
    #########################################
    ## Historical data and size
    #########################################
    v_prev, roi_prev, x_prev = p["v_prev"], p["roi_prev"], p["x_prev"]
    n = v_prev.size

    #########################################
    ## Numerical scaling
    ## First we have to make a scale so the 
    ## optimizer can properly find an interior
    ## point
    #########################################
    v_prev = np.maximum(v_prev, 1e-12)       # previous volume
    scale = max(float(np.mean(v_prev)), 1.0) # scaling param
    v_scaled = v_prev / scale                # scaled volume
    B = p["Budget"]                          # budget size
    kappa_scaled = p["kappa_bar"]            # dimensionless
    rho_cap = p.get("rho_cap", 0.1)          # diversity
    ramp = p.get("ramp", 0.1)                # ramp
    c = v_prev * np.maximum(roi_prev, 0.0)   # costs
    v_trailing = p.get("v_trailing", v_prev) # trailing volume
 
    #########################################
    ## Eligibility Gates
    ## We then have to generate a cost per units
    ## of signal and then using trailing volume
    ## make a boolean which flags eligibility
    #########################################
    eligible = (
        (roi_prev >= p["ROI_min"]) & 
        (v_trailing >= p["V_min"]) &
        (v_prev > 0)
    ).astype(float)

    #########################################
    ## Get the number of Eligible uids
    ## Not everyone bets in an epoch so we have
    ## to shutdown the budget if people simply
    ## do not bet.
    #########################################
    N_total = len(v_prev)
    N_eligible = int(np.sum(eligible))
 
    # 3) Decision variable
    x = cp.Variable(n)

    # 4) Constraints
    eps = 1e-9
    cons = []
    cons += [x >= 0, x <= 1]
    cons += [x <= eligible]                                         # eligibility
    cons += [c @ x <= B]                                            # budget
    cons += [c @ x <= kappa_scaled * ((v_prev @ x)+eps)]            # payout/volume cap
    cons += [x - x_prev <= ramp, x_prev - x <= ramp]                # ramp
    cons += [c[i] * x[i] <= rho_cap * B for i in range(n)]          # diversity
    
    # 5) Objective
    prob = cp.Problem(cp.Maximize(v_prev @ x), cons)
    prob.solve(solver=cp.ECOS, verbose=False)

    # 6) === Dual diagnostics table ===
    if prob.status in ("optimal", "optimal_inaccurate") and verbose:
        print("\n====================[ Phase 1 Dual Summary ]====================")

        labels = [
            ("x >= 0",            cons[0]),
            ("x <= 1",            cons[1]),
            ("x <= eligible",     cons[2]),
            ("budget cap",        cons[3]),
            ("ROI cap",           cons[4]),
            ("ramp upper",        cons[5]),
            ("ramp lower",        cons[6]),
        ]

        # Diversity group (vector of n)
        diversity_cons = cons[7:]

        rows = []
        for name, cstr in labels:
            if hasattr(cstr, "dual_value") and cstr.dual_value is not None:
                duals = np.array(cstr.dual_value, dtype=float)
                mag = np.max(np.abs(duals))
                rows.append((name, mag))
            else:
                rows.append((name, 0.0))

        # Handle diversity separately
        mags = []
        for c in diversity_cons:
            if hasattr(c, "dual_value") and c.dual_value is not None:
                mags.append(np.max(np.abs(np.array(c.dual_value, dtype=float))))
        rows.append(("diversity (all)", float(np.max(mags)) if mags else 0.0))

        # Pretty print summary
        print(f"{'Constraint':35s} | {'Dual Magnitude':>15s}")
        print("-" * 55)
        for name, mag in rows:
            marker = "⛔" if mag > 1e-6 else " "
            print(f"{name:35s} | {mag:15.6f} {marker}")
        print("=" * 55 + "\n")

    # 6) Return (moved outside verbose block)
    x_star = None if x.value is None else x.value.copy()
    T_star = float(v_prev @ x_star) if x_star is not None else 0.0
    payout = float(np.dot(v_prev * np.maximum(roi_prev, 0.0), x_star)) if x_star is not None else 0.0
    num_funded = int(np.sum((v_prev * np.maximum(roi_prev, 0.0) * x_star) > 1e-9)) if x_star is not None else 0
    num_eligible = int(np.sum(eligible))

    return {
        "status": prob.status,
        "T_star": T_star,   # in the same units as your printed "qualified"
        "x_star": x_star,
        "payout": payout,
        "num_funded": num_funded,
        "num_eligible": num_eligible,
    }

def solve_phase2(p, T1, verbose=True):
    """
    Phase 2: Minimize payout cost C = c @ x
    subject to:
        - All Phase 1 constraints
        - T >= (1 - epsilon) * T1  (lock in max volume)
    """
    #########################################
    ## Historical data and size
    #########################################
    v_prev, roi_prev, x_prev = p["v_prev"], p["roi_prev"], p["x_prev"]
    n = v_prev.size
    eps = p["epsilon"]

    #########################################
    ## Numerical scaling
    ## First we have to make a scale so the 
    ## optimizer can properly find an interior
    ## point
    #########################################
    v_prev = np.maximum(v_prev, 1e-12)
    scale = max(float(np.mean(v_prev)), 1.0)
    v_scaled = v_prev / scale
    B = p["Budget"]
    kappa = p["kappa_bar"]               # dimensionless
    rho_cap = p.get("rho_cap", 0.1)      # diversity
    ramp = p.get("ramp", 1.0)            # ramp

    #########################################
    ## Eligibility Gates
    ## We then have to generate a cost per units
    ## of signal and then using trailing volume
    ## make a boolean which flags eligibility
    #########################################
    c = v_prev * np.maximum(roi_prev, 0.0)
    v_trailing = p.get("v_trailing", v_prev)
    eligible = (
        (roi_prev >= p["ROI_min"]) &
        (v_trailing >= p["V_min"]) &
        (v_prev > 0)
    ).astype(float)

    #########################################
    ## Get the number of Eligible uids
    ## Not everyone bets in an epoch so we have
    ## to shutdown the budget if people simply
    ## do not bet.
    #########################################
    N_total = len(v_prev)
    N_eligible = int(np.sum(eligible))

    # Variables
    x = cp.Variable(n)
    T = cp.Variable()

    # Constraints
    cons = []
    cons += [x >= 0, x <= 1]
    cons += [x <= eligible]
    cons += [c @ x <= B]
    cons += [c @ x <= kappa * (T+eps)]
    cons += [x - x_prev <= ramp, x_prev - x <= ramp]
    cons += [c[i] * x[i] <= rho_cap * B for i in range(n)]
    ## Flow Equivalency.
    cons += [T == v_prev @ x]
    cons += [T >= (1.0 - eps) * T1]
 
    # Solve: minimize payout cost
    prob = cp.Problem(cp.Minimize(c @ x), cons)
    prob.solve(solver=cp.ECOS, verbose=False)

    return {
        "status": prob.status,
        "payout": None if x.value is None else float(c @ x.value),
        "x_star": None if x.value is None else x.value.copy(),
        "T_val": None if T.value is None else float(T.value),
        "C_star": None if x.value is None else (c * x.value).copy()
    }

"""
Function: compute_joint_kappa

Purpose: Computes the value of kappa as an endogenous variable that depends
          on previous volume and roi.  The goal is to restrict the payout rate
          of the budget by setting the “exchange rate” between qualified flow and token budget.
"""
def compute_joint_kappa(sim, gamma=1.5, lookback=50, smooth=0.3):
    # bootstrap if too little history
    if sim.v_prev.shape[0] < 5:
        return 0.05

    # slice last 'lookback' epochs (handles lookback > history gracefully)
    V_hist = sim.v_prev[-lookback:]
    P_hist = sim.profit_prev[-lookback:]

    # per-epoch totals and ROIs
    vols = np.sum(V_hist, axis=1)                                     # shape: (L,)
    rois = np.sum(P_hist, axis=1) / np.maximum(vols, 1e-12)           # epoch ROI

    # IMPORTANT: align with optimizer's cost definition (c uses max(roi, 0.0))
    rois_pos = np.maximum(rois, 0.0)

    # volume normalization centered at 1: penalize only *above-average* volume
    V_bar = np.mean(vols)
    ratio = vols / np.maximum(V_bar, 1e-12)
    penalty = 1.0 + gamma * np.maximum(ratio - 1.0, 0.0)              # no shrink at ratio≈1

    # adjusted per-epoch ROI with a volume penalty, then volume-weighted average
    adj = rois_pos / penalty
    w = vols / np.maximum(np.sum(vols), 1e-12)                         # volume weights sum to 1
    J_t = float(np.dot(adj, w))                                        # weighted mean

    # exponential smoothing (unchanged)
    if not hasattr(sim, "joint_kappa"):
        sim.joint_kappa = J_t
    else:
        sim.joint_kappa = (1 - smooth) * sim.joint_kappa + smooth * J_t

    # numeric floor only
    kappa_bar = max(float(sim.joint_kappa), 1e-12)
    return kappa_bar


def compute_payouts(p, x_opt):
    """
    Compute individual miner payouts and total payout given allocations x_opt.

    - Payout vector P[i] = c[i] * x_opt[i]
      where c[i] = v_prev[i] * max(roi_prev[i], 0.0).

    Sanity check (not enforced here):
      sum(P) <= Budget
      sum(P) <= kappa_bar * T_final

    Returns (P, P_total).
    """
    v_prev   = p["v_prev"]
    roi_prev = p["roi_prev"]
    c = v_prev * np.maximum(roi_prev, 0.0)
    P = c * x_opt
    return P, float(P.sum())


def publish_update(p, sol1, sol2, x_opt, T_final, P, P_total):
    """
    Minimal "publish/update" epoch for demo purposes:
      - Print summary
      - Suggest next kappa_bar = Budget / T_final
      - Update x_prev = x_opt (carry-over to next epoch)

    In a real validator:
      - You would record & sign the primal-dual certificates.
      - You would commit all values on-chain/off-chain as needed.
      - You would feed duals to a dashboard.
    """
    print("---- Publish & Update ----")
    print("Budget:", p["Budget"])
    print("kappa_bar:", p["kappa_bar"])
    print("Final T:", T_final)
    print("Total payout:", P_total)

    # Simple, data-driven suggestion for next epoch's kappa (no arbitrary dials):
    #   next_kappa = Budget / T_final   (use EWMA(T) if you want smoothing)
    if T_final and T_final > 0:
        next_kappa = p["Budget"] / T_final
        print("Suggested next kappa_bar:", next_kappa)

    # Carry over allocations to next epoch (for ramp)
    x_prev_next = np.clip(x_opt.copy(), 0.0, 1.0)
    print("x_prev updated (first 10):", x_prev_next[:10])


# ------------------------- Duals table (pretty print) --------------------------

def _fmt_scalar(val):
    """Format a scalar dual neatly for display."""
    return f"{float(val):.6g}" if val is not None else "None"

def _fmt_vector_stats(vec):
    """Summarize a vector of duals: mean, max, and how many are > 0 (binding)."""
    if vec is None:
        return "None"
    v = np.asarray(vec, dtype=float).flatten()
    if v.size == 0:
        return "[]"
    mean = np.mean(v)
    mx = np.max(v)
    cnt = int(np.sum(v > 1e-9))  # count how many are effectively positive/binding
    return f"mean={mean:.3g}, max={mx:.3g}, #>0={cnt}/{v.size}"

def print_dual_table(title, duals_meta):
    """
    Print a compact "scoreboard" for the duals (shadow prices) with:
      - Greek symbol
      - Programmatic name
      - Human description
      - Value (scalar) or summary stats (vector)
    """
    print(f"---- {title} ----")
    header = f"{'Greek':<6} | {'Name':<24} | {'Description':<32} | {'Value / Stats'}"
    print(header)
    print("-" * len(header))
    # Order chosen to match our discussion; keys absent are skipped
    for key in ["lambda_B","lambda_k","alpha","eta","lambda_i","mu","rho_plus","rho_minus","nu_plus","nu_minus"]:
        if key not in duals_meta:
            continue
        meta = duals_meta[key]
        greek = meta["greek"]
        name = key
        desc = meta["desc"]
        val = meta["value"]
        if np.isscalar(val) or (hasattr(val, "shape") and np.size(val) == 1):
            val_str = _fmt_scalar(val)
        else:
            val_str = _fmt_vector_stats(val)
        print(f"{greek:<6} | {name:<24} | {desc:<32} | {val_str}")
    print()  # blank line after table
