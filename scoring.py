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
from scipy.stats import spearmanr, pearsonr
from tabulate import tabulate
import bittensor as bt

from collections import defaultdict
from typing import Dict, Any, List
from datetime import datetime, timedelta, timezone
from constants import (
  ROLLING_HISTORY_IN_DAYS,
  ROI_MIN,
  VOLUME_MIN,
  VOLUME_FEE,
  VOLUME_DECAY,
  RAMP,
  RHO_CAP,
  KAPPA_NEXT,
  KAPPA_SCALING_FACTOR,
  ENABLE_STATIC_WEIGHTING,
  GENERAL_POOL_WEIGHT_PERCENTAGE,
  MINER_WEIGHT_PERCENTAGE,
  MIN_EPOCHS_FOR_ELIGIBILITY,
  MIN_PREDICTIONS_FOR_ELIGIBILITY,
  BURN_UID,
  EXCESS_MINER_WEIGHT_UID,
  EXCESS_MINER_MIN_WEIGHT,
  MAX_EPOCH_BUDGET_PERCENTAGE_FOR_BOOST,
  MINER_POOL_BUDGET_BOOST_PERCENTAGE,
  MINER_POOL_WEIGHT_BOOST_PERCENTAGE
)

def score_miners(
    all_uids: List[int],
    all_hotkeys: List[str],
    trading_history: List[Dict[str, Any]],
    current_epoch_budget: float,
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

    if current_epoch_budget is None:
        raise ValueError("current_epoch_budget is required")
    if trading_history is None:
        raise ValueError("trading_history is required")
    if all_uids is None:
        raise ValueError("all_uids is required")
    if all_hotkeys is None:
        raise ValueError("all_hotkeys is required")
    
    # Normalize trading_history to always be a list
    # Handle case where API might return dict with "data" field (defensive programming)
    if isinstance(trading_history, dict):
        if "data" in trading_history:
            trading_history = trading_history["data"]
        else:
            raise ValueError(f"trading_history is a dict but missing 'data' field. Keys: {list(trading_history.keys())}")
    if not isinstance(trading_history, list):
        raise ValueError(f"trading_history must be a list or dict with 'data' field, got {type(trading_history)}")

    # Convert the trading history to match the format expected by the scoring function
    """
    {
        "position_id": 69,
        "account_id": 3,
        "profile_id": "0xc70",
        "miner_id": 17,
        "miner_hotkey": "5EqZoEKc6c8TaG4xRRHTT1uZiQF5jkjQCeUV5t77L6YbeaJ8",
        "is_general_pool": false,
        "market_id": "684074",
        "token_id": "17172534480191114731684649039231483628133799672384902614021128663005351577066",
        "date_created": "2025-11-21T23:18:22.076Z",
        "volume": 4.99998838889,
        "expected_fees": 0.0499998838889,
        "actual_fees": 0.05,
        "pnl": 6.11112161111,
        "is_correct": true,
        "is_completed": true,
        "completed_at": "2025-11-22T11:31:15.127Z",
        "is_reward_eligible": true
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

    # Calculate fees collected for the target epoch
    # If target_epoch_idx is None, use the last epoch (current behavior)
    epoch_idx = target_epoch_idx if target_epoch_idx is not None else -1
    miner_fees = np.sum(miner_history["fees_prev"][epoch_idx]) if miner_history["n_entities"] > 0 else 0.0
    gp_fees = np.sum(general_pool_history["fees_prev"][epoch_idx]) if general_pool_history["n_entities"] > 0 else 0.0
    current_epoch_fees_collected = miner_fees + gp_fees
    
    print(f"Current epoch fees collected: {current_epoch_fees_collected:,.2f}")
    print(f"-> Miner pool fees: {miner_fees:,.2f}")
    print(f"-> General pool fees: {gp_fees:,.2f}")

    if ENABLE_STATIC_WEIGHTING:
        # Calculate the budget for each pool based on our constants that reallocate the total fees collected to the miners and general pool.
        miner_pool_epoch_fees = current_epoch_fees_collected * MINER_WEIGHT_PERCENTAGE
        general_pool_epoch_fees = current_epoch_fees_collected * GENERAL_POOL_WEIGHT_PERCENTAGE

        # Calculate the max budget for each pool based on our constants. This is the max budget for the pool for the epoch.
        max_current_epoch_budget = current_epoch_budget * MAX_EPOCH_BUDGET_PERCENTAGE_FOR_BOOST
        miner_pool_epoch_max_budget = max_current_epoch_budget * MINER_WEIGHT_PERCENTAGE

        # If max epoch budgets are greater than our fees, set the fees to the max budget. This gives more weights (and in turn, more incentives) to the miners when we aren't using the full budget.
        miner_pool_max_budget_weighted = miner_pool_epoch_fees
        if miner_pool_epoch_max_budget > miner_pool_epoch_fees:
            miner_pool_max_budget_weighted = miner_pool_epoch_max_budget
            #bt.logging.info(f"Miner pool max budget weighted is greater than fees ({miner_pool_epoch_fees:,.2f}). Setting to {miner_pool_epoch_max_budget:,.2f} ({MAX_EPOCH_BUDGET_PERCENTAGE_FOR_BOOST * 100:.2f}% boost)")
            print(f"Miner pool max budget is greater than fees ({miner_pool_epoch_fees:,.2f}). Setting to {miner_pool_epoch_max_budget:,.2f} ({MAX_EPOCH_BUDGET_PERCENTAGE_FOR_BOOST * 100:.2f}% boost)")

        # Calculate the subnet budget for each pool based on our constants. This is the total budget for the subnet for the epoch.
        miner_pool_epoch_budget = current_epoch_budget * MINER_WEIGHT_PERCENTAGE
        general_pool_epoch_budget = current_epoch_budget * GENERAL_POOL_WEIGHT_PERCENTAGE
        
    else:
        # Use the fees collected for each pool directly, which acts as dynamic weighting.
        miner_pool_epoch_budget = miner_fees
        general_pool_epoch_budget = gp_fees

        # Calculate the max budget for the miner pool based on our constants. This is the max budget for the pool for the epoch.
        miner_pool_max_budget_weighted = miner_pool_epoch_budget
        if MINER_POOL_BUDGET_BOOST_PERCENTAGE > 0:
            miner_pool_max_budget_weighted = miner_pool_epoch_budget * (1 + MINER_POOL_BUDGET_BOOST_PERCENTAGE)
            if miner_pool_max_budget_weighted + general_pool_epoch_budget > current_epoch_budget:
                miner_pool_max_budget_weighted = miner_pool_epoch_budget
                #bt.logging.info(f"Miner pool budget boost would exceed total budget. Using regular budget instead.\n")
                print(f"Miner pool budget boost would exceed total budget. Using regular budget instead.\n")
            else:
                #bt.logging.info(f"Miner pool budget boost would not exceed total budget. Using boosted budget of {miner_pool_max_budget_weighted:,.2f} (+{MINER_POOL_BUDGET_BOOST_PERCENTAGE * 100:.2f}%)\n")
                print(f"Miner pool budget boost would not exceed total budget. Using boosted budget of {miner_pool_max_budget_weighted:,.2f} (+{MINER_POOL_BUDGET_BOOST_PERCENTAGE * 100:.2f}%)\n")
        

    # Calculate the miner pool scores using epoch-based history
    miners_scores = score_with_epochs(
        epoch_history=miner_history,
        budget=miner_pool_epoch_budget,
        max_budget_weighted=miner_pool_max_budget_weighted,
        roi_min=ROI_MIN,
        volume_min=VOLUME_MIN,
        require_epoch_preds=False,
        verbose=verbose
    )
    
    # Calculate the general pool scores using epoch-based history
    general_pool_scores = score_with_epochs(
        epoch_history=general_pool_history,
        budget=general_pool_epoch_budget,
        max_budget_weighted=None, # General pool is not eligible for the max budget boost.
        roi_min=ROI_MIN,
        volume_min=VOLUME_MIN,
        require_epoch_preds=True,
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
    - miner_profiles: dict mapping of miner_id to polymarket_id
    - account_map: dict mapping of entity_id to account_id
    """
    # Determine date range and number of epochs
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = today - timedelta(days=ROLLING_HISTORY_IN_DAYS)
    
    # Determine number of epochs based on target_epoch_idx
    if target_epoch_idx is not None:
        # For historical simulation: create epochs 0 through target_epoch_idx
        n_epochs = target_epoch_idx + 1
    else:
        # For current epoch: use full rolling window (default behavior)
        n_epochs = ROLLING_HISTORY_IN_DAYS
    
    # Create list of epoch dates (0 = oldest, n_epochs-1 = most recent)
    epoch_dates = [(start_date + timedelta(days=i)).date() for i in range(n_epochs)]
    
    # First pass: collect all entity IDs and organize trades by epoch
    entity_set = set()
    epoch_trades = defaultdict(list)  # epoch_idx -> list of trades
    miner_profiles = {}
    account_map = {}

    for trade in trading_history:
        # Skip if the trade does not have an account_id (should not happen)
        if trade["account_id"] is None:
            continue
        
        # Skip if the trade is not completed
        if not trade["is_completed"]:
            continue
            
        # Parse date
        date_completed = trade["completed_at"]
        if isinstance(date_completed, str):
            date_completed = datetime.fromisoformat(date_completed.replace('Z', '+00:00'))
        trade_date = date_completed.date()
        
        # Find epoch index
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
            if miner_id not in all_uids or miner_id >= len(all_hotkeys) or all_hotkeys[miner_id] != miner_hotkey:
                continue
            entity_id = miner_id
            if miner_id not in miner_profiles:
                miner_profiles[miner_id] = trade["profile_id"]
            elif miner_profiles[miner_id] != trade["profile_id"]:
                # append additional profile ids to be validated later in the validator
                miner_profiles[miner_id] += f",{trade['profile_id']}"

        else:
            # General pool
            if not trade["is_general_pool"]:
                continue
            entity_id = trade["profile_id"]

        # Get account ID and map to entity_id
        account_id = trade["account_id"]
        if entity_id not in account_map:
            account_map[entity_id] = account_id
                        
        entity_set.add(entity_id)
        epoch_trades[epoch_idx].append((entity_id, trade))
    
    # Create entity mapping
    entity_ids = sorted(list(entity_set))
    entity_map = {eid: idx for idx, eid in enumerate(entity_ids)}
    n_entities = len(entity_ids)
    # n_epochs was already calculated above based on target_epoch_idx
    
    # Initialize matrices (like simulate_epochs.py)
    volume_prev = np.zeros((n_epochs, n_entities))          # volume
    qualified_prev = np.zeros((n_epochs, n_entities))       # qualified volume
    unqualified_prev = np.zeros((n_epochs, n_entities))     # losing volume
    profit_prev = np.zeros((n_epochs, n_entities))          # profit
    fees_prev = np.zeros((n_epochs, n_entities))            # fees collected
    trade_counts = np.zeros((n_epochs, n_entities))         # number of trades
    correct_trade_counts = np.zeros((n_epochs, n_entities)) # number of correct trades
    
    # Second pass: populate matrices
    for epoch_idx in range(n_epochs):
        if epoch_idx not in epoch_trades:
            continue
            
        for entity_id, trade in epoch_trades[epoch_idx]:
            col_idx = entity_map[entity_id]
            
            volume = trade["volume"]
            pnl = trade["pnl"]
            is_correct = trade["is_correct"]
            is_reward_eligible = trade["is_reward_eligible"]
            
            # Calculate metrics
            #fee = volume * VOLUME_FEE
            fee = trade["actual_fees"] if trade["actual_fees"] is not None else 0.0

            # Always collect fees for all trades, even if the trade is not reward eligible
            fees_prev[epoch_idx, col_idx] += fee
            
            if is_reward_eligible and fee > 0:
                volume_prev[epoch_idx, col_idx] += volume
                if is_correct:
                    # Winning trade: qualified volume (after fee deduction)
                    #qualified = volume * (1.0 - VOLUME_FEE)
                    qualified = volume - fee
                    qualified_prev[epoch_idx, col_idx] += qualified
                    correct_trade_counts[epoch_idx, col_idx] += 1
                else:
                    # Losing trade: unqualified volume
                    unqualified_prev[epoch_idx, col_idx] += volume
                
                profit_prev[epoch_idx, col_idx] += pnl
                trade_counts[epoch_idx, col_idx] += 1  # Count each trade
    
    return {
        "volume_prev": volume_prev,
        "qualified_prev": qualified_prev,
        "unqualified_prev": unqualified_prev,
        "profit_prev": profit_prev,
        "fees_prev": fees_prev,
        "trade_counts": trade_counts,
        "correct_trade_counts": correct_trade_counts,
        "entity_ids": entity_ids,
        "entity_map": entity_map,
        "epoch_dates": [str(d) for d in epoch_dates],
        "n_epochs": n_epochs,
        "n_entities": n_entities,
        "miner_profiles": miner_profiles,
        "account_map": account_map
    }

def check_build_up_eligibility(epoch_history: Dict[str, Any]) -> np.ndarray:
    """
    Check if entities meet the build-up period requirements for eligibility.
    
    Returns a boolean array where True means the entity meets build-up requirements.
    """
    n_entities = epoch_history["n_entities"]
    n_epochs = epoch_history["n_epochs"]
    trade_counts = epoch_history["trade_counts"]  # (n_epochs, n_entities)
    volume_prev = epoch_history["volume_prev"]   # (n_epochs, n_entities)
    
    # Initialize eligibility array
    eligible = np.ones(n_entities, dtype=bool)
    
    for entity_idx in range(n_entities):
        # Check minimum epochs requirement
        entity_epochs_with_trades = np.sum(trade_counts[:, entity_idx] > 0)
        if entity_epochs_with_trades < MIN_EPOCHS_FOR_ELIGIBILITY:
            eligible[entity_idx] = False
            continue
            
        # Check minimum trades requirement
        total_trades = np.sum(trade_counts[:, entity_idx])
        if total_trades < MIN_PREDICTIONS_FOR_ELIGIBILITY:
            eligible[entity_idx] = False
            continue
    
    return eligible

def score_with_epochs(
    epoch_history: Dict[str, Any],
    budget: float,
    roi_min: float,
    volume_min: float,
    max_budget_weighted: float,
    require_epoch_preds: bool = False,
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
    volume_prev_matrix = epoch_history["volume_prev"]           # (n_epochs, n_entities)
    qualified_prev_matrix = epoch_history["qualified_prev"]     # (n_epochs, n_entities)
    unqualified_prev_matrix = epoch_history["unqualified_prev"] # (n_epochs, n_entities)
    profit_prev_matrix = epoch_history["profit_prev"]           # (n_epochs, n_entities)
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
    
    """
    We need to collect this epoch's current data.
    The qualified volume is then the latest array entry and we
    mask the volume by lookin at the trailing ROI.
    
    This concept is designed to gate out traders who don't actually trade
    from freeloading in this epoch but also ensure there is no flicker from
    traders with no long term signal 
    """
    # Calculate trailing aggregates (sum across all epochs, axis=0)
    total_volume = np.sum(volume_prev_matrix, axis=0)
    total_profit = np.sum(profit_prev_matrix, axis=0)
    
    # Calculate trailing ROI for each entity
    roi_trailing = np.divide(
        total_profit,
        np.maximum(total_volume, 1e-12),
        out=np.zeros_like(total_profit),
        where=total_volume > 0
    )
    
    # Check build-up eligibility requirements
    build_up_eligible = check_build_up_eligibility(epoch_history)
    
    # Debug: Print build-up eligibility stats
    if verbose:
        n_build_up_eligible = np.sum(build_up_eligible)
        print(f"Build-up eligibility: {n_build_up_eligible}/{n_entities} entities meet build-up requirements")
        print(f"  - MIN_EPOCHS_FOR_ELIGIBILITY: {MIN_EPOCHS_FOR_ELIGIBILITY}")
        print(f"  - MIN_TRADES_FOR_ELIGIBILITY: {MIN_PREDICTIONS_FOR_ELIGIBILITY}")
    
    # Combine all eligibility requirements
    qual_mask = (
        (roi_trailing >= roi_min) & 
        (qualified_prev_matrix >= volume_min) &
        build_up_eligible
    )
    qual_volume = np.where(qual_mask, qualified_prev_matrix, 0.0)
    
    # Get latest epoch data (last row = most recent epoch, index -1)
    current_epoch_idx = epoch_history["n_epochs"] - 1
    current_epoch_volume = volume_prev_matrix[current_epoch_idx]
    current_epoch_profit = profit_prev_matrix[current_epoch_idx]
    current_epoch_roi = np.divide(
        current_epoch_profit,
        np.maximum(current_epoch_volume, 1e-12),
        out=np.zeros_like(current_epoch_profit),
        where=current_epoch_volume > 0,
    )
    #current_epoch_qual_mask = (current_epoch_roi >= roi_min) & (current_epoch_volume >= volume_min)
    #current_epoch_qual_volume = np.where(current_epoch_qual_mask, current_epoch_volume, 0.0)
    
    # --- Calculate v_memory from historical data ---
    # v_memory is a decaying memory of historical volume
    # We need to simulate the decay process from the historical data
    v_memory = np.zeros(n_entities)
    # Simulate the decay process through all historical epochs
    for epoch_idx in range(epoch_history["n_epochs"]):
        epoch_volume = volume_prev_matrix[epoch_idx]
        v_memory = VOLUME_DECAY * v_memory + epoch_volume
    
    # --- Effective volume: balance fresh vs memory ---
    alpha = 1.0 / (2.0 - VOLUME_DECAY)                 # e.g., decay=0.98 -> alpha≈0.980392
    v_eff = alpha * current_epoch_volume + (1.0 - alpha) * v_memory

    """
    Kappa computation is a custom function as its a bit complex.
    It is broken down in the compute joint kappa function.
    """
    kappa_bar = compute_joint_kappa_from_history(epoch_history)

    # @TODO: verify with Stephen (currently not in use)
    x_prev = np.zeros(n_entities)
    last_duals = None
    last_alloc = np.zeros(n_entities)
    last_tokens = np.zeros(n_entities)
    last_scores = np.zeros(n_entities)
    
    """
    A dictionary that contains the critial data for the solvers
    """
    p_dict = {
        "budget": budget,                           # alotted budget
        "max_budget_weighted": max_budget_weighted, # max budget weighted
        "total_volume": np.sum(total_volume),       # all trailing volume
        "total_q_volume": np.sum(qual_volume),      # all trailing volume
        "v_prev": qualified_prev_matrix,            # qualified volume array
        "v_block": current_epoch_volume,            # total qualified vol this epoch
        "v_memory": v_memory,                       # decaying volume
        "v_eff": v_eff,                             # History respecting volume
        "v_qual": qual_volume,                      # qualified volume
        "block_v_qual": qual_volume,                # block qualified volume
        "roi_trailing": roi_trailing,               # trailing roi
        "roi_block": current_epoch_roi,             # block roi array
        "roi_qual": qual_volume,                    # qualified volume
        "block_roi_qual": qual_volume,              # block qualified volume
        "epoch_history": epoch_history,             # epoch history for build-up checks
        # --- CONSTRAINT SETTINGS ---
        "roi_min": roi_min,                         # minimum roi constraint
        "v_min": volume_min,                        # minimum volume constraint
        "x_prev": x_prev,                           # allocations this epoch. @TODO: verify with Stephen (currently not in use)
        "kappa_bar": kappa_bar,                     # payout rate
        "ramp": RAMP,                               # allocation delta rate
        "rho_cap": RHO_CAP,                         # max allocation per miner
        "require_epoch_preds": require_epoch_preds  # require current epoch predictions toggle
    }
    
    """
    Solver Phase 1 

    Objective:  maximize routed qualified volume 
    
    Method: Choose gates x* that allows a bettor to route volume through
            and get paid by the budget for their signal

    Return: optimal 0 <= x* <= 1 for all i miners and total qualified volume T* 
            which is the maximum we can afford to fund with out budget B.  Also returns 
            total payout, number of funded bettors, number of possible eligible bettors
            as well as a status print out (see validator_min.py)
    """
    sol1 = solve_phase1(p_dict, verbose=verbose)
    if sol1["x_star"] is not None:
        x_prev = np.clip(sol1["x_star"], 0.0, 1.0)
    
    """
    Solver Phase 2 

    Objective:  Redistribute payout from weak skill to strong skill and adjust gates x* to x** 
    
    Method: Choose gates x** by passing in x* to the 2nd problem and minimize the cost
            of funding the volume.

    Return: optimal 0 <= x** <= 1 for all i miners which minimizes total cost of budget 
            paid out.  Returns new budget and cost vector c* as well.
    """
    # --- Phase 2: maximize signal given T* and x* ---
    sol2 = None
    if sol1["status"] in ("optimal", "optimal_inaccurate") and sol1["T_star"] > 0:
        sol2 = solve_phase2(p_dict, x1=sol1["x_star"], T1=sol1["T_star"], verbose=verbose)
        if sol2["x_star"] is not None:
            x_prev = np.clip(sol2["x_star"], 0.0, 1.0)

    # --- store final state for next epoch ---
    last_duals = sol2.get("duals", None) if sol2 else None
    last_alloc = x_prev.copy()  

    """
    Miner scoring and allocation
    
    The notion here is use the final optimization values to sum up the scores
    of those miners that actually added signal while scoring the 
    """
    if sol2 is not None:
        x_star = np.array(sol2.get("x_star", np.zeros_like(x_prev)))
        c_star = np.array(sol2.get("c_star", np.zeros_like(x_prev)))
    else:
        x_star = np.zeros_like(x_prev)
        c_star = np.zeros_like(x_prev)

    # emission weight = funded volume × signal strength (ROI)
    raw_scores = x_star * c_star
    # normalize so total payout == budget
    total_score = np.sum(raw_scores)
    if total_score > 0:
        normalized_scores = raw_scores / total_score
    else:
        normalized_scores = np.zeros_like(raw_scores)

    # convert to token allocations using the actual payout from Phase 2 or Phase 1
    if sol2 and sol2.get("payout") is not None:
        epoch_payout = sol2["payout"]
    elif sol1 and sol1.get("payout") is not None:
        epoch_payout = sol1["payout"]
    else:
        epoch_payout = 0.0
    
    token_allocations = normalized_scores * epoch_payout

    if verbose:
        # Set numpy print options for cleaner output
        np.set_printoptions(precision=8, suppress=True, floatmode='fixed')
        print("normalized_scores", normalized_scores)
        print("token_allocations", token_allocations)
        # Reset to default
        np.set_printoptions()

    """
    Diagnostics:  For every epoch it is important to print out a lot of data 
                    so we can observe the optimizer and ensure it is working 
                    well.
    """
    # Spearman
    roi = np.maximum(roi_trailing, 0.0)

    # Block-level accounting / diagnostics
    block_unqualified = unqualified_prev_matrix[current_epoch_idx].sum()
    block_qualified = qualified_prev_matrix[current_epoch_idx].sum()
    block_fees = VOLUME_FEE * (block_unqualified + block_qualified)

    # Historical (all epochs so far)
    total_unqualified = np.sum(unqualified_prev_matrix)
    total_qualified = np.sum(qualified_prev_matrix)
    
    if verbose:
        print(
            f"###########################################\n"
            f"## [Validator]: epoch {current_epoch_idx}:\n"
            f"## Stage 1: Optimization\n"
            f"###########################################\n"
            f"total_unqualified = {total_unqualified:.2f}, \n"
            f"total_qualified   = {total_qualified:.2f},\n"
            f"block_unqualified = {block_unqualified:.2f}, \n"
            f"block_qualified   = {block_qualified:.2f},\n"
            f"total       = {block_unqualified + block_qualified:.2f},\n"
            f"fees        = {block_fees:.2f},\n"
            f"budget      = {budget:.2f},\n"
            f"status      = {sol1['status']},\n"
            f"T*          = {sol1['T_star']:.2f},\n"
            f"net         = {block_fees - sol1['payout']:.2f}\n"
            f"payout      = {sol1['payout']:.2f}\n"
            f"numfunded   = {sol1['num_funded']:.2f}\n"
            f"eligible    = {sol1['num_eligible']:.2f}\n"
            f"###########################################\n"
            f"\n"
            f"###########################################\n"
            f"[Validator]: epoch {current_epoch_idx}:\n"
            f"Stage 2: Optimization\n"
            f"###########################################\n"
            f"status      = {sol2['status'] if sol2 else 'n/a'},\n"
            f"Payout*     = {sol2['payout'] if sol2 else 0.0:.2f},\n"
            f"Phase 1  ρ(ROI, x*) = {spearmanr(roi, sol1['x_star'])[0]},\n"
            f"Phase 2  ρ(ROI, x**) = {spearmanr(roi, sol2['x_star'])[0] if sol2 else 'n/a'},\n"
        )
    
    return {
        "entity_ids": entity_ids,
        "scores": normalized_scores,
        "tokens": token_allocations,
        "sol1": sol1,
        "sol2": sol2,
        "total_volume": total_volume,
        "total_profit": total_profit,
        "roi_trailing": roi_trailing,
        "kappa_bar": kappa_bar
    }

def solve_phase1(p, verbose=False):
    """
    Phase 1: Maximize routed qualified volume given budget & payout constraints. 
    """
    #########################################
    ## Historical data and size
    #########################################
    v_prev, v_eff ,v_block, roi_trailing, x_prev = p["v_prev"], p["v_eff"], p["v_block"], p["roi_trailing"], p["x_prev"]
    n = v_block.size
    #########################################
    ## Numerical scaling
    ## First we have to make a scale so the 
    ## optimizer can properly find an interior
    ## point
    #########################################
    B            = p["budget"]                   # budget size
    v_memory     = p.get("v_memory", v_prev)     # decaying volume
    v_block      = np.maximum(v_block, 1e-12)    # block volume
    c            = v_eff*roi_trailing            # costs
    ## settings
    kappa        = p["kappa_bar"]                # dimensionless
    rho_cap      = p.get("rho_cap", RHO_CAP)     # diversity
    ramp         = p.get("ramp", RAMP)           # ramp
   
    #########################################
    ## Eligibility Gates
    ## We then have to generate a cost per units
    ## of signal and then using trailing volume
    ## make a boolean which flags eligibility
    #########################################
    # Get build-up eligibility from the epoch history
    epoch_history = p.get("epoch_history")
    if epoch_history is not None:
        build_up_eligible = check_build_up_eligibility(epoch_history)
    else:
        # Fallback: assume all are eligible if no epoch history provided
        build_up_eligible = np.ones(len(roi_trailing), dtype=bool)
    
    # require_epoch_preds: If True, require current epoch predictions (v_block) for eligibility
    require_epoch_preds = p.get("require_epoch_preds", False)
    if require_epoch_preds:
        # Require current epoch predictions - use v_block (total volume this epoch)
        eligible = (
            (roi_trailing >= p["roi_min"]) & 
            (v_block >= p["v_min"]) &
            build_up_eligible
        ).astype(float)
    else:
        # Use decaying volume memory - only use v_memory (historical volume with decay)
        eligible = (
            (roi_trailing >= p["roi_min"]) & 
            (v_memory >= p["v_min"]) &
            build_up_eligible
        ).astype(float)

    if verbose:
        print({"kappa": kappa})
        print({"roi_trailing": roi_trailing*eligible})
 
    # 3) Decision variable
    x = cp.Variable(n)

    # 4) Constraints
    eps = 1e-9
    cons = []
    cons += [x >= 0, x <= 1]                                         # gates
    cons += [kappa * (v_eff @ x) <= B]                               # budget
    cons += [x <= eligible]                                          # eligibility
    cons += [kappa * v_eff[i] * x[i] <= rho_cap*B for i in range(n)] # diversity
    
    #TODO: these might no longer be needed at all.
    #cons += [c @ x <= kappa * (v_eff @ x + eps)]            # payout/volume cap
    #cons += [x - x_prev <= ramp, x_prev - x <= ramp]        # ramp
    #cons += [x >= 1e-2*eligible]                            # dust constraint
    #cons += [c[i] * x[i] <= rho_cap * B for i in range(n)]  # diversity cap    
   
    # 5) Objective
    prob = cp.Problem(cp.Maximize(v_block@x), cons)
    prob.solve(solver=cp.ECOS, verbose=False)

    # 6) === Dual diagnostics table ===
    if prob.status in ("optimal", "optimal_inaccurate"):
        if verbose:
            print("\n====================[ Phase 1 Dual Summary ]====================")

            labels = [
                ("x >= 0",            cons[0]),
                ("x <= 1",            cons[1]),
                ("budget cap",        cons[2]),
                ("x <= eligible",     cons[3]),
            ]

            # Diversity group (vector of n)
            diversity_cons = cons[4:]

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

        # 6) Return
        x_star = None if x.value is None else x.value.copy()
        T_star = float(v_block @ x_star) if x_star is not None else 0.0
        
        payout = float(np.dot(v_block * np.maximum(roi_trailing, 0.0), x_star)) if x_star is not None else 0.0
        num_funded = int(np.sum((v_block * np.maximum(roi_trailing, 0.0) * x_star) > 0)) if x_star is not None else 0
        num_eligible = int(np.sum(eligible))

        return {
            "status": prob.status,
            "T_star": T_star,   # in the same units as your printed "qualified"
            "x_star": x_star,
            "payout": payout,
            "num_funded": num_funded,
            "num_eligible": num_eligible,
        }

def solve_phase2(p, x1, T1, verbose=False):
    """
    Phase 2: Redistribute fixed payout to favor higher ROI while staying
    close to x1 (Phase 1 gates).
    """
    v            = p["v_block"]
    n            = v.size
    v_eff        = p["v_eff"]
    roi          = p["roi_trailing"]
    B            = p["budget"]
    max_budget = p.get("max_budget_weighted", B) if p.get("max_budget_weighted") is not None else B # default max_budget to budget if not provided
    kappa        = p["kappa_bar"]

    # Token cost per miner if fully opened
    c = kappa * v_eff

    ##eligibility
    # Get build-up eligibility from the epoch history
    epoch_history = p.get("epoch_history")
    if epoch_history is not None:
        build_up_eligible = check_build_up_eligibility(epoch_history)
    else:
        # Fallback: assume all are eligible if no epoch history provided
        build_up_eligible = np.ones(len(roi), dtype=bool)

    # require_epoch_preds: If True, require current epoch predictions (v_block) for eligibility
    require_epoch_preds = p.get("require_epoch_preds", False)
    if require_epoch_preds:
        # Require current epoch predictions - use v_block (total volume this epoch)
        eligible = (
            (roi >= p["roi_min"]) & 
            (v >= p["v_min"]) &
            build_up_eligible
        ).astype(float)
    else:
        # Use decaying volume memory - only use v_eff (historical volume with decay)
        eligible = (
            (roi >= p["roi_min"]) & 
            (v_eff >= p["v_min"]) &
            build_up_eligible
        ).astype(float)

    # Total payout from Phase 1 (fixed)
    P1 = float(np.dot(c, x1))

    # signed ROI deviation from kappa
    delta = roi - kappa

    # weight positive deltas (above kappa) more, negative ones less
    w = delta

    # Decision variable
    x = cp.Variable(n)

    # Constraints
    cons = []
    cons += [x >= 0, x <= 1]
    cons += [x <= eligible]                 # eligibility
    cons += [c @ x <= min(P1, B, max_budget)] # fixed total payout (ensured not greater than budget or max budget weighted)

    # Objective: favor ROI while staying close to prior gates
    # dynamic smoothness penalty λ
    roi_spread = np.std(p["roi_trailing"])
    vol_mean   = np.mean(p["v_eff"])

    # dynamic smoothness penalty λ. higher value means more smoothness, less volatility.
    lam = 5

    if verbose:
        print({"watx": w @ x, "scale": lam * cp.sum_squares(x - x1)})

    obj = cp.Maximize(w @ x -  lam * cp.sum_squares(x - x1))

    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.ECOS, verbose=False)

    if x.value is not None:
        dx = x.value - x1
        s1 = spearmanr(roi, x1)[0]
        s2 = spearmanr(roi, x.value)[0]
        if verbose:
            print("Δx mean:", float(np.mean(np.abs(dx))))
            print("Spearman Δ:", float(s2 - s1))
            print("ρ(ROI, x**):", float(s2))

    return {
        "status": prob.status,
        "x_star": None if x.value is None else x.value.copy(),
        "c_star": c,
        "payout": min(P1, B, max_budget), # ensure payout is not greater than budget or max budget weighted
    }

"""
Function: compute_joint_kappa

Purpose: Computes the value of kappa as an endogenous variable that depends
            on previous volume and roi.  The goal is to restrict the payout rate
            of the budget by setting the “exchange rate” between qualified flow and token budget.
"""
def compute_joint_kappa_from_history(epoch_history: Dict[str, Any], lookback=ROLLING_HISTORY_IN_DAYS, smooth=0.3, fee_rate=VOLUME_FEE, kappa_next=KAPPA_NEXT, joint_kappa=None) -> float:
    """
    Compute kappa_bar from epoch history matrices.
    
    κ_next from *aggregates* over the last `lookback` epochs:
        κ = (sum fees over active epochs) / (sum positive profit over active epochs)
    Active epoch = total volume>0 and total profit>0.
    No per-epoch division => no 1/0 explosions. Uses prior κ if not enough signal.
    """
    v_prev = epoch_history["qualified_prev"]
    profit_prev = epoch_history["profit_prev"]
    
    prev_kappa = kappa_next
    if v_prev.shape[0] < 5:
        return prev_kappa

    V_hist = v_prev[-lookback:]
    P_hist = profit_prev[-lookback:]

    vols = np.sum(V_hist, axis=1)               # per-epoch total volume
    prof = np.sum(P_hist, axis=1)               # per-epoch total profit

    # --- active epochs: positive profit and nonzero volume ---
    active = (vols > 0) & (prof > 0)
    if np.count_nonzero(active) < max(3, lookback // 10):
        return prev_kappa

    fees_sum     = fee_rate * float(np.sum(vols[active]))
    posprofit_sum= float(np.sum(prof[active]))   # strictly >0 by mask

    # aggregate ratio; guarded but should be safe due to mask
    kappa_stat = fees_sum / max(posprofit_sum, 1e-12)

    # light smoothing, no target constants
    prior = joint_kappa if joint_kappa is not None else kappa_stat
    joint_kappa = (1.0 - smooth) * prior + smooth * kappa_stat
    
    return max(joint_kappa, 1e-12)/KAPPA_SCALING_FACTOR

def print_pool_stats(miner_history, general_pool_history, include_current_epoch=False, miner_scores=None, general_pool_scores=None):
    """Print historical stats for both pools."""

    headers = ["Rank", "PID", "# Epochs", "Preds", "Total Vol", "Qual Vol", "PNL", "ROI"]
    if include_current_epoch:
        headers.append("Ep. Preds")
        headers.append("Ep. Vol")
        headers.append("Ep. Qual Vol")
        headers.append("Ep. PNL")
        headers.append("Ep. ROI")
        headers.append("Ep. Earnings")
        headers.append("Ep. Earnings/PnL")
        headers.append("Ep. Earnings/Vol")
    
    # Analyze miner pool
    if miner_history['n_entities'] > 0:
        print("--- MINER POOL STATS ---")
        miner_stats = create_pool_stats_table(miner_history, "Miner", include_current_epoch, miner_scores)
        print(tabulate(miner_stats, headers=headers, tablefmt="grid", stralign="right"))
    else:
        print("--- MINER POOL STATS ---")
        print("No miners found in data")
    
    # Analyze general pool
    if general_pool_history['n_entities'] > 0:
        print("\n--- GENERAL POOL STATS ---")
        gp_stats = create_pool_stats_table(general_pool_history, "General", include_current_epoch, general_pool_scores)
        print(tabulate(gp_stats, headers=headers, tablefmt="grid", stralign="right"))
    else:
        print("\n--- GENERAL POOL STATS ---")
        print("No general pool users found in data")

def create_pool_stats_table(epoch_history, pool_type, include_current_epoch=False, scores=None):
    """Create a table of historical stats for a pool."""
    v_prev_matrix = epoch_history["qualified_prev"]           # (n_epochs, n_entities) - qualified volume
    unqualified_matrix = epoch_history["unqualified_prev"]  # (n_epochs, n_entities) - unqualified volume
    profit_matrix = epoch_history["profit_prev"]      # (n_epochs, n_entities) - profit
    trade_counts_matrix = epoch_history["trade_counts"]  # (n_epochs, n_entities) - number of trades
    entity_ids = epoch_history["entity_ids"]
    n_entities = epoch_history["n_entities"]
    n_epochs = epoch_history["n_epochs"]
    
    if n_entities == 0:
        return []
    
    # Get current epoch data (last epoch)
    current_epoch_idx = n_epochs - 1
    current_epoch_qualified = v_prev_matrix[current_epoch_idx] if n_epochs > 0 else np.zeros(n_entities)
    current_epoch_unqualified = unqualified_matrix[current_epoch_idx] if n_epochs > 0 else np.zeros(n_entities)
    current_epoch_volume = current_epoch_qualified + current_epoch_unqualified
    current_epoch_profit = profit_matrix[current_epoch_idx] if n_epochs > 0 else np.zeros(n_entities)
    current_epoch_trades = trade_counts_matrix[current_epoch_idx] if n_epochs > 0 else np.zeros(n_entities)
    
    table_data = []
    
    for entity_idx in range(n_entities):
        entity_id = entity_ids[entity_idx]
        
        # Calculate stats for this entity across all epochs
        qualified_volume = np.sum(v_prev_matrix[:, entity_idx])  # Sum across all epochs
        unqualified_volume = np.sum(unqualified_matrix[:, entity_idx])  # Sum across all epochs
        total_volume = qualified_volume + unqualified_volume
        total_pnl = np.sum(profit_matrix[:, entity_idx])  # Sum across all epochs
        
        # Count number of epochs this entity traded in (has non-zero volume)
        trading_epochs = np.sum((v_prev_matrix[:, entity_idx] > 0) | (unqualified_matrix[:, entity_idx] > 0))
        
        # Calculate total number of predictions (trades) for this entity
        total_predictions = int(np.sum(trade_counts_matrix[:, entity_idx]))
        
        # Calculate ROI (only if there's qualified volume)
        if qualified_volume > 0:
            roi = total_pnl / qualified_volume
        else:
            roi = 0.0
        
        # Calculate current epoch stats
        epoch_volume = current_epoch_volume[entity_idx]
        epoch_qualified_volume = current_epoch_qualified[entity_idx]
        epoch_pnl = current_epoch_profit[entity_idx]
        epoch_trades = int(current_epoch_trades[entity_idx])
        
        # Calculate current epoch ROI
        if epoch_qualified_volume > 0:
            epoch_roi = epoch_pnl / epoch_qualified_volume
        else:
            epoch_roi = 0.0
        
        # Get earnings (tokens) for this entity if scores are provided
        earnings = 0
        if scores is not None and 'tokens' in scores:
            # Find the entity in the scores
            for i, score_entity_id in enumerate(scores.get('entity_ids', [])):
                if score_entity_id == entity_id:
                    earnings = scores['tokens'][i] if scores['tokens'][i] > 0 else 0
                    break

        # Calculate the % of earnings of their PnL
        earnings_percentage = 0.0
        if earnings > 0:
            earnings_percentage = earnings / total_pnl

        # Calculate the % of earnings to their current epoch volume
        earnings_percentage_to_volume = 0.0
        if epoch_volume > 0:
            earnings_percentage_to_volume = earnings / epoch_volume
        
        # Format entity ID based on pool type
        if pool_type == "Miner":
            display_id = f"{entity_id}"
        else:
            display_id = entity_id
        
        # Build base row data
        row_data = [
            display_id,
            int(trading_epochs),
            total_predictions,
            f"${total_volume:,.0f}",
            f"${qualified_volume:,.0f}",
            f"${total_pnl:,.2f}",
            f"{roi:.4f}",
        ]
        
        # Add current epoch columns if requested
        if include_current_epoch:
            row_data.extend([
                epoch_trades,
                f"${epoch_volume:,.0f}",
                f"${epoch_qualified_volume:,.0f}",
                f"${epoch_pnl:,.2f}",
                f"{epoch_roi:.4f}",
                f"{earnings:.2f}",
                f"{earnings_percentage:.4f}",
                f"{earnings_percentage_to_volume:.4f}",
            ])
        
        table_data.append(row_data)
    
    # Sort by earnings (tokens) if scores are provided, otherwise by PnL
    if scores is not None and 'tokens' in scores:
        # Sort by earnings (last column when include_current_epoch=True)
        if include_current_epoch:
            table_data.sort(key=lambda x: float(x[-3]), reverse=True)  # Sort by Earnings
        else:
            # If no current epoch data but we have scores, we need to get earnings differently
            # For now, fall back to PnL sorting
            table_data.sort(key=lambda x: float(x[5].replace('$', '').replace(',', '')), reverse=True)  # Sort by PnL
    else:
        # Default sorting by PnL (descending) - highest PnL first
        # Alternative sorting options (uncomment one):
        # table_data.sort(key=lambda x: float(x[6]), reverse=True)  # Sort by ROI
        # table_data.sort(key=lambda x: float(x[3].replace('$', '').replace(',', '')), reverse=True)  # Sort by Total Volume
        # table_data.sort(key=lambda x: float(x[4].replace('$', '').replace(',', '')), reverse=True)  # Sort by Qualified Volume
        table_data.sort(key=lambda x: float(x[5].replace('$', '').replace(',', '')), reverse=True)  # Sort by PnL
    
    # Add rank numbers after sorting
    ranked_data = []
    for rank, row in enumerate(table_data, 1):
        ranked_data.append([rank] + row)
    
    return ranked_data

def calculate_weights(miners_scores: Dict[str, Any], general_pool_scores: Dict[str, Any], current_epoch_budget: float, miners_to_penalize: List[int], all_uids: List[int]) -> List[float]:
    """
    Calculate the weights for the miners and general pool based on the scores and the current epoch budget.
    """
    ### Calculate proper weight allocation for the epoch ###
    # Step 1: Calculate individual miner weights as percentage of total budget
    miner_tokens_allocated = miners_scores['tokens'] if 'tokens' in miners_scores else np.zeros(len(miners_scores['entity_ids']))
    miner_entity_ids = miners_scores['entity_ids'] if 'entity_ids' in miners_scores else []
    general_pool_tokens_allocated = general_pool_scores['tokens'] if 'tokens' in general_pool_scores else np.zeros(len(general_pool_scores['entity_ids']))
    general_pool_entity_ids = general_pool_scores['entity_ids'] if 'entity_ids' in general_pool_scores else []
    
    # Calculate total subnet budget for normalization
    total_epoch_budget = current_epoch_budget
    #bt.logging.info(f"Total epoch budget: {total_epoch_budget:,.2f}")
    print(f"Total epoch budget: {total_epoch_budget:,.2f}")
    
    # Initialize weight dictionary
    miner_weights = {}
    
    # Process miner pool weights
    sorted_indices = np.argsort(miner_tokens_allocated)[::-1]
    for idx in sorted_indices:
        miner_uid = miner_entity_ids[idx]
        miner_tokens = miner_tokens_allocated[idx]
        
        # If the miner is in the list to penalize, set the weight to 0
        if miner_uid in miners_to_penalize:
            #bt.logging.info(f"Miner {miner_uid} failed validation. Tokens lost: {miner_tokens:,.2f}. Setting weight to 0.")
            print(f"Miner {miner_uid} failed validation. Tokens lost: {miner_tokens:,.2f}. Setting weight to 0.")
            miner_weights[miner_uid] = 0
            continue
        
        # Calculate weight as percentage of total epoch budget
        miner_weight = miner_tokens / total_epoch_budget
        #bt.logging.info(f"Miner {miner_uid} tokens allocated: {miner_tokens:,.2f}, weight: {miner_weight:.4f}")
        #print(f"Miner {miner_uid} tokens allocated: {miner_tokens:,.2f}, weight: {miner_weight:.4f}")
        miner_weights[miner_uid] = miner_weight
    
    total_miner_pool_tokens = np.sum(miner_tokens_allocated)
    #bt.logging.info(f"Miner pool total epoch units: {total_miner_pool_tokens:,.2f}")
    print(f"Miner pool total epoch units: {total_miner_pool_tokens:,.2f}")

    # Step 2: Calculate general pool total weight and assign to BURN_UID.
    total_general_pool_tokens = np.sum(general_pool_tokens_allocated)
    #bt.logging.info(f"General pool total epoch units: {total_general_pool_tokens:,.2f}")
    print(f"General pool total epoch units: {total_general_pool_tokens:,.2f}")
    if ENABLE_STATIC_WEIGHTING:
        # If static weighting is enabled, the general pool weight is always GENERAL_POOL_WEIGHT_PERCENTAGE of the total epoch budget.
        general_pool_weight = GENERAL_POOL_WEIGHT_PERCENTAGE * total_epoch_budget
        #bt.logging.info(f"General pool BURN_UID weight: {general_pool_weight:.4f} (static weighting: always {GENERAL_POOL_WEIGHT_PERCENTAGE * 100:.2f}% of total epoch budget)")
        print(f"General pool BURN_UID weight: {general_pool_weight:.4f} (static weighting: always {GENERAL_POOL_WEIGHT_PERCENTAGE * 100:.2f}% of total epoch budget)")
    else:
        # If dynamic weighting is enabled, the general pool weight is the percentage of the total epoch budget that the general pool tokens represent.
        general_pool_weight = (total_general_pool_tokens / total_epoch_budget)
        #bt.logging.info(f"General pool BURN_UID weight: {general_pool_weight:.4f} (dynamic weighting: {total_general_pool_tokens:,.2f} / {total_epoch_budget:,.2f})")
        print(f"General pool BURN_UID weight: {general_pool_weight:.4f} (dynamic weighting: {total_general_pool_tokens:,.2f} / {total_epoch_budget:,.2f})")
    miner_weights[BURN_UID] = general_pool_weight

    # Step 3: Calculate total allocated weight and excess
    total_allocated_weight = sum(miner_weights.values())
    excess_weight = 1.0 - total_allocated_weight
    #bt.logging.info(f"Subtotal allocated weight: {total_allocated_weight:.4f}")
    print(f"Subtotal allocated weight: {total_allocated_weight:.4f} (({total_miner_pool_tokens:,.2f} + {total_general_pool_tokens:,.2f}) / {total_epoch_budget:,.2f})")

    # If we have excess weight and the miner pool weight boost percentage is greater than 0, we will boost the miner pool weights by the percentage.
    if excess_weight > 0 and MINER_POOL_WEIGHT_BOOST_PERCENTAGE > 0:
        # Boost the miner pool weights by the percentage
        miner_weights = {uid: weight * (1 + MINER_POOL_WEIGHT_BOOST_PERCENTAGE) for uid, weight in miner_weights.items()}
        #bt.logging.info(f"Boosted miner pool weights by {MINER_POOL_WEIGHT_BOOST_PERCENTAGE * 100:.2f}%")
        print(f"Boosted miner pool weights by {MINER_POOL_WEIGHT_BOOST_PERCENTAGE * 100:.2f}%")
        # Recalculate the total allocated weight and excess weight
        total_allocated_weight = sum(miner_weights.values())
        excess_weight = 1.0 - total_allocated_weight
    
    # Assign excess weight to EXCESS_MINER_WEIGHT_UID, but ensure it is at least EXCESS_MINER_MIN_WEIGHT
    miner_weights[EXCESS_MINER_WEIGHT_UID] = max(excess_weight, EXCESS_MINER_MIN_WEIGHT)
    
    #bt.logging.info(f"Total allocated weight: {total_allocated_weight:.4f}")
    print(f"Total allocated weight: {total_allocated_weight:.4f}")
    #bt.logging.info(f"Excess weight assigned to EXCESS_MINER_WEIGHT_UID: {miner_weights[EXCESS_MINER_WEIGHT_UID]:.4f}")
    print(f"Excess weight assigned to EXCESS_MINER_WEIGHT_UID: {miner_weights[EXCESS_MINER_WEIGHT_UID]:.4f}")

    # Step 4: Convert weight dictionary to array format for Bittensor
    # Create weights array matching the metagraph UIDs
    weights = []
    for uid in all_uids:
        if uid in miner_weights:
            # Use the full weight (not divided by epoch count)
            # The epoch distribution happens automatically over 20 epochs
            if miner_weights[uid] > 0:
                weights.append(miner_weights[uid])
            else:
                weights.append(0.0)
        else:
            # UID not in our weight calculation, assign 0
            weights.append(0.0)
    
    # Normalize weights to ensure they sum to 1.0
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    
    bt.logging.info(f"Setting weights: {weights}")
    bt.logging.info(f"Total weight sum: {sum(weights):.6f}")

    return weights
