"""
Simulation script to test scoring.py with mock trading data.

This script:
1. Loads mock_trading_data.json
2. Extracts unique miner_ids and hotkeys
3. Calls score_miners() to compute scores
4. Prints results and diagnostics
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path so we can import scoring
sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring import score_miners
from constants import MINER_WEIGHT_PERCENTAGE, GENERAL_POOL_WEIGHT_PERCENTAGE
import numpy as np
from tabulate import tabulate


def load_mock_data(filepath="tests/mock_trading_data.json"):
    """Load the mock trading data from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def extract_miner_info(trading_history):
    """
    Extract unique miner UIDs and hotkeys from trading history.
    
    Returns:
        all_uids: List[int] - list of unique miner IDs
        all_hotkeys: Dict[int, str] - mapping of miner_id -> hotkey
    """
    miner_map = {}  # miner_id -> hotkey
    
    for trade in trading_history:
        # Skip general pool trades
        if trade.get("is_general_pool", False):
            continue
            
        miner_id = trade.get("miner_id")
        miner_hotkey = trade.get("miner_hotkey")
        
        # Skip if missing miner info
        if miner_id is None or miner_hotkey is None:
            continue
            
        # Store the mapping
        if miner_id not in miner_map:
            miner_map[miner_id] = miner_hotkey
        else:
            # Verify consistency (same miner_id should always have same hotkey)
            if miner_map[miner_id] != miner_hotkey:
                print(f"WARNING: Inconsistent hotkey for miner_id {miner_id}")
    
    all_uids = sorted(list(miner_map.keys()))
    all_hotkeys = miner_map
    
    return all_uids, all_hotkeys


def calculate_historical_payouts(miner_history, general_pool_history, all_uids, all_hotkeys, trading_history):
    """Calculate payouts for each historical epoch and return as arrays."""
    n_epochs = miner_history['n_epochs']
    epoch_dates = miner_history['epoch_dates']
    
    # Initialize payout arrays
    mp_payouts = np.zeros(n_epochs)
    gp_payouts = np.zeros(n_epochs)
    
    print("Calculating historical payouts for each epoch...")
    
    # Store previous allocations for sequential simulation
    prev_mp_allocations = None
    prev_gp_allocations = None
    
    for epoch_idx in range(n_epochs):
        try:
            # For historical simulation, we need to simulate what the scoring would have been
            # as of that epoch date, using all historical data up to that point
            epoch_date = epoch_dates[epoch_idx]
            
            # Filter to include all trades settled up to and including this epoch date
            # This simulates what data would have been available for scoring on that day
            epoch_trades = []
            for trade in trading_history:
                if trade.get("is_settled", False):
                    trade_date_str = trade.get("date_settled")
                    if trade_date_str:
                        # Convert both dates to strings for comparison since epoch_date is a string
                        if isinstance(trade_date_str, str):
                            trade_date_str_clean = trade_date_str
                        else:
                            trade_date_str_clean = str(trade_date_str)
                        
                        if trade_date_str_clean <= epoch_date:
                            epoch_trades.append(trade)
            
            print(f"Epoch {epoch_idx} ({epoch_date}): {len(epoch_trades)} trades")
            
            if epoch_trades:
                # Run scoring for this specific epoch (silently)
                # This simulates what the scoring would have been on that day
                epoch_miner_history, epoch_gp_history, epoch_miners_scores, epoch_gp_scores, epoch_mp_budget, epoch_gp_budget = score_miners(
                    all_uids=all_uids,
                    all_hotkeys=all_hotkeys,
                    trading_history=epoch_trades,
                    verbose=False,
                    target_epoch_idx=epoch_idx
                )
                
                # Debug budget info
                print(f"  -> MP budget: ${epoch_mp_budget:.2f}, GP budget: ${epoch_gp_budget:.2f}")
                
                mp_payout = np.sum(epoch_miners_scores['scores'])
                gp_payout = np.sum(epoch_gp_scores['scores'])
                
                # Debug info
                print(f"  -> MP entities: {epoch_miner_history['n_entities']}, GP entities: {epoch_gp_history['n_entities']}")
                print(f"  -> MP eligible: {epoch_miners_scores['sol1']['num_eligible'] if epoch_miners_scores['sol1'] else 0}, GP eligible: {epoch_gp_scores['sol1']['num_eligible'] if epoch_gp_scores['sol1'] else 0}")
                
                # More detailed debugging for optimization results
                if epoch_miners_scores['sol1']:
                    print(f"  -> MP Phase 1 status: {epoch_miners_scores['sol1']['status']}, T*: ${epoch_miners_scores['sol1']['T_star']:.2f}")
                if epoch_gp_scores['sol1']:
                    print(f"  -> GP Phase 1 status: {epoch_gp_scores['sol1']['status']}, T*: ${epoch_gp_scores['sol1']['T_star']:.2f}")
                
                # Debug eligibility details for a few entities
                if epoch_miner_history['n_entities'] > 0:
                    total_vol = np.sum(epoch_miner_history['v_prev'], axis=0)
                    print(f"  -> MP total volume range: ${np.min(total_vol):.2f} - ${np.max(total_vol):.2f}")
                    print(f"  -> MP entities with vol>0: {np.sum(total_vol > 0)}")
                
                # Debug ramp constraint
                if epoch_miners_scores['sol1'] and epoch_miners_scores['sol1']['x_star'] is not None:
                    x_star = epoch_miners_scores['sol1']['x_star']
                    print(f"  -> MP max x_star: {np.max(x_star):.4f}, entities with x>0: {np.sum(x_star > 1e-6)}")
                    
                    # Debug budget per eligible entity
                    eligible_count = epoch_miners_scores['sol1']['num_eligible']
                    if eligible_count > 0:
                        budget_per_eligible = epoch_mp_budget / eligible_count
                        print(f"  -> MP budget per eligible: ${budget_per_eligible:.2f}")
                        
                        # Debug ROI for comparison between epochs
                        if epoch_idx in [25, 28, 29]:  # Compare a few epochs
                            if epoch_miner_history['n_entities'] > 0:
                                # Calculate ROI from the data we have
                                total_vol = np.sum(epoch_miner_history['v_prev'], axis=0)
                                total_profit = np.sum(epoch_miner_history['profit_prev'], axis=0)
                                roi_trailing = np.divide(total_profit, np.maximum(total_vol, 1e-12))
                                
                                funded_mask = x_star > 1e-6
                                if np.any(funded_mask):
                                    funded_rois = roi_trailing[funded_mask]
                                    print(f"  -> EPOCH {epoch_idx}: ROI range for funded entities: {np.min(funded_rois):.4f} to {np.max(funded_rois):.4f}")
                                    print(f"  -> EPOCH {epoch_idx}: Entities with ROI > 0: {np.sum(funded_rois > 0)}")
                                    print(f"  -> EPOCH {epoch_idx}: Entities with ROI <= 0: {np.sum(funded_rois <= 0)}")
                                else:
                                    print(f"  -> EPOCH {epoch_idx}: No entities funded, but ROI range for all: {np.min(roi_trailing):.4f} to {np.max(roi_trailing):.4f}")
                                    print(f"  -> EPOCH {epoch_idx}: All entities with ROI > 0: {np.sum(roi_trailing > 0)}")
                                
                                # Debug constraint analysis
                                if epoch_miners_scores['sol1']:
                                    T_star = epoch_miners_scores['sol1']['T_star']
                                    print(f"  -> EPOCH {epoch_idx}: T* = ${T_star:.2f}, Budget = ${epoch_mp_budget:.2f}")
                                    if T_star > 0:
                                        print(f"  -> EPOCH {epoch_idx}: Budget utilization: {T_star/epoch_mp_budget*100:.1f}%")
                                    else:
                                        print(f"  -> EPOCH {epoch_idx}: Why is T* = 0 when ROI > 0 and budget exists?")
                                        
                                    # Debug kappa constraint
                                    if epoch_miner_history['n_epochs'] < 5:
                                        print(f"  -> EPOCH {epoch_idx}: Using bootstrap kappa = 0.05 (n_epochs = {epoch_miner_history['n_epochs']})")
                                    else:
                                        print(f"  -> EPOCH {epoch_idx}: Using calculated kappa (n_epochs = {epoch_miner_history['n_epochs']})")
                                    
                                    # Check if ROI violates kappa constraint
                                    max_roi = np.max(roi_trailing)
                                    print(f"  -> EPOCH {epoch_idx}: Max ROI = {max_roi:.4f} ({max_roi*100:.1f}%)")
                                    
                                    # The constraint violation message is misleading - let's see actual kappa
                                    # We need to get the actual calculated kappa value
                                    print(f"  -> EPOCH {epoch_idx}: Need to check actual calculated kappa value")
                
                print(f"  -> MP payout: ${mp_payout:.2f}, GP payout: ${gp_payout:.2f}")
                
                mp_payouts[epoch_idx] = mp_payout
                gp_payouts[epoch_idx] = gp_payout
            else:
                print(f"  -> No trades found")
                mp_payouts[epoch_idx] = 0.0
                gp_payouts[epoch_idx] = 0.0
                
        except Exception as e:
            # If scoring fails for this epoch, set to $0
            print(f"Warning: Could not calculate payouts for epoch {epoch_idx}: {e}")
            mp_payouts[epoch_idx] = 0.0
            gp_payouts[epoch_idx] = 0.0
    
    print(f"Completed historical payout calculations for {n_epochs} epochs.")
    return mp_payouts, gp_payouts


def create_daily_stats_table(miner_history, general_pool_history, miners_scores=None, general_pool_scores=None, 
                            historical_mp_payouts=None, historical_gp_payouts=None):
    """Create a table of daily stats showing epoch, date, volume, budget, and payouts."""
    n_epochs = miner_history['n_epochs']
    epoch_dates = miner_history['epoch_dates']
    
    table_data = []
    for epoch_idx in range(n_epochs):
        # Calculate miner pool raw volume (qualified + unqualified) and budget for this epoch
        miner_qualified_volume = np.sum(miner_history['v_prev'][epoch_idx]) if miner_history['n_entities'] > 0 else 0.0
        miner_unqualified_volume = np.sum(miner_history['unqualified_prev'][epoch_idx]) if miner_history['n_entities'] > 0 else 0.0
        miner_volume = miner_qualified_volume + miner_unqualified_volume
        raw_miner_budget = np.sum(miner_history['fees_prev'][epoch_idx]) if miner_history['n_entities'] > 0 else 0.0
        
        # Calculate general pool raw volume (qualified + unqualified) and budget for this epoch
        gp_qualified_volume = np.sum(general_pool_history['v_prev'][epoch_idx]) if general_pool_history['n_entities'] > 0 else 0.0
        gp_unqualified_volume = np.sum(general_pool_history['unqualified_prev'][epoch_idx]) if general_pool_history['n_entities'] > 0 else 0.0
        gp_volume = gp_qualified_volume + gp_unqualified_volume
        raw_gp_budget = np.sum(general_pool_history['fees_prev'][epoch_idx]) if general_pool_history['n_entities'] > 0 else 0.0
        
        # Calculate totals
        total_volume = miner_volume + gp_volume
        total_budget = raw_miner_budget + raw_gp_budget
        mp_budget = total_budget * MINER_WEIGHT_PERCENTAGE
        gp_budget = total_budget * GENERAL_POOL_WEIGHT_PERCENTAGE
        total_budget = mp_budget + gp_budget
        
        # Get payouts for this epoch
        if epoch_idx == n_epochs - 1:
            # Current epoch (most recent) - use provided scores
            if miners_scores is not None and general_pool_scores is not None:
                mp_payouts = np.sum(miners_scores['scores'])
                gp_payouts = np.sum(general_pool_scores['scores'])
            else:
                mp_payouts = 0.0
                gp_payouts = 0.0
        else:
            # Historical epochs - use pre-calculated payouts
            if historical_mp_payouts is not None and historical_gp_payouts is not None:
                mp_payouts = historical_mp_payouts[epoch_idx]
                gp_payouts = historical_gp_payouts[epoch_idx]
            else:
                mp_payouts = 0.0
                gp_payouts = 0.0
        
        total_payouts = mp_payouts + gp_payouts
        
        # Calculate payout percentage (payout / budget)
        payout_percentage = (total_payouts / total_budget * 100) if total_budget > 0 else 0.0
        
        table_data.append([
            epoch_idx,
            epoch_dates[epoch_idx],
            f"${miner_volume:,.0f}",
            f"${gp_volume:,.0f}",
            f"${total_volume:,.0f}",
            f"${raw_miner_budget:,.0f}",
            f"${raw_gp_budget:,.0f}",
            f"${mp_budget:,.0f}",
            f"${gp_budget:,.0f}",
            f"${total_budget:,.0f}",
            f"${mp_payouts:,.0f}",
            f"${gp_payouts:,.0f}",
            f"${total_payouts:,.0f}",
            f"{payout_percentage:.1f}%"
        ])
    
    return table_data


def analyze_pool_stats(miner_history, general_pool_history):
    """Analyze and print historical stats for both pools."""
    
    print("\n" + "="*80)
    print("HISTORICAL POOL STATISTICS")
    print("="*80)
    
    # Analyze miner pool
    if miner_history['n_entities'] > 0:
        print("\n--- MINER POOL HISTORICAL STATS ---")
        miner_stats = create_pool_stats_table(miner_history, "Miner")
        headers = ["UID", "Num Epochs", "Preds", "Total Volume", "Qualified Volume", "PNL", "ROI"]
        print(tabulate(miner_stats, headers=headers, tablefmt="grid", stralign="right"))
    else:
        print("\n--- MINER POOL HISTORICAL STATS ---")
        print("No miners found in data")
    
    # Analyze general pool
    if general_pool_history['n_entities'] > 0:
        print("\n--- GENERAL POOL HISTORICAL STATS ---")
        gp_stats = create_pool_stats_table(general_pool_history, "General")
        headers = ["PID", "Num Epochs", "Preds", "Total Volume", "Qualified Volume", "PNL", "ROI"]
        print(tabulate(gp_stats, headers=headers, tablefmt="grid", stralign="right"))
    else:
        print("\n--- GENERAL POOL HISTORICAL STATS ---")
        print("No general pool users found in data")


def create_pool_stats_table(epoch_history, pool_type):
    """Create a table of historical stats for a pool."""
    v_prev_matrix = epoch_history["v_prev"]           # (n_epochs, n_entities) - qualified volume
    unqualified_matrix = epoch_history["unqualified_prev"]  # (n_epochs, n_entities) - unqualified volume
    profit_matrix = epoch_history["profit_prev"]      # (n_epochs, n_entities) - profit
    trade_counts_matrix = epoch_history["trade_counts"]  # (n_epochs, n_entities) - number of trades
    entity_ids = epoch_history["entity_ids"]
    n_entities = epoch_history["n_entities"]
    n_epochs = epoch_history["n_epochs"]
    
    if n_entities == 0:
        return []
    
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
        
        # Format entity ID based on pool type
        if pool_type == "Miner":
            display_id = f"{entity_id}"
        else:
            display_id = entity_id
        
        table_data.append([
            display_id,
            int(trading_epochs),
            total_predictions,
            f"${total_volume:,.0f}",
            f"${qualified_volume:,.0f}",
            f"${total_pnl:,.2f}",
            f"{roi:.4f}"
        ])
    
    # Sort by total volume (descending)
    #table_data.sort(key=lambda x: float(x[2].replace('$', '').replace(',', '')), reverse=True)
    # Sort by UID (ascending)
    #table_data.sort(key=lambda x: x[0], reverse=True)
    # Sort by PnL (descending)
    table_data.sort(key=lambda x: float(x[5].replace('$', '').replace(',', '')), reverse=True)
    
    return table_data


def print_results(miner_history, general_pool_history, miners_scores, general_pool_scores, 
                  miner_budget, gp_budget, historical_mp_payouts, historical_gp_payouts):
    """Print formatted results from scoring."""
    
    print("\n" + "="*80)
    print("SCORING SIMULATION RESULTS")
    print("="*80)
    
    # Daily stats table
    print(f"\n--- DAILY STATS (Last 30 Epochs) ---")
    daily_stats = create_daily_stats_table(
        miner_history, 
        general_pool_history, 
        miners_scores, 
        general_pool_scores,
        historical_mp_payouts,
        historical_gp_payouts
    )
    headers = ["Epoch", "Date", "MP Vol", "GP Vol", "Volume", "Raw MP Budget", "Raw GP Budget", "MP Budget", "GP Budget", "Budget", "MP Payout", "GP Payout", "Payout", "Payout %"]
    print(tabulate(daily_stats, headers=headers, tablefmt="grid", stralign="right"))
    
    # Budget information
    print("\n--- BUDGET ALLOCATION ---")
    print(f"Miner Pool Budget:        ${miner_budget:,.2f}")
    print(f"General Pool Budget:      ${gp_budget:,.2f}")
    print(f"Total Budget:             ${miner_budget + gp_budget:,.2f}")
    
    # Miner pool results
    print("\n--- MINER POOL RESULTS ---")
    print(f"Number of miners:         {miner_history['n_entities']}")
    print(f"Eligible miners:          {miners_scores['sol1']['num_eligible'] if miners_scores['sol1'] else 0}")
    print(f"Funded miners:            {miners_scores['sol1']['num_funded'] if miners_scores['sol1'] else 0}")
    
    if miners_scores['sol1']:
        print(f"Phase 1 Status:           {miners_scores['sol1']['status']}")
        print(f"Phase 1 T*:               ${miners_scores['sol1']['T_star']:,.2f}")
        print(f"Phase 1 Payout:           ${miners_scores['sol1']['payout']:,.2f}")
    
    if miners_scores['sol2']:
        print(f"Phase 2 Status:           {miners_scores['sol2']['status']}")
        print(f"Phase 2 T:                ${miners_scores['sol2']['T_val']:,.2f}")
        print(f"Phase 2 Payout:           ${miners_scores['sol2']['payout']:,.2f}")
    
    # Top miners
    if len(miners_scores['scores']) > 0:
        print("\n--- TOP 10 MINERS BY SCORE ---")
        scores = miners_scores['scores']
        entity_ids = miners_scores['entity_ids']
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        
        print(f"{'Rank':<6} {'Miner ID':<10} {'Score':<15} {'x_opt':<10} {'ROI':<10}")
        print("-" * 60)
        for i, idx in enumerate(sorted_indices[:10]):
            if scores[idx] > 1e-9:  # Only show non-zero scores
                miner_id = entity_ids[idx]
                score = scores[idx]
                x_opt = miners_scores['x_opt'][idx]
                roi = miners_scores['roi_trailing'][idx]
                print(f"{i+1:<6} {miner_id:<10} ${score:<14.2f} {x_opt:<10.4f} {roi:<10.4f}")
    
    # General pool results
    print("\n--- GENERAL POOL RESULTS ---")
    print(f"Number of users:          {general_pool_history['n_entities']}")
    print(f"Eligible users:           {general_pool_scores['sol1']['num_eligible'] if general_pool_scores['sol1'] else 0}")
    print(f"Funded users:             {general_pool_scores['sol1']['num_funded'] if general_pool_scores['sol1'] else 0}")
    
    if general_pool_scores['sol1']:
        print(f"Phase 1 Status:           {general_pool_scores['sol1']['status']}")
        print(f"Phase 1 T*:               ${general_pool_scores['sol1']['T_star']:,.2f}")
        print(f"Phase 1 Payout:           ${general_pool_scores['sol1']['payout']:,.2f}")
    
    if general_pool_scores['sol2']:
        print(f"Phase 2 Status:           {general_pool_scores['sol2']['status']}")
        print(f"Phase 2 T:                ${general_pool_scores['sol2']['T_val']:,.2f}")
        print(f"Phase 2 Payout:           ${general_pool_scores['sol2']['payout']:,.2f}")
    
    # Top general pool users
    if len(general_pool_scores['scores']) > 0:
        print("\n--- TOP 10 GENERAL POOL USERS BY SCORE ---")
        scores = general_pool_scores['scores']
        entity_ids = general_pool_scores['entity_ids']
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        
        print(f"{'Rank':<6} {'Profile ID':<20} {'Score':<15} {'x_opt':<10} {'ROI':<10}")
        print("-" * 70)
        for i, idx in enumerate(sorted_indices[:10]):
            if scores[idx] > 1e-9:  # Only show non-zero scores
                profile_id = entity_ids[idx]
                score = scores[idx]
                x_opt = general_pool_scores['x_opt'][idx]
                roi = general_pool_scores['roi_trailing'][idx]
                print(f"{i+1:<6} {profile_id:<20} ${score:<14.2f} {x_opt:<10.4f} {roi:<10.4f}")
    
    # Summary statistics
    print("\n--- SUMMARY STATISTICS ---")
    total_miner_payout = np.sum(miners_scores['scores'])
    total_gp_payout = np.sum(general_pool_scores['scores'])
    print(f"Total Miner Payouts:      ${total_miner_payout:,.2f}")
    print(f"Total GP Payouts:         ${total_gp_payout:,.2f}")
    print(f"Total Payouts:            ${total_miner_payout + total_gp_payout:,.2f}")
    print(f"Miner Budget Utilization: {(total_miner_payout/miner_budget*100) if miner_budget > 0 else 0:.1f}%")
    print(f"GP Budget Utilization:    {(total_gp_payout/gp_budget*100) if gp_budget > 0 else 0:.1f}%")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main simulation function."""
    print("Loading mock trading data...")
    
    # Load the mock data
    trading_history = load_mock_data()
    print(f"Loaded {len(trading_history)} trades")
    
    # Extract miner information
    print("\nExtracting miner information...")
    all_uids, all_hotkeys = extract_miner_info(trading_history)
    print(f"Found {len(all_uids)} unique miners")
    print(f"Miner UIDs: {all_uids}")
    
    # Run the scoring function
    print("\nRunning scoring algorithm...")
    print("This may take a moment...\n")
    
    miner_history, general_pool_history, miners_scores, general_pool_scores, \
        miner_budget, gp_budget = score_miners(
            all_uids=all_uids,
            all_hotkeys=all_hotkeys,
            trading_history=trading_history
        )
    
    # Calculate historical payouts for all epochs
    historical_mp_payouts, historical_gp_payouts = calculate_historical_payouts(
        miner_history, general_pool_history, all_uids, all_hotkeys, trading_history
    )
    
    # Print historical pool statistics first
    analyze_pool_stats(miner_history, general_pool_history)
    
    # Print scoring results
    print_results(
        miner_history, 
        general_pool_history, 
        miners_scores, 
        general_pool_scores,
        miner_budget,
        gp_budget,
        historical_mp_payouts,
        historical_gp_payouts
    )


if __name__ == "__main__":
    main()