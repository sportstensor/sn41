#!/usr/bin/env python3
"""
Advanced Mock Trading Data Generator for Prediction Market Scoring System

This script generates sophisticated synthetic trading data using realistic
gambler behavior, skill-based trading logic, and market dynamics.
It combines simulation logic from the sims/ directory into a self-contained generator.
"""

import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import argparse
import json
from dataclasses import dataclass
from scipy.special import logit, expit
from numpy.random import default_rng

# =============================================================================
# CONFIGURATION - Based on sims/config.py
# =============================================================================

# Mixture weights for skill distribution (losers, break-even, pros)
SKILL_WEIGHTS = (0.90, 0.08, 0.02)

# Means and stds for each component (ROI per bet)
SKILL_MEANS = (-0.10, 0.00, 0.02)   # losing, average, skilled
SKILL_STDS = (0.05, 0.03, 0.03)

# Bankroll distribution (lognormal)
BANKROLL_MU = 11.0
BANKROLL_SIGMA = 0.8
BANKROLL_FLOOR = 50_000.0

# Correlation between skill and bankroll
RHO = 0.1

# Kelly fraction beta distribution
KELLY_ALPHA = 0.7
KELLY_BETA = 3.0

# Protocol fee
FEE_RATE = 0.01

# =============================================================================
# MINER PROFILE TYPES
# =============================================================================

@dataclass
class MinerProfile:
    """Defines a miner trading profile"""
    name: str
    skill_mean: float
    skill_std: float
    bankroll_multiplier: float
    participation_rate: float
    kelly_aggression_range: Tuple[float, float]
    volume_range: Tuple[int, int]
    trading_frequency: float  # trades per day
    description: str

# Predefined miner profiles
MINER_PROFILES = {
    'whale': MinerProfile(
        name='whale',
        skill_mean=0.01,
        skill_std=0.02,
        bankroll_multiplier=10.0,
        participation_rate=0.8,
        kelly_aggression_range=(0.3, 0.7),
        volume_range=(5000, 10000),
        trading_frequency=2.0,
        description='High-volume professional traders with significant capital'
    ),
    'professional': MinerProfile(
        name='professional',
        skill_mean=0.005,
        skill_std=0.015,
        bankroll_multiplier=3.0,
        participation_rate=0.9,
        kelly_aggression_range=(0.2, 0.6),
        volume_range=(500, 5000),
        trading_frequency=3.0,
        description='Skilled traders with consistent strategies'
    ),
    'casual': MinerProfile(
        name='casual',
        skill_mean=-0.05,
        skill_std=0.08,
        bankroll_multiplier=1.0,
        participation_rate=0.3,
        kelly_aggression_range=(0.1, 0.4),
        volume_range=(100, 2000),
        trading_frequency=0.5,
        description='Occasional traders with lower skill and capital'
    ),
    'degenerate': MinerProfile(
        name='degenerate',
        skill_mean=-0.15,
        skill_std=0.1,
        bankroll_multiplier=0.5,
        participation_rate=0.95,
        kelly_aggression_range=(0.5, 0.9),
        volume_range=(100, 5000),
        trading_frequency=5.0,
        description='High-frequency traders with poor risk management'
    ),
    'break_even': MinerProfile(
        name='break_even',
        skill_mean=0.0,
        skill_std=0.05,
        bankroll_multiplier=1.5,
        participation_rate=0.6,
        kelly_aggression_range=(0.2, 0.5),
        volume_range=(100, 1000),
        trading_frequency=1.5,
        description='Average traders who roughly break even'
    )
}

# =============================================================================
# GAMBLER CLASS - Based on sims/gambler.py
# =============================================================================

class Gambler:
    """Individual gambler with skill-based betting behavior"""
    
    def __init__(self, skill: float, bankroll: float, profile: MinerProfile, rng):
        self.skill = skill
        self.bankroll = bankroll
        self.original_bankroll = bankroll
        self.profile = profile
        self.rng = rng
        
        # Kelly aggression based on profile
        kelly_alpha, kelly_beta = KELLY_ALPHA, KELLY_BETA
        self.kelly_aggression = float(np.clip(
            self.rng.beta(kelly_alpha, kelly_beta),
            profile.kelly_aggression_range[0], 
            profile.kelly_aggression_range[1]
        ))
        
        # Trading history
        self.trades = []
        self.total_volume = 0.0
        self.total_pnl = 0.0
        
    def stake_size(self, odds: float) -> float:
        """Compute stake size via Kelly fraction and perceived edge"""
        fair_prob = 1.0 / odds
        
        # Perceived probability based on skill
        p = np.clip(fair_prob + self.skill * (1 - fair_prob), 1e-6, 1 - 1e-6)
        
        # Kelly fraction
        f_kelly = (p * (odds - 1) - (1 - p)) / (odds - 1)
        
        # Confidence factor: weak → >1 (overbet), strong → <1 (conservative)
        conf = 1.5 - np.tanh(5 * self.skill)
        f_star = np.clip(f_kelly * conf * self.kelly_aggression, 0.01, 0.15)
        
        stake = float(self.bankroll * f_star)
        
        # Apply volume constraints from profile
        min_vol, max_vol = self.profile.volume_range
        stake = np.clip(stake, min_vol, max_vol)
        
        return stake
    
    def should_trade_today(self, base_participation: float = 0.25) -> bool:
        """Determine if gambler should trade today"""
        # Combine base participation with profile participation rate
        effective_rate = base_participation * self.profile.participation_rate
        return self.rng.random() < effective_rate
    
    def update_bankroll(self, win: bool, stake: float, odds: float) -> None:
        """Update bankroll after a bet resolves"""
        if win:
            profit = stake * (odds - 1.0)
            self.bankroll += profit
            self.total_pnl += profit
        else:
            self.bankroll -= stake
            self.total_pnl -= stake
        
        self.total_volume += stake

# =============================================================================
# MARKET GENERATION
# =============================================================================

SPORTS_LEAGUES = {
    'mlb': ['yankees', 'redsox', 'dodgers', 'giants', 'astros', 'rangers', 'braves', 'mets', 
            'guardians', 'twins', 'mariners', 'angels', 'royals', 'whitesox', 'tigers', 'rays',
            'athletics', 'giants', 'brewers', 'cubs', 'padres', 'dodgers', 'reds', 'pirates',
            'bluejays', 'orioles', 'nationals', 'marlins', 'phillies', 'marlins'],
    'nba': ['lakers', 'warriors', 'celtics', 'heat', 'nuggets', 'suns', 'bucks', 'nets',
            '76ers', 'knicks', 'thunder', 'grizzlies', 'jazz', 'trailblazers', 'magic', 'hawks',
            'pistons', 'cavs', 'wizards', 'hornets', 'kings', 'clippers', 'timberwolves', 'rockets',
            'spurs', 'mavericks', 'pacers', 'raptors', 'nets', 'knicks'],
    'nfl': ['cowboys', 'eagles', 'patriots', 'bills', 'chiefs', 'ravens', '49ers', 'rams',
            'steelers', 'bengals', 'packers', 'vikings', 'titans', 'colts', 'browns', 'ravens',
            'jets', 'dolphins', 'texans', 'jaguars', 'cardinals', 'seahawks', 'falcons', 'saints',
            'panthers', 'bucs', 'broncos', 'raiders', 'giants', 'commanders', 'bears', 'lions']
}

def generate_market_id(league: str, team1: str, team2: str, year: int) -> str:
    """Generate a market ID for a sports game"""
    return f"{league}_{team1}_{team2}_{year}"

def generate_markets(num_markets: int = 50, year: int = 2025) -> List[str]:
    """Generate a list of market IDs"""
    markets = []
    for _ in range(num_markets):
        league = random.choice(list(SPORTS_LEAGUES.keys()))
        teams = random.sample(SPORTS_LEAGUES[league], 2)
        market_id = generate_market_id(league, teams[0], teams[1], year)
        markets.append(market_id)
    return markets

# =============================================================================
# TRADING SIMULATION - Based on sims/ledger.py
# =============================================================================

class TradingSimulator:
    """Simulates realistic trading behavior with market dynamics"""
    
    def __init__(self, seed: int = 0):
        self.rng = default_rng(seed)
        self.trades = []
        self.trade_id_counter = 1
        
    def generate_odds(self, base_odds: float = 2.0, odds_sigma: float = 0.3) -> float:
        """Generate realistic odds with lognormal distribution"""
        return float(max(1.01, np.exp(self.rng.normal(np.log(base_odds), odds_sigma))))
    
    def simulate_trade(self, gambler: Gambler, market_id: str, date: datetime, 
                      base_odds: float = 2.0, odds_sigma: float = 0.3) -> Optional[Dict]:
        """Simulate a single trade for a gambler"""
        
        # Generate odds
        odds = self.generate_odds(base_odds, odds_sigma)
        fair_prob = 1.0 / odds
        
        # Hidden "true" probability of event (unknown to bookmaker)
        world_sigma = 0.05
        true_prob = expit(logit(fair_prob) + self.rng.normal(0, world_sigma))
        
        # Bettor's perception based on skill
        λ = np.tanh(gambler.skill * 5.0)
        p_true = expit((1 - λ) * logit(fair_prob) + λ * logit(true_prob))
        
        # Calculate stake size
        stake = gambler.stake_size(odds)
        if stake <= 0:
            return None
        
        # Determine outcome
        win = self.rng.random() < p_true
        profit = stake * (odds - 1.0) if win else -stake
        
        # Update gambler
        gambler.update_bankroll(win, stake, odds)
        
        # Generate trade data
        trade = {
            'trade_id': self.trade_id_counter,
            'account_id': gambler.account_id,
            'profile_id': gambler.profile_id,
            'miner_id': gambler.miner_id,
            'miner_hotkey': gambler.miner_hotkey,
            'is_general_pool': gambler.is_general_pool,
            'market_id': market_id,
            'date_created': date.strftime('%Y-%m-%d'),
            'volume': round(stake, 2),
            'pnl': round(profit, 2),
            'is_correct': win,
            'is_settled': True,  # All trades are settled for simulation
            'date_settled': (date + timedelta(days=int(self.rng.integers(1, 4)))).strftime('%Y-%m-%d'),
            'trade_type': self.rng.choice(['buy', 'sell']),
            'price': round(self.rng.uniform(0.3, 0.8), 1),
            'is_reward_eligible': True
        }
        
        self.trade_id_counter += 1
        self.trades.append(trade)
        
        return trade

# =============================================================================
# MINER GENERATION
# =============================================================================

def create_miner(uid: int, profile_name: str, account_id: int, rng) -> Gambler:
    """Create a miner with specified profile"""
    profile = MINER_PROFILES[profile_name]
    
    # Generate skill using profile parameters
    skill = rng.normal(profile.skill_mean, profile.skill_std)
    
    # Generate bankroll with correlation to skill
    base_bankroll = np.exp(rng.normal(BANKROLL_MU, BANKROLL_SIGMA)) * BANKROLL_FLOOR
    skill_correlation = RHO * skill * base_bankroll
    bankroll = max(BANKROLL_FLOOR, base_bankroll + skill_correlation) * profile.bankroll_multiplier
    
    # Create gambler
    gambler = Gambler(skill, bankroll, profile, rng)
    gambler.miner_id = uid
    gambler.miner_hotkey = f"5F{rng.integers(1, 10000):04x}"
    gambler.profile_id = f"0x{rng.integers(1, 1000):04x}"  # Generate unique profile_id for this miner
    gambler.is_general_pool = False

    # Generate account ID
    gambler.account_id = account_id
    
    return gambler

def generate_miner_profiles(num_miners: int, profile_distribution: Dict[str, float], 
                          general_pool_percentage: float, rng) -> List[Gambler]:
    """Generate miners with specified profile distribution"""
    gamblers = []
    
    for i in range(num_miners):
        # Select profile based on distribution
        profile_name = rng.choice(list(profile_distribution.keys()), 
                                 p=list(profile_distribution.values()))
        
        account_id = i + 1
        gambler = create_miner(i + 1, profile_name, account_id, rng)
        
        # Determine if this miner is in the general pool
        is_general_pool = rng.random() < general_pool_percentage
        gambler.is_general_pool = is_general_pool
        
        # If general pool, set miner_id and miner_hotkey to None
        if is_general_pool:
            gambler.miner_id = None
            gambler.miner_hotkey = None
        
        gamblers.append(gambler)
    
    return gamblers

# =============================================================================
# DATE GENERATION
# =============================================================================

def generate_trading_dates(base_date: datetime, num_days: int, 
                          include_weekends: bool = True) -> List[datetime]:
    """Generate trading dates with realistic patterns"""
    dates = []
    current_date = base_date - timedelta(days=num_days)
    
    for _ in range(num_days):
        if include_weekends or current_date.weekday() < 5:  # Monday=0, Sunday=6
            dates.append(current_date)
        current_date += timedelta(days=1)
    
    return dates

# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_advanced_mock_data(
    num_miners: int = 10,
    num_days: int = 60,
    base_date: str = "today",
    output_file: str = "tests/advanced_mock_data.json",
    output_format: str = "json",
    profile_distribution: Dict[str, float] = None,
    num_markets: int = 50,
    base_odds: float = 2.0,
    odds_sigma: float = 0.3,
    participation_rate: float = 0.25,
    general_pool_percentage: float = 0.0,
    seed: int = None
) -> None:
    """Generate sophisticated mock trading data"""
    
    # Set up random number generator
    if seed is not None:
        rng = default_rng(seed)
        random.seed(seed)
    else:
        rng = default_rng()
    
    # Parse base date
    if base_date == 'today':
        base_date = datetime.now()
    else:
        base_date = datetime.strptime(base_date, '%Y-%m-%d')
    
    # Default profile distribution
    if profile_distribution is None:
        profile_distribution = {
            'casual': 0.4,
            'break_even': 0.3,
            'professional': 0.2,
            'whale': 0.05,
            'degenerate': 0.05
        }
    
    print(f"Generating data for {num_miners} miners over {num_days} days...")
    print(f"Profile distribution: {profile_distribution}")
    
    # Generate markets
    markets = generate_markets(num_markets, base_date.year)
    print(f"Generated {len(markets)} markets")
    
    # Generate miners
    gamblers = generate_miner_profiles(num_miners, profile_distribution, general_pool_percentage, rng)
    print(f"Generated {len(gamblers)} miners")
    print(f"General pool percentage: {general_pool_percentage:.1%}")
    general_pool_count = sum(1 for g in gamblers if g.is_general_pool)
    print(f"General pool miners: {general_pool_count}/{len(gamblers)}")
    
    # Generate trading dates
    trading_dates = generate_trading_dates(base_date, num_days)
    print(f"Generated {len(trading_dates)} trading days")
    
    # Initialize simulator
    simulator = TradingSimulator(seed=seed if seed else 0)
    
    # Simulate trading
    all_trades = []
    
    for date in trading_dates:
        # Determine participation rate for this day (higher on NFL Sundays)
        day_participation = participation_rate
        if date.weekday() == 6:  # Sunday
            day_participation *= 1.5  # More activity on NFL Sundays
        
        for gambler in gamblers:
            # Check if gambler should trade today
            if not gambler.should_trade_today(day_participation):
                continue
            
            # Determine number of trades for this gambler today
            num_trades_today = rng.poisson(gambler.profile.trading_frequency)
            
            for _ in range(num_trades_today):
                market_id = rng.choice(markets)
                trade = simulator.simulate_trade(gambler, market_id, date, base_odds, odds_sigma)
                if trade:
                    all_trades.append(trade)
    
    # Create DataFrame
    df = pd.DataFrame(all_trades)
    
    # Convert miner_id to nullable integer type
    df['miner_id'] = df['miner_id'].astype('Int64')
    
    # Save data
    if output_format.lower() == 'json':
        output_file = output_file.replace('.csv', '.json')
        df.to_json(output_file, orient='records', indent=2, date_format='iso')
    else:
        output_file = output_file.replace('.json', '.csv')
        df.to_csv(output_file, index=False)
    
    print(f"\nGenerated {len(all_trades)} trades for {num_miners} miners")
    print(f"Data saved to: {output_file} ({output_format.upper()} format)")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total Volume: ${df['volume'].sum():,.0f}")
    print(f"Total PnL: ${df['pnl'].sum():,.0f}")
    print(f"Overall Win Rate: {(df['is_correct'].sum() / len(df) * 100):.1f}%")
    print(f"Markets: {df['market_id'].nunique()}")
    print(f"Date Range: {df['date_created'].min()} to {df['date_created'].max()}")
    
    # Print miner summary
    print(f"\nMiner Summary:")
    miner_summary = df.groupby('miner_id').agg({
        'volume': 'sum',
        'pnl': 'sum',
        'is_correct': 'sum',
        'trade_id': 'count'
    }).round(2)
    miner_summary.columns = ['Total_Volume', 'Total_PnL', 'Wins', 'Total_Trades']
    miner_summary['Win_Rate'] = (miner_summary['Wins'] / miner_summary['Total_Trades'] * 100).round(1)
    print(miner_summary)
    
    # Print profile summary
    print(f"\nProfile Summary:")
    for gambler in gamblers:
        profile_name = gambler.profile.name
        print(f"Miner {gambler.miner_id} ({profile_name}): "
              f"Skill={gambler.skill:.3f}, "
              f"Bankroll=${gambler.original_bankroll:,.0f}, "
              f"Trades={len([t for t in all_trades if t['miner_id'] == gambler.miner_id])}, "
              f"PnL=${gambler.total_pnl:,.0f}")

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main function with comprehensive CLI"""
    parser = argparse.ArgumentParser(
        description='Generate advanced mock trading data with realistic gambler behavior',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic generation with default profiles
  python gen_mock_data.py --miners 20 --days 30
  
  # Professional-heavy scenario
  python gen_mock_data.py --miners 10 --days 60 --profiles professional:0.6,whale:0.3,casual:0.1
  
  # High-volume whale scenario
  python gen_mock_data.py --miners 5 --days 90 --profiles whale:0.8,professional:0.2 --odds-base 3.0
  
  # Degenerate gambler scenario
  python gen_mock_data.py --miners 15 --days 45 --profiles degenerate:0.5,casual:0.5
  
  # With general pool miners
  python gen_mock_data.py --miners 20 --days 30 --general-pool 0.3
  
  # Professional scenario with 50% general pool
  python gen_mock_data.py --miners 10 --days 60 --scenario professional-heavy --general-pool 0.5
        """
    )
    
    # Basic parameters
    parser.add_argument('--miners', type=int, default=10, 
                       help='Number of miners (default: 10)')
    parser.add_argument('--days', type=int, default=60, 
                       help='Number of trading days (default: 60)')
    parser.add_argument('--date', type=str, default='today', 
                       help='Base date for predictions (e.g., 2025-09-10 or today)')
    parser.add_argument('--output', type=str, default='tests/advanced_mock_data.json', 
                       help='Output file (default: tests/advanced_mock_data.json)')
    parser.add_argument('--format', type=str, default='json', choices=['csv', 'json'], 
                       help='Output format: csv or json (default: json)')
    parser.add_argument('--seed', type=int, 
                       help='Random seed for reproducible results')
    
    # Market parameters
    parser.add_argument('--markets', type=int, default=50, 
                       help='Number of markets to generate (default: 50)')
    parser.add_argument('--odds-base', type=float, default=2.0, 
                       help='Base odds for markets (default: 2.0)')
    parser.add_argument('--odds-sigma', type=float, default=0.3, 
                       help='Odds volatility (default: 0.3)')
    
    # Trading parameters
    parser.add_argument('--participation', type=float, default=0.25, 
                       help='Base participation rate (default: 0.25)')
    parser.add_argument('--general-pool', type=float, default=0.0, 
                       help='Percentage of miners in general pool (default: 0.0)')
    
    # Profile distribution
    parser.add_argument('--profiles', type=str, 
                       help='Profile distribution as comma-separated name:weight pairs '
                            '(e.g., "casual:0.4,professional:0.3,whale:0.3")')
    
    # Preset scenarios
    parser.add_argument('--scenario', type=str, 
                       choices=['balanced', 'professional-heavy', 'whale-dominant', 
                               'casual-heavy', 'degenerate-heavy'],
                       help='Use a preset scenario instead of custom profiles')
    
    args = parser.parse_args()
    
    # Handle preset scenarios
    if args.scenario:
        scenario_profiles = {
            'balanced': {'casual': 0.4, 'break_even': 0.3, 'professional': 0.2, 'whale': 0.05, 'degenerate': 0.05},
            'professional-heavy': {'professional': 0.6, 'whale': 0.3, 'casual': 0.1},
            'whale-dominant': {'whale': 0.8, 'professional': 0.2},
            'casual-heavy': {'casual': 0.6, 'break_even': 0.3, 'degenerate': 0.1},
            'degenerate-heavy': {'degenerate': 0.5, 'casual': 0.5}
        }
        profile_distribution = scenario_profiles[args.scenario]
        print(f"Using scenario '{args.scenario}': {profile_distribution}")
    elif args.profiles:
        # Parse custom profile distribution
        profile_distribution = {}
        for pair in args.profiles.split(','):
            name, weight = pair.split(':')
            profile_distribution[name.strip()] = float(weight)
        print(f"Using custom profiles: {profile_distribution}")
    else:
        profile_distribution = None  # Use default
    
    # Generate data
    generate_advanced_mock_data(
        num_miners=args.miners,
        num_days=args.days,
        base_date=args.date,
        output_file=args.output,
        output_format=args.format,
        profile_distribution=profile_distribution,
        num_markets=args.markets,
        base_odds=args.odds_base,
        odds_sigma=args.odds_sigma,
        participation_rate=args.participation,
        general_pool_percentage=args.general_pool,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
