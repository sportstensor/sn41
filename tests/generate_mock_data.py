#!/usr/bin/env python3
"""
Mock Trading Data Generator for Prediction Market Scoring System

This script generates synthetic trading data for testing the miner scoring system.
It creates realistic sports prediction market data with varied miner behaviors.
"""

import random
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import argparse
import json

# Configuration
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

def generate_markets(num_markets: int = 50) -> List[str]:
    """Generate a list of market IDs"""
    markets = []
    for _ in range(num_markets):
        league = random.choice(list(SPORTS_LEAGUES.keys()))
        teams = random.sample(SPORTS_LEAGUES[league], 2)
        year = 2025
        market_id = generate_market_id(league, teams[0], teams[1], year)
        markets.append(market_id)
    return markets

def generate_prediction_dates(base_date: datetime, num_predictions: int) -> List[datetime]:
    """Generate prediction dates with realistic distribution"""
    dates = []

    # 30% oldest predictions (settled, oldest)
    oldest_count = int(num_predictions * 0.3)
    for _ in range(oldest_count):
        #days_ago = random.randint(31, 60)  # 31-60 days ago
        #days_ago = random.randint(20, 30)  # 20-30 days ago
        days_ago = random.randint(40, 60)  # 40-60 days ago
        dates.append(base_date - timedelta(days=days_ago))
    
    # 30% older predictions (settled, for time decay testing)
    older_count = int(num_predictions * 0.3)
    for _ in range(older_count):
        #days_ago = random.randint(10, 20)  # 10-20 days ago
        #days_ago = random.randint(1, 10)  # 1-10 days ago
        days_ago = random.randint(20, 40)  # 20-40 days ago
        dates.append(base_date - timedelta(days=days_ago))
    
    # 30% recent predictions (settled, recent)
    recent_count = int(num_predictions * 0.3)
    for _ in range(recent_count):
        #days_ago = random.randint(1, 10)  # 1-10 days ago
        days_ago = random.randint(1, 20)  # 1-20 days ago
        dates.append(base_date - timedelta(days=days_ago))
    
    # 10% future predictions (unsettled)
    future_count = num_predictions - oldest_count - older_count - recent_count
    for _ in range(future_count):
        days_ahead = random.randint(1, 20)  # 1-20 days in future
        dates.append(base_date + timedelta(days=days_ahead))
    
    return dates

def generate_miner_data(profile_id: str, miner_id: int, miner_hotkey: str, is_general_pool: bool, num_predictions: int, markets: List[str], 
                       base_date: datetime, min_volume: int = 100, max_volume: int = 10000) -> List[Dict]:
    """Generate prediction data for a single miner"""
    predictions = []
    
    # Generate prediction dates
    prediction_dates = generate_prediction_dates(base_date, num_predictions)
    
    # Determine settled vs unsettled ratio (most miners should have >10 settled)
    settled_ratio = 0.9 if miner_id != 10 else 0.3  # miner uid 10 has fewer settled
    settled_count = int(num_predictions * settled_ratio)
    
    for i in range(num_predictions):
        # Basic prediction data
        volume = random.randint(min_volume, max_volume)
        #is_correct = random.choice([True, False])  # 50% win rate
        is_correct = random.uniform(0, 1) > 0.4
        market_id = random.choice(markets)
        date_created = prediction_dates[i]
        
        # Determine if settled - future dates are ALWAYS unsettled
        if date_created > base_date:
            is_settled = False  # Future dates are never settled
        else:
            is_settled = i < settled_count  # Past dates follow the settled ratio
        
        # Calculate PnL
        if is_correct:
            pnl = volume * random.uniform(0.1, 1.2)
        else:
            pnl = -volume
        
        # Generate settlement date if settled
        if is_settled:
            days_to_settle = random.randint(1, 3)
            date_settled = date_created + timedelta(days=days_to_settle)
        else:
            date_settled = None
        
        # Trade type and price
        trade_type = random.choice(['buy', 'sell'])
        price = random.uniform(0.3, 0.8)
        
        prediction = {
            'trade_id': len(predictions) + 1,
            'profile_id': profile_id,
            'miner_id': int(miner_id) if miner_id is not None else None,
            'miner_hotkey': miner_hotkey,
            'is_general_pool': is_general_pool,
            'market_id': market_id,
            'date_created': date_created.strftime('%Y-%m-%d'),
            'volume': volume,
            'pnl': pnl,
            'is_correct': is_correct,
            'is_settled': is_settled,
            'date_settled': date_settled.strftime('%Y-%m-%d') if date_settled else None,
            'trade_type': trade_type,
            'price': round(price, 1)
        }
        
        predictions.append(prediction)
    
    return predictions

def generate_mock_data(num_miners: int = 10, base_date: str = "today", 
                      output_file: str = "mock_trading_data.csv", output_format: str = "csv") -> None:
    """Generate complete mock trading dataset"""
    
    # Parse base date
    if base_date == 'today':
        base_date = datetime.now()
    else:
        base_date = datetime.strptime(base_date, '%Y-%m-%d')
    
    # Generate markets
    markets = generate_markets(50)
    
    # Generate miner data with varied prediction counts
    all_predictions = []
    trade_id_counter = 1
    
    for i in range(num_miners):
        #miner_id = f"miner_{i+1:03d}"
        miner_id = i + 1
        profile_id = f"0x{random.randint(1, 100):04x}"
        miner_hotkey = f"5F{random.randint(1, 100):04x}"
        
        # 30% chance of being a general pool
        is_general_pool = random.choice([True, False]) if i % 3 == 0 else False

        if is_general_pool:
            miner_id = None
            miner_hotkey = None
        
        num_predictions = random.randint(1, 100)
        print(f"Generating {num_predictions} predictions for {profile_id}...")
        
        miner_predictions = generate_miner_data(
            profile_id=profile_id,
            miner_id=miner_id,
            miner_hotkey=miner_hotkey,
            is_general_pool=is_general_pool,
            num_predictions=num_predictions,
            markets=markets,
            base_date=base_date
        )
        
        # Update trade IDs to be sequential
        for prediction in miner_predictions:
            prediction['trade_id'] = trade_id_counter
            trade_id_counter += 1
        
        all_predictions.extend(miner_predictions)
    
    # Create DataFrame and save
    df = pd.DataFrame(all_predictions)
    
    # Convert miner_id to nullable integer type to preserve ints in JSON
    df['miner_id'] = df['miner_id'].astype('Int64')
    
    if output_format.lower() == 'json':
        output_file = output_file.replace('.csv', '.json')
        df.to_json(output_file, orient='records', indent=2, date_format='iso')
    else:
        output_file = output_file.replace('.json', '.csv')
        df.to_csv(output_file, index=False)
    
    print(f"\nGenerated {len(all_predictions)} predictions for {num_miners} miners")
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
        'is_settled': 'sum',
        'trade_id': 'count'
    }).round(2)
    miner_summary.columns = ['Total_Volume', 'Total_PnL', 'Wins', 'Settled', 'Total_Predictions']
    miner_summary['Win_Rate'] = (miner_summary['Wins'] / miner_summary['Total_Predictions'] * 100).round(1)
    print(miner_summary)

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Generate mock trading data for prediction market scoring')
    parser.add_argument('--miners', type=int, default=10, help='Number of miners (default: 10)')
    parser.add_argument('--date', type=str, default='today', help='Base date for predictions (ie: 2025-09-10 or today)')
    parser.add_argument('--output', type=str, default='mock_trading_data.json', help='Output file (default: mock_trading_data.json)')
    parser.add_argument('--format', type=str, default='json', choices=['csv', 'json'], help='Output format: csv or json (default: json)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    if args.seed:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    generate_mock_data(
        num_miners=args.miners,
        base_date=args.date,
        output_file="tests/" + args.output,
        output_format=args.format
    )

if __name__ == "__main__":
    main()
