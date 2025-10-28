#!/usr/bin/env python3
"""
Polymarket Trade Data Fetcher with CSV Export

This script fetches historical trade data from the Polymarket API for a given profile ID
and exports it to CSV format for validator scoring.
"""

import requests
import json
import csv
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import argparse
import hashlib
from tabulate import tabulate


class PolymarketTradeFetcher:
    """Fetches trade data and PnL information from Polymarket API"""
    
    def __init__(self):
        self.base_url = "https://data-api.polymarket.com"
        self.gamma_url = "https://gamma-api.polymarket.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Polymarket-Trade-Fetcher/1.0',
            'Accept': 'application/json'
        })
        # Cache for market data to avoid repeated API calls
        self.market_cache = {}
        
    def fetch_trades(self, 
                    user_address: str,
                    limit: int = 100,
                    offset: int = 0,
                    side: Optional[str] = None,
                    before: Optional[int] = None,
                    after: Optional[int] = None) -> List[Dict]:
        """
        Fetch trades for a given user address using data-api
        """
        params = {
            'user': user_address.lower(),  # Ensure lowercase address
            'limit': min(limit, 10000),
            'offset': offset
        }
        
        if side:
            params['side'] = side
        if before:
            params['before'] = before
        if after:
            params['after'] = after
            
        try:
            response = self.session.get(f"{self.base_url}/trades", params=params)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        except requests.exceptions.RequestException as e:
            print(f"Error fetching trades: {e}")
            return []
    
    def fetch_user_activity(self,
                           user_address: str,
                           activity_type: str = "TRADE",
                           limit: int = 1000,
                           offset: int = 0,
                           start: Optional[int] = None,
                           end: Optional[int] = None) -> List[Dict]:
        """
        Fetch user activity from the activity endpoint
        """
        params = {
            'user': user_address.lower(),
            'type': activity_type,
            'limit': min(limit, 10000),
            'offset': offset
        }
        
        if start:
            params['start'] = start
        if end:
            params['end'] = end
            
        try:
            response = self.session.get(f"{self.base_url}/activity", params=params)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        except requests.exceptions.RequestException as e:
            print(f"Error fetching activity: {e}")
            return []
    
    def fetch_positions(self, 
                       user_address: str,
                       limit: int = 100,
                       offset: int = 0) -> List[Dict]:
        """
        Fetch current positions for a user to get PnL data
        """
        params = {
            'user': user_address.lower(),
            'limit': min(limit, 500),
            'offset': offset
        }
            
        try:
            response = self.session.get(f"{self.base_url}/positions", params=params)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        except requests.exceptions.RequestException as e:
            print(f"Error fetching positions: {e}")
            return []
    
    def fetch_all_trades(self, 
                        user_address: str,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> List[Dict]:
        """
        Fetch all trades for a user, handling pagination
        """
        all_trades = []
        offset = 0
        limit = 1000
        
        # Convert dates to timestamps if provided
        after = int(start_date.timestamp()) if start_date else None
        before = int(end_date.timestamp()) if end_date else None
        
        while True:
            print(f"Fetching trades batch: offset={offset}")
            trades = self.fetch_trades(
                user_address=user_address,
                limit=limit,
                offset=offset,
                before=before,
                after=after
            )
            
            if not trades:
                break
                
            all_trades.extend(trades)
            offset += len(trades)
            
            if len(trades) < limit:
                break
                
            time.sleep(0.1)  # Be respectful to the API
        
        print(f"Total trades fetched: {len(all_trades)}")
        return all_trades
    
    def fetch_all_positions(self, user_address: str) -> List[Dict]:
        """
        Fetch all positions for a user, handling pagination
        """
        all_positions = []
        offset = 0
        limit = 500
        
        while True:
            positions = self.fetch_positions(
                user_address=user_address,
                limit=limit,
                offset=offset
            )
            
            if not positions:
                break
                
            all_positions.extend(positions)
            offset += len(positions)
            
            if len(positions) < limit:
                break
                
            time.sleep(0.1)
        
        return all_positions
    
    def fetch_closed_positions(self, 
                              user_address: str,
                              limit: int = 100,
                              offset: int = 0,
                              sort_by: str = 'REALIZEDPNL',
                              sort_direction: str = 'DESC',
                              markets: Optional[List[str]] = None) -> List[Dict]:
        """
        Fetch closed positions for a user to get realized PnL data
        
        Args:
            user_address: Polygon wallet address
            limit: Number of results per page (max 500)
            offset: Number of results to skip
            sort_by: Sort criteria (REALIZEDPNL, TITLE, PRICE, AVGPRICE)
            sort_direction: Sort direction (ASC, DESC)
            markets: List of condition IDs to filter by
            
        Returns:
            List of closed position dictionaries with realized PnL
        """
        params = {
            'user': user_address.lower(),
            'limit': min(limit, 500),  # API max is 500 for closed positions
            'offset': offset,
            'sortBy': sort_by,
            'sortDirection': sort_direction
        }
        
        # Add market filter if provided
        if markets:
            # Join multiple condition IDs with comma as per API docs
            params['market'] = ','.join(markets)
            
        try:
            response = self.session.get(f"{self.base_url}/closed-positions", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching closed positions: {e}")
            return []
    
    def fetch_all_closed_positions(self, 
                                 user_address: str,
                                 markets: Optional[List[str]] = None,
                                 max_positions: Optional[int] = None) -> List[Dict]:
        """
        Fetch all closed positions for a user, handling pagination and batching
        
        Args:
            user_address: Polygon wallet address
            markets: List of condition IDs to filter by
            max_positions: Maximum number of positions to fetch (None for all)
            
        Returns:
            List of all closed position dictionaries
        """
        all_positions = []
        
        if not markets:
            # If no specific markets, fetch all closed positions
            return self._fetch_all_closed_positions_paginated(user_address, max_positions)
        
        # Batch markets to avoid URL length limits and API restrictions
        batch_size = 25  # Conservative batch size for market parameter
        market_batches = [markets[i:i + batch_size] for i in range(0, len(markets), batch_size)]
        
        print(f"Fetching closed positions in {len(market_batches)} batches of up to {batch_size} markets each")
        
        for i, market_batch in enumerate(market_batches):
            print(f"Processing batch {i+1}/{len(market_batches)} with {len(market_batch)} markets")
            
            # Fetch positions for this batch of markets
            batch_positions = self._fetch_all_closed_positions_paginated(
                user_address, 
                max_positions=None,  # No limit per batch
                markets=market_batch
            )
            
            all_positions.extend(batch_positions)
            
            # Check if we've hit the max positions limit
            if max_positions and len(all_positions) >= max_positions:
                break
                
            # Small delay between batches
            time.sleep(0.2)
        
        # Remove duplicates (in case same position appears in multiple batches)
        seen_assets = set()
        unique_positions = []
        for pos in all_positions:
            asset = pos.get('asset')
            if asset and asset not in seen_assets:
                seen_assets.add(asset)
                unique_positions.append(pos)
        
        return unique_positions[:max_positions] if max_positions else unique_positions
    
    def _fetch_all_closed_positions_paginated(self, 
                                            user_address: str,
                                            max_positions: Optional[int] = None,
                                            markets: Optional[List[str]] = None) -> List[Dict]:
        """
        Internal method to fetch closed positions with pagination
        """
        all_positions = []
        offset = 0
        limit = 100  # Smaller batches for closed positions
        
        while True:
            if max_positions and len(all_positions) >= max_positions:
                break
                
            remaining = max_positions - len(all_positions) if max_positions else None
            current_limit = min(limit, remaining) if remaining else limit
            
            positions = self.fetch_closed_positions(
                user_address=user_address,
                limit=current_limit,
                offset=offset,
                markets=markets
            )
            
            if not positions:
                break
                
            all_positions.extend(positions)
            offset += len(positions)
            
            # If we got fewer positions than requested, we've reached the end
            if len(positions) < current_limit:
                break
                
            # Small delay to be respectful to the API
            time.sleep(0.1)
        
        return all_positions[:max_positions] if max_positions else all_positions
    
    def process_trades_to_csv_format(self, 
                                     trades: List[Dict], 
                                     positions: List[Dict],
                                     closed_positions: List[Dict] = None,
                                     miner_id: str = "unknown") -> List[Dict]:
        """
        Process trades and positions into the required CSV format
        """
        # Create position lookup by asset
        position_map = {pos['asset']: pos for pos in positions if 'asset' in pos}
        
        # Create closed position lookup by asset (prioritize this for PnL data)
        closed_position_map = {}
        if closed_positions:
            closed_position_map = {pos['asset']: pos for pos in closed_positions if 'asset' in pos}
        
        csv_records = []
        
        for trade in trades:
            # Generate unique trade ID (using hash of key fields)
            trade_id_source = f"{trade.get('transactionHash', '')}{trade.get('timestamp', '')}{trade.get('asset', '')}"
            trade_id = hashlib.md5(trade_id_source.encode()).hexdigest()[:12]
            
            # Get position data if available
            asset = trade.get('asset', '')
            position = position_map.get(asset, {})
            closed_position = closed_position_map.get(asset, {})
            
            # Prioritize closed position data for PnL
            if closed_position:
                # Use closed position data (most accurate for historical PnL)
                is_settled = True  # Closed positions are always settled
                pnl = closed_position.get('realizedPnl', 0)
                avg_price = closed_position.get('avgPrice', 0)
                total_bought = closed_position.get('totalBought', 0)
                date_settled = closed_position.get('endDate', '')
                market_title = closed_position.get('title', trade.get('title', ''))
            elif position:
                # Use current position data
                is_settled = position.get('redeemable', False)
                if is_settled:
                    pnl = position.get('realizedPnl', 0)
                else:
                    pnl = position.get('cashPnl', 0)
                avg_price = position.get('avgPrice', 0)
                total_bought = position.get('totalBought', 0)
                date_settled = position.get('endDate', '')
                market_title = position.get('title', trade.get('title', ''))
            else:
                # No position data - cannot calculate PnL
                is_settled = False
                pnl = 0
                avg_price = 0
                total_bought = 0
                date_settled = ''
                market_title = trade.get('title', '')
            
            # Calculate basic trade metrics
            size = float(trade.get('size', 0))
            price = float(trade.get('price', 0))
            volume = size * price
            
            # Determine if trade was correct (for settled markets)
            is_correct = False
            if is_settled and pnl > 0:
                is_correct = True
            
            # Format timestamps
            timestamp = trade.get('timestamp', 0)
            date_created = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d') if timestamp else ''
            
            # Create CSV record
            csv_record = {
                'trade_id': trade_id,
                'miner_id': miner_id,
                'market_id': trade.get('slug', trade.get('conditionId', 'unknown')),
                'date_created': date_created,
                'volume': round(volume, 2),
                'pnl': round(pnl, 2),
                'is_correct': is_correct,
                'is_settled': is_settled,
                'date_settled': date_settled,
                'trade_type': trade.get('side', '').lower(),
                'price': round(price, 4),
                # Additional fields for analysis
                'outcome': trade.get('outcome', ''),
                'market_title': market_title,
                'transaction_hash': trade.get('transactionHash', ''),
                'avg_price': round(avg_price, 4),
                'total_bought': round(total_bought, 2)
            }
            
            csv_records.append(csv_record)
        
        return csv_records
    
    def write_csv(self, records: List[Dict], output_file: str = 'polymarket_trades.csv'):
        """
        Write records to CSV file
        """
        if not records:
            print("No records to write")
            return
        
        # Define the exact headers you want
        headers = [
            'trade_id', 'miner_id', 'market_id', 'date_created', 
            'volume', 'pnl', 'is_correct', 'is_settled', 
            'date_settled', 'trade_type', 'price'
        ]
        
        # Optional: include additional fields
        extended_headers = headers + ['outcome', 'market_title', 'transaction_hash']
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=extended_headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(records)
        
        print(f"CSV file written: {output_file}")
        print(f"Total records: {len(records)}")
    
    def display_trades_table(self, records: List[Dict], limit: int = 10) -> None:
        """
        Display trades data in a formatted table
        """
        if not records:
            print("No trades to display")
            return
        
        # Limit the number of records to display
        display_records = records[:limit]
        
        # Define the columns to display
        table_columns = [
            'trade_id', 'market_title', 'date_created', 'outcome', 
            'price', 'volume', 'pnl', 'is_correct', 'is_settled'
        ]
        
        # Prepare data for table
        table_data = []
        for record in display_records:
            row = []
            for col in table_columns:
                value = record.get(col, '')
                # Format boolean values
                if col in ['is_correct', 'is_settled']:
                    value = '✓' if value else '✗'
                # Truncate long strings
                elif isinstance(value, str) and len(value) > 20:
                    value = value[:17] + '...'
                row.append(value)
            table_data.append(row)
        
        # Create headers
        headers = [col.replace('_', ' ').title() for col in table_columns]
        
        print(f"\n=== Trades Data (showing first {len(display_records)} of {len(records)} trades) ===")
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        if len(records) > limit:
            print(f"\n... and {len(records) - limit} more trades (use --csv to export all)")
    
    def generate_summary_stats(self, records: List[Dict]) -> Dict:
        """
        Generate summary statistics from the processed records
        """
        if not records:
            return {}
        
        total_trades = len(records)
        settled_trades = sum(1 for r in records if r['is_settled'])
        correct_trades = sum(1 for r in records if r['is_correct'])
        total_volume = sum(r['volume'] for r in records)
        total_pnl = sum(r['pnl'] for r in records)
        
        buy_trades = sum(1 for r in records if r['trade_type'] == 'buy')
        sell_trades = sum(1 for r in records if r['trade_type'] == 'sell')
        
        win_rate = (correct_trades / settled_trades * 100) if settled_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'settled_trades': settled_trades,
            'open_trades': total_trades - settled_trades,
            'correct_trades': correct_trades,
            'win_rate': round(win_rate, 2),
            'total_volume': round(total_volume, 2),
            'total_pnl': round(total_pnl, 2),
            'buy_trades': buy_trades,
            'sell_trades': sell_trades
        }


def main():
    parser = argparse.ArgumentParser(description='Fetch Polymarket trade data and export to CSV')
    parser.add_argument('--user', '-u', 
                       required=True,
                       help='User wallet address (required)')
    parser.add_argument('--miner-id', '-m',
                       default='unknown',
                       help='Miner ID for the CSV records')
    parser.add_argument('--output', '-o',
                       default='polymarket_trades.csv',
                       help='Output CSV filename (default: polymarket_trades.csv)')
    parser.add_argument('--start-date', '-s',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', '-e',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--csv', action='store_true',
                       help='Export data to CSV file')
    parser.add_argument('--summary', action='store_true',
                       help='Print summary statistics')
    parser.add_argument('--limit', '-l', type=int, default=10,
                       help='Limit number of trades to display in table (default: 10)')
    
    args = parser.parse_args()
    
    # Parse dates if provided
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Initialize fetcher
    fetcher = PolymarketTradeFetcher()
    
    print(f"Fetching trades for user: {args.user}")
    if start_date:
        print(f"Start date: {start_date}")
    if end_date:
        print(f"End date: {end_date}")
    
    # Fetch all trades
    trades = fetcher.fetch_all_trades(
        user_address=args.user,
        start_date=start_date,
        end_date=end_date
    )
    
    if not trades:
        print("No trades found for this user")
        return
    
    print(f"Fetched {len(trades)} trades")
    
    # Extract condition IDs from trades for closed positions query
    condition_ids = [trade.get('conditionId') for trade in trades if trade.get('conditionId')]
    print(f"Found {len(condition_ids)} unique condition IDs from trades")
    
    # Fetch closed positions using condition IDs from trades
    print("Fetching closed positions for historical PnL data...")
    closed_positions = fetcher.fetch_all_closed_positions(
        user_address=args.user,
        markets=condition_ids  # No artificial limit - batching handles this
    )
    print(f"Fetched {len(closed_positions)} closed positions")
    
    # Fetch current positions as fallback
    print("Fetching current positions as fallback...")
    positions = fetcher.fetch_all_positions(args.user)
    print(f"Fetched {len(positions)} current positions")
    
    # Process trades into CSV format
    print("Processing trades into CSV format...")
    csv_records = fetcher.process_trades_to_csv_format(
        trades=trades,
        positions=positions,
        closed_positions=closed_positions,
        miner_id=args.miner_id
    )
    
    # Display trades table by default
    fetcher.display_trades_table(csv_records, args.limit)
    
    # Write to CSV only if requested
    if args.csv:
        print("\nExporting to CSV...")
        fetcher.write_csv(csv_records, args.output)
    
    # Print summary if requested
    if args.summary:
        stats = fetcher.generate_summary_stats(csv_records)
        print("\n=== Summary Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    # Show PnL analysis
    settled_trades = [r for r in csv_records if r['is_settled']]
    if settled_trades:
        total_pnl = sum(r['pnl'] for r in settled_trades)
        total_volume = sum(r['volume'] for r in settled_trades)
        winning_trades = sum(1 for r in settled_trades if r['pnl'] > 0)
        losing_trades = sum(1 for r in settled_trades if r['pnl'] < 0)
        
        print(f"\n=== PnL Analysis ===")
        print(f"Settled trades: {len(settled_trades)}")
        print(f"Total volume: ${total_volume:.2f}")
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Winning trades: {winning_trades}")
        print(f"Losing trades: {losing_trades}")
        if settled_trades:
            win_rate = (winning_trades / len(settled_trades)) * 100
            print(f"Win rate: {win_rate:.1f}%")
    else:
        print(f"\n=== PnL Analysis ===")
        print("No settled trades found for PnL analysis")


if __name__ == "__main__":
    main()