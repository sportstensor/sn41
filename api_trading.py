#!/usr/bin/env python3
"""
Almanac API Interactive Trading Client

This script provides an interactive trading session with the Almanac API.
It allows you to:
- Generate Polymarket API credentials
- Initiate trading sessions and place orders
- Fetch positions summary
- Check and claim Polymarket proceeds (Almanac redeem + Polymarket relayer; batch losing cleanup)
- Funds: proxy wallet balance (API), deposit to proxy (USDC.e or Polymarket bridge native USDC/POL), withdraw pUSD via relayer
- Link/unlink Bittensor UID to Almanac account
- Manage multiple credential sets (wallet accounts)

Credential Sets:
Supports multiple wallet accounts via prefixed environment variables (e.g., WALLET1_EOA_WALLET_ADDRESS).
Each set can have its own wallet address, private key, proxy funder, and Polymarket API credentials.
Switch between sets at runtime - sessions are automatically cleared when switching accounts.

Polygon JSON-RPC defaults to https://polygon.drpc.org (Polygon docs mainnet endpoint); set POLYGON_RPC_URL in api_trading.env to override.

Requirements:
- Python 3.10+
- Almanac account (setup at https://almanac.market)
- EOA wallet private key for signing transactions
- Optional: Polymarket API credentials (can be generated via this script)

Python dependencies:
- requests
- dotenv
- tabulate
- py-clob-client-v2
- eth-account
- bittensor
- web3
- py-builder-relayer-client

pip install -r requirements-trading.txt
"""

import os
import json
import math
from dataclasses import dataclass
from pathlib import Path
import requests
from dotenv import load_dotenv
from py_clob_client_v2 import ClobClient, OrderArgs, MarketOrderArgs, OrderType, PartialCreateOrderOptions
from py_clob_client_v2.order_builder.constants import BUY, SELL
from eth_account import Account
from eth_account.messages import encode_defunct
import time
import secrets
import bittensor as bt
from datetime import datetime
from tabulate import tabulate
from constants import VOLUME_FEE, PRICE_BUFFER_ADJUSTMENT
from web3 import Web3
from requests.compat import json as requests_json
from py_builder_relayer_client.client import RelayClient
from py_builder_relayer_client.models import OperationType, SafeTransaction
from py_builder_relayer_client.exceptions import RelayerClientException

ALMANAC_API_URL = "https://api.almanac.market/api"
#ALMANAC_API_URL = "http://localhost:3001/api"
POLYMARKET_CLOB_HOST = "https://clob.polymarket.com"
POLYGON_CHAIN_ID = 137
# Polymarket data-api: redeemable positions (aligned with Almanac ClaimProceedsModal)
POLY_POSITIONS_WINNERS_URL = (
    "https://data-api.polymarket.com/positions"
    "?user={proxy_lc}&sizeThreshold=.1&redeemable=true&limit=100&offset=0&sortBy=CASHPNL"
)
POLY_POSITIONS_MIXED_URL = (
    "https://data-api.polymarket.com/positions"
    "?user={proxy_lc}&sizeThreshold=.1&redeemable=true&limit=50"
)
MAX_TOTAL_REDEMPTIONS = 20
MAX_LOSING_POSITIONS_TO_CLOSE = 15
VALUE_WINNER_MIN = 0.01
VALUE_LOSER_MAX = 0.01
SIZE_MIN_TOKENS = 0.1

# Polygon proxy/Safe settlement token (Polymarket v2)
PROXY_TOKEN_SYMBOL = "pUSD"
PROXY_TOKEN_POLYGON = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"
# Polygon USDC.e (bridged)
USDC_E_POLYGON = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
NATIVE_USDC_POLYGON = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"
# CLOB V2 collateral wrap/unwrap contracts
# wrap(asset, to, amount): USDC/USDC.e -> pUSD
# unwrap(asset, to, amount): pUSD -> USDC/USDC.e
COLLATERAL_ONRAMP_ADDRESS = "0x93070a847efEf7F70739046A929D47a521F5B8ee"
COLLATERAL_OFFRAMP_ADDRESS = "0x2957922Eb93258b93368531d39fAcCA3B4dC5854"
POLYMARKET_BRIDGE_DEPOSIT_URL = "https://bridge.polymarket.com/deposit"
POLYMARKET_RELAYER_URL = "https://relayer-v2.polymarket.com"
# https://docs.polygon.technology/pos/reference/rpc-endpoints/ (mainnet table + public RPCs)
DEFAULT_POLYGON_RPC_URL = "https://polygon.drpc.org"
USDC_E_DECIMALS = 6
PROXY_TOKEN_DECIMALS = 6
AUTO_WRAP_USDCE_PROCEEDS = True
UINT256_MAX = (1 << 256) - 1

ERC20_MIN_ABI = [
    {
        "type": "function",
        "name": "balanceOf",
        "stateMutability": "view",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"type": "uint256"}],
    },
    {
        "type": "function",
        "name": "transfer",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"type": "bool"}],
    },
    {
        "type": "function",
        "name": "decimals",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"type": "uint8"}],
    },
    {
        "type": "function",
        "name": "allowance",
        "stateMutability": "view",
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "outputs": [{"name": "remaining", "type": "uint256"}],
    },
    {
        "type": "function",
        "name": "approve",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"type": "bool"}],
    },
]

COLLATERAL_ONRAMP_WRAP_ABI = [
    {
        "type": "function",
        "name": "wrap",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "_asset", "type": "address"},
            {"name": "_to", "type": "address"},
            {"name": "_amount", "type": "uint256"},
        ],
        "outputs": [],
    }
]

# EIP-712 domain contract for Polymarket CTF Exchange -- V2
EIP712_DOMAIN_CONTRACT = "0xE111180000d2663C0091e4f400237545B87B996B"
EIP712_DOMAIN_NEGRISK_CONTRACT = "0xe2222d279d744050d28e00520010520000310F59"
# EIP-712 domain contract for Polymarket CTF Exchange -- V1 (DEPRECATED)
#EIP712_DOMAIN_CONTRACT = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
#EIP712_DOMAIN_NEGRISK_CONTRACT = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
ENV_PATH = Path("api_trading.env")

# Default number of positions to fetch per page
DEFAULT_POSITIONS_LIMIT = 25

# Debug mode: if True, injects a static non-eligible market into search results for testing
DEBUG = False

# Static non-eligible market for testing (will be rejected by backend API)
DEBUG_STATIC_MARKET = {
    "id": "680904",
    "question": "Will Bill Ackman say \"Communist\" or \"Communism\" during the X Space event on November 18?",
    "conditionId": "0xaaac5595aecf8ba003fdb425c1697e9ac2e528aae492c052b781486c453e5ffe",
    "slug": "will-bill-ackman-say-communist-or-communism-during-the-x-space-event-on-november-18",
    "title": "Will Bill Ackman say \"Communist\" or \"Communism\" during the X Space event on November 18?",
    "outcomes": ["Yes", "No"],
    "outcome_prices": [0.155, 0.845],
    "clob_token_ids": [
        "39362723615320203601565062388169914485014370896130401193658074341957733150044",
        "104010206690696361697404185514804406127715733374741772559130541144838358362611"
    ],
    "active": True,
    "closed": False,
    "restricted": True,
}


CURRENT_SESSION = None
SELECTED_MARKET = None
SELECTED_CREDENTIAL_SET = None  # Stores the name of the selected credential set (None = default)
CREDENTIAL_SETS = {}  # Dictionary of available credential sets

def _detect_credential_sets():
    """
    Scan the .env file for credential sets.
    Supports both default (no prefix) and named sets (with prefix like WALLET1_, WALLET2_, etc.)
    
    Returns:
        dict: Dictionary mapping credential set names to their credential dicts
    """
    credential_sets = {}
    
    if not ENV_PATH.exists():
        return credential_sets
    
    # Load the .env file to get all variables
    load_dotenv(dotenv_path=str(ENV_PATH), override=True)
    
    # Required credential keys (must have values)
    required_keys = [
        "EOA_WALLET_ADDRESS",
        "EOA_WALLET_PK",
        "EOA_PROXY_FUNDER"
    ]
    
    # Optional credential keys (must exist but can be empty)
    optional_keys = [
        "POLYMARKET_API_KEY",
        "POLYMARKET_API_SECRET",
        "POLYMARKET_API_PASSPHRASE"
    ]
    
    # All credential keys (for detection purposes)
    all_credential_keys = required_keys + optional_keys
    
    # Read the .env file directly to detect prefixes and values
    try:
        env_vars = {}  # Store all env vars from file
        
        with open(ENV_PATH, 'r') as f:
            lines = f.readlines()
        
        # First pass: read all variables from file
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '=' not in line:
                continue
            
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")  # Remove quotes if present
            env_vars[key] = value
        
        # Track which prefixes we've seen
        seen_prefixes = set()
        
        # Second pass: detect prefixes
        for key in env_vars.keys():
            # Check if this key matches any credential key (with or without prefix)
            for cred_key in all_credential_keys:
                if key == cred_key:
                    # Default credential set (no prefix)
                    seen_prefixes.add("")
                elif key.endswith(f"_{cred_key}"):
                    # Named credential set (has prefix)
                    prefix = key[:-len(f"_{cred_key}")]
                    seen_prefixes.add(prefix)
        
        # Third pass: collect credentials for each prefix
        for prefix in seen_prefixes:
            creds = {}
            all_required_present = True
            
            # Check required keys (must have values)
            for req_key in required_keys:
                if prefix:
                    env_key = f"{prefix}_{req_key}"
                else:
                    env_key = req_key
                
                value = env_vars.get(env_key, "").strip()
                if value:
                    creds[req_key] = value
                else:
                    all_required_present = False
                    break
            
            # Only proceed if all required keys are present with values
            if not all_required_present:
                continue
            
            # Add optional keys (can be empty)
            for opt_key in optional_keys:
                if prefix:
                    env_key = f"{prefix}_{opt_key}"
                else:
                    env_key = opt_key
                
                value = env_vars.get(env_key, "").strip()
                # Store even if empty (allows for credential generation later)
                creds[opt_key] = value
            
            # Add the credential set
            set_name = prefix if prefix else "default"
            credential_sets[set_name] = creds
        
    except Exception as exc:
        print(f"Warning: Could not detect credential sets: {exc}")
    
    return credential_sets

def _get_credential(key: str) -> str | None:
    """
    Get a credential value, checking the selected credential set first,
    then falling back to default environment variables.
    
    Args:
        key: The credential key (e.g., "EOA_WALLET_ADDRESS")
    
    Returns:
        The credential value or None if not found
    """
    # If a credential set is selected, use it
    if SELECTED_CREDENTIAL_SET and SELECTED_CREDENTIAL_SET in CREDENTIAL_SETS:
        creds = CREDENTIAL_SETS[SELECTED_CREDENTIAL_SET]
        if key in creds:
            return creds[key]
    
    # Fallback to default credentials (no prefix)
    if "default" in CREDENTIAL_SETS and key in CREDENTIAL_SETS["default"]:
        return CREDENTIAL_SETS["default"][key]
    
    # Final fallback to environment variables
    return os.getenv(key)

def select_credential_set():
    """
    Allow user to select which credential set to use.
    Clears the current trading session when switching accounts.
    """
    global SELECTED_CREDENTIAL_SET, CREDENTIAL_SETS, CURRENT_SESSION, SELECTED_MARKET
    
    # Refresh credential sets
    CREDENTIAL_SETS = _detect_credential_sets()
    
    if not CREDENTIAL_SETS:
        print("\nNo credential sets found in the environment file.")
        print("Please configure credentials in api_trading.env")
        # Clear selection and session if no sets available
        SELECTED_CREDENTIAL_SET = None
        CURRENT_SESSION = None
        SELECTED_MARKET = None
        return
    
    # Validate that the currently selected set still exists
    if SELECTED_CREDENTIAL_SET and SELECTED_CREDENTIAL_SET not in CREDENTIAL_SETS:
        print(f"\n⚠ Previously selected credential set '{SELECTED_CREDENTIAL_SET}' no longer exists.")
        print("Clearing session and resetting selection...")
        SELECTED_CREDENTIAL_SET = None
        CURRENT_SESSION = None
        SELECTED_MARKET = None
    
    print("\nAvailable credential sets:")
    sets_list = sorted(CREDENTIAL_SETS.keys())
    for idx, set_name in enumerate(sets_list, start=1):
        marker = " (current)" if set_name == SELECTED_CREDENTIAL_SET else ""
        print(f"  {idx}) {set_name}{marker}")
    
    print(f"  {len(sets_list) + 1}) Cancel")
    
    choice = input(f"\nSelect credential set (1-{len(sets_list) + 1}): ").strip()
    
    try:
        choice_num = int(choice)
        if 1 <= choice_num <= len(sets_list):
            new_set = sets_list[choice_num - 1]
            
            # If switching to a different credential set, clear the session
            if new_set != SELECTED_CREDENTIAL_SET:
                if CURRENT_SESSION:
                    print("\n⚠ Clearing existing trading session (switching accounts)...")
                    CURRENT_SESSION = None
                    SELECTED_MARKET = None
            
            SELECTED_CREDENTIAL_SET = new_set
            print(f"\n✓ Selected credential set: {SELECTED_CREDENTIAL_SET}")
            # Show wallet address for confirmation
            wallet = _get_credential("EOA_WALLET_ADDRESS")
            if wallet:
                print(f"  Wallet Address: {wallet}")
        elif choice_num == len(sets_list) + 1:
            print("Cancelled.")
        else:
            print("Invalid choice.")
    except ValueError:
        print("Invalid input. Please enter a number.")

def _format_price(value):
    try:
        f = float(value)
        # Clip to [0,1] range for probabilities if out-of-range but close
        if -0.05 <= f <= 1.05:
            f = min(max(f, 0.0), 1.0)
        return f"{f:.2f}"
    except Exception:
        return "-"

def _format_game_start_time(game_start_time):
    """
    Format game start time to display like "Dec 5, 2025 7p EST"
    
    Args:
        game_start_time: ISO format datetime string or timestamp
        
    Returns:
        Formatted string like "Dec 5, 2025 7p EST" or None if parsing fails
    """
    if not game_start_time:
        return None
    
    try:
        # Try parsing as ISO format string
        if isinstance(game_start_time, str):
            # Handle ISO format with or without timezone
            if 'T' in game_start_time:
                dt = datetime.fromisoformat(game_start_time.replace('Z', '+00:00'))
            else:
                # Try timestamp
                dt = datetime.fromtimestamp(int(game_start_time))
        elif isinstance(game_start_time, (int, float)):
            # Assume timestamp
            dt = datetime.fromtimestamp(game_start_time)
        else:
            return None
        
        # Format: "Dec 5, 2025 7p EST"
        # Get month abbreviation, day, year
        month_abbr = dt.strftime("%b")
        day = dt.day
        year = dt.year
        
        # Format hour (12-hour format, no leading zero)
        hour = dt.hour
        if hour == 0:
            hour_str = "12"
            period = "a"
        elif hour < 12:
            hour_str = str(hour)
            period = "a"
        elif hour == 12:
            hour_str = "12"
            period = "p"
        else:
            hour_str = str(hour - 12)
            period = "p"
        
        # Get timezone abbreviation (try to get EST/EDT, etc.)
        # For simplicity, we'll use the timezone offset or default to EST
        tz_str = "EST"  # Default
        try:
            if dt.tzinfo:
                offset = dt.utcoffset().total_seconds() / 3600
                if offset == -5:
                    tz_str = "EST"
                elif offset == -4:
                    tz_str = "EDT"
                elif offset == -8:
                    tz_str = "PST"
                elif offset == -7:
                    tz_str = "PDT"
                else:
                    tz_str = dt.strftime("%Z") or "EST"
        except Exception:
            pass
        
        return f"{month_abbr} {day}, {year} {hour_str}{period} {tz_str}"
    except Exception:
        return None

def _extract_outcomes_summary(market: dict) -> str:
    """
    Try to extract a concise outcomes summary like:
    'Yes 0.41 | No 0.59'  or  'A 0.10 | B 0.20 | C 0.70'
    Supports a few common shapes defensively.
    """
    # Preferred: 'outcomes' list with aligned 'outcome_prices' list
    outcomes = market.get("outcomes")
    outcome_prices = market.get("outcome_prices")
    if isinstance(outcomes, list) and isinstance(outcome_prices, list) and len(outcomes) == len(outcome_prices):
        parts = []
        for name, price in zip(outcomes, outcome_prices):
            name_str = name if isinstance(name, str) else (name.get("name") if isinstance(name, dict) else str(name))
            parts.append(f"{name_str} {_format_price(price)}")
        if parts:
            return " | ".join(parts[:6])
    

    return ""  # no concise summary available

def _normalize_search_results(payload) -> list:
    """
    Accepts either a list of markets or common envelope shapes and returns a list.
    Handles: {results: [...]}, {data: [...]}, {markets: [...]}, {items: [...]}
    Falls back to [] if nothing recognized.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "data", "markets", "items"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
        # Some APIs wrap under {success, data: {...}} with inner list keys
        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("results", "markets", "items", "list"):
                val = data.get(key)
                if isinstance(val, list):
                    return val
    return []

def _extract_event_list(payload_list: list) -> list:
    """
    From a generic list returned by search, prefer items that look like Events (contain 'markets').
    If none contain 'markets', return the original list and treat items as markets directly.
    """
    if not isinstance(payload_list, list):
        return []
    has_event_shape = any(isinstance(it, dict) and isinstance(it.get("markets"), list) for it in payload_list)
    return payload_list if has_event_shape else payload_list

def _display_markets_for_event(event: dict) -> list:
    """
    Print markets for a single event with concise outcomes summary in a table format.
    Returns the list of markets for further selection.
    """
    markets = event.get("markets") or []
    if not isinstance(markets, list) or not markets:
        print("No markets found for this event.")
        return []
    # Sort markets: Moneyline first, then by volume (big to small)
    def _sort_key(m):
        # Check if it's a Moneyline market
        sports_market_type = m.get("sports_market_type") or m.get("sportsMarketType") or ""
        is_moneyline = sports_market_type.lower() == "moneyline"
        
        # Get volume (try multiple field names)
        volume = m.get("volume") or m.get("totalVolume") or m.get("total_volume") or 0
        try:
            volume = float(volume)
        except (ValueError, TypeError):
            volume = 0.0
        
        # Return tuple: (is_moneyline, -volume)
        # is_moneyline: True=0 (sorts first), False=1 (sorts later)
        # -volume: negative so bigger volumes sort first
        return (0 if is_moneyline else 1, -volume)
    
    markets = sorted(markets, key=_sort_key)
    
    # Find maximum number of outcomes across all markets to determine column count
    max_outcomes = 0
    for m in markets:
        outcomes = m.get("outcomes") or []
        if isinstance(outcomes, list):
            max_outcomes = max(max_outcomes, len(outcomes))
    
    # Build table rows
    table_rows = []
    for idx, m in enumerate(markets, start=1):
        title = m.get("title") or m.get("question") or m.get("name") or "Untitled"
        market_id = m.get("id") or m.get("marketId") or m.get("_id") or "unknown"
        
        # Extract outcomes and prices for this market
        outcomes = m.get("outcomes") or []
        outcome_prices = m.get("outcome_prices") or []
        
        # Build row: #, Market, then each outcome (name + price), then Market ID
        row = [
            str(idx),
            _truncate_text(title, 50)
        ]
        
        # Add outcome name and price for each outcome column
        if isinstance(outcomes, list) and isinstance(outcome_prices, list) and len(outcomes) == len(outcome_prices):
            for outcome, price in zip(outcomes, outcome_prices):
                outcome_name = outcome if isinstance(outcome, str) else (outcome.get("name") if isinstance(outcome, dict) else str(outcome))
                formatted_price = _format_price(price)
                row.append(f"{outcome_name} {formatted_price}")
        else:
            # Fallback: try to extract outcomes from other formats
            for i in range(len(outcomes) if isinstance(outcomes, list) else 0):
                outcome = outcomes[i]
                outcome_name = outcome if isinstance(outcome, str) else (outcome.get("name") if isinstance(outcome, dict) else str(outcome))
                price = outcome_prices[i] if isinstance(outcome_prices, list) and i < len(outcome_prices) else None
                if price is not None:
                    formatted_price = _format_price(price)
                    row.append(f"{outcome_name} {formatted_price}")
                else:
                    row.append(outcome_name if outcome_name else "-")
        
        # Pad with "-" if market has fewer outcomes than max
        while len(row) < 2 + max_outcomes:
            row.append("-")
        
        # Add Volume column (round to nearest dollar, no dollar sign)
        volume = m.get("volume") or m.get("totalVolume") or m.get("total_volume") or 0
        try:
            volume_float = float(volume)
            volume_rounded = round(volume_float)
            row.append(str(volume_rounded))
        except (ValueError, TypeError):
            row.append("-")
        
        # Add Liquidity column (round to nearest dollar, no dollar sign)
        liquidity = m.get("liquidity") or m.get("totalLiquidity") or m.get("total_liquidity") or 0
        try:
            liquidity_float = float(liquidity)
            liquidity_rounded = round(liquidity_float)
            row.append(str(liquidity_rounded))
        except (ValueError, TypeError):
            row.append("-")
        
        # Add Type column (sports_market_type if it exists, replace _ with spaces)
        sports_market_type = m.get("sports_market_type") or m.get("sportsMarketType") or "-"
        if sports_market_type != "-":
            sports_market_type = sports_market_type.replace("_", " ")
        row.append(sports_market_type)
        
        # Add Market ID as last column
        row.append(market_id)
        
        table_rows.append(row)
    
    # Define table headers: #, Market, then Outcome 1, Outcome 2, etc., then Volume, Liquidity, Type, then Market ID
    outcome_headers = [f"Outcome {i+1}" for i in range(max_outcomes)]
    headers = ["#", "Market"] + outcome_headers + ["Volume", "Liquidity", "Type", "Market ID"]
    
    # Print table
    print(f"\nMarkets ({len(markets)} found):")
    print(tabulate(table_rows, headers=headers, tablefmt="grid", stralign="left"))
    
    return markets

def fetch_clob_prices(token_ids: list) -> dict | None:
    """
    Fetch latest prices from the CLOB API for given token IDs.
    
    Args:
        token_ids: List of token ID strings
        
    Returns:
        Dictionary mapping token IDs to price data:
        {
            "token_id_1": {"BUY": "0.45", "SELL": "0.44"},
            "token_id_2": {"BUY": "0.52", "SELL": "0.51"}
        }
        Returns None if the request fails.
    """
    if not token_ids:
        return None
    
    try:
        # Build BookParams array with both BUY and SELL side for each token
        book_params = []
        for token_id in token_ids:
            book_params.append({"token_id": token_id, "side": "BUY"})
            book_params.append({"token_id": token_id, "side": "SELL"})
        
        # Make POST request to CLOB API
        response = requests.post(
            f"{POLYMARKET_CLOB_HOST}/prices",
            json=book_params,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"Warning: Failed to fetch CLOB prices (status {response.status_code})")
            return None
        
        prices_data = response.json()
        return prices_data
    except Exception as exc:
        print(f"Warning: Error fetching CLOB prices: {exc}")
        return None

def _update_all_markets_prices_from_clob(markets: list) -> list:
    """
    Batch fetch latest prices from CLOB API for all markets and update their outcome_prices.
    Stores original prices in _original_outcome_prices for fallback.
    
    Args:
        markets: List of market dictionaries with clob_token_ids
        
    Returns:
        List of updated market dictionaries with fresh outcome_prices
    """
    if not isinstance(markets, list) or not markets:
        return markets
    
    # Collect all token IDs from all markets
    all_token_ids = []
    market_token_map = {}  # Map token_id -> list of (market_idx, outcome_idx) tuples
    
    for market_idx, market in enumerate(markets):
        clob_token_ids = market.get("clob_token_ids")
        if isinstance(clob_token_ids, list) and clob_token_ids:
            # Store original prices for fallback
            original_prices = market.get("outcome_prices")
            if original_prices:
                market["_original_outcome_prices"] = original_prices.copy() if isinstance(original_prices, list) else original_prices
            
            # Track which market/outcome each token belongs to
            for outcome_idx, token_id in enumerate(clob_token_ids):
                token_id_str = str(token_id)
                if token_id_str not in market_token_map:
                    market_token_map[token_id_str] = []
                    all_token_ids.append(token_id)
                market_token_map[token_id_str].append((market_idx, outcome_idx))
    
    # If no token IDs found, return markets unchanged
    if not all_token_ids:
        return markets
    
    # Batch fetch all prices from CLOB API
    prices_data = fetch_clob_prices(all_token_ids)
    if not prices_data:
        return markets
    
    # Update all markets with fetched prices
    for market_idx, market in enumerate(markets):
        clob_token_ids = market.get("clob_token_ids")
        if not isinstance(clob_token_ids, list) or not clob_token_ids:
            continue
        
        updated_prices = []
        for token_id in clob_token_ids:
            # Try multiple formats for token ID lookup
            token_prices = None
            token_id_str = str(token_id)
            if token_id_str in prices_data:
                token_prices = prices_data[token_id_str]
            else:
                try:
                    token_id_num = int(token_id)
                    if str(token_id_num) in prices_data:
                        token_prices = prices_data[str(token_id_num)]
                except (ValueError, TypeError):
                    pass
            
            if token_prices:
                # Use mid price (average of BUY and SELL) or BUY price as fallback
                buy_price = token_prices.get("BUY")
                sell_price = token_prices.get("SELL")
                if buy_price and sell_price:
                    try:
                        buy_float = float(buy_price)
                        sell_float = float(sell_price)
                        mid_price = (buy_float + sell_float) / 2.0
                        updated_prices.append(mid_price)
                    except (ValueError, TypeError):
                        try:
                            updated_prices.append(float(buy_price))
                        except (ValueError, TypeError):
                            updated_prices.append(None)
                elif buy_price:
                    try:
                        updated_prices.append(float(buy_price))
                    except (ValueError, TypeError):
                        updated_prices.append(None)
                else:
                    updated_prices.append(None)
            else:
                updated_prices.append(None)
        
        # Update market with fresh prices (only if we got valid prices)
        if updated_prices and any(p is not None for p in updated_prices):
            market["outcome_prices"] = updated_prices
            # Store raw CLOB price data for this market's tokens
            market_clob_prices = {}
            for token_id in clob_token_ids:
                token_id_str = str(token_id)
                if token_id_str in prices_data:
                    market_clob_prices[token_id_str] = prices_data[token_id_str]
                else:
                    try:
                        token_id_num = int(token_id)
                        if str(token_id_num) in prices_data:
                            market_clob_prices[str(token_id_num)] = prices_data[str(token_id_num)]
                    except (ValueError, TypeError):
                        pass
            if market_clob_prices:
                market["_clob_prices"] = market_clob_prices
    
    return markets

def _update_market_prices_from_clob(market: dict) -> dict:
    """
    Fetch latest prices from CLOB API and update the market's outcome_prices.
    Stores original prices in _original_outcome_prices for comparison.
    This is a single-market version - use _update_all_markets_prices_from_clob for batch updates.
    
    Args:
        market: Market dictionary with clob_token_ids
        
    Returns:
        Updated market dictionary with fresh outcome_prices
    """
    # Use the batch function for single market
    updated_markets = _update_all_markets_prices_from_clob([market])
    return updated_markets[0] if updated_markets else market

def _display_outcomes_and_choose(market: dict):
    """
    Show detailed outcomes for a market (if available) and let the user pick one.
    Uses CLOB prices if available, falls back to original prices if CLOB fetch failed.
    Returns a tuple (chosen_outcome_name, chosen_outcome_price, chosen_token_id).
    """
    outcomes = market.get("outcomes")
    outcome_prices = market.get("outcome_prices")  # These should be CLOB prices if fetched successfully
    original_prices = market.get("_original_outcome_prices")  # Original prices from API (fallback)
    clob_token_ids = market.get("clob_token_ids")
    clob_prices = market.get("_clob_prices")  # Raw CLOB price data

    # Check if CLOB fetch was successful
    clob_fetch_successful = clob_prices is not None

    # Normalize into list of dicts {name, price?, tokenId?} with index alignment
    normalized = []
    if isinstance(outcomes, list):
        for idx, o in enumerate(outcomes):
            name = o if isinstance(o, str) else (o.get("name") if isinstance(o, dict) else str(o))
            price = None
            token_id = None
            
            # Get current price (prefer CLOB prices if fetch was successful)
            if clob_fetch_successful and isinstance(outcome_prices, list) and idx < len(outcome_prices):
                price = outcome_prices[idx]
            
            # Fallback to original prices if CLOB fetch failed
            if price is None and isinstance(original_prices, list) and idx < len(original_prices):
                price = original_prices[idx]
            
            if isinstance(clob_token_ids, list) and idx < len(clob_token_ids):
                token_id = clob_token_ids[idx]
            
            # Fallback to other price sources if still no price
            if price is None and isinstance(o, dict):
                price = (
                    o.get("price")
                    or o.get("lastPrice")
                    or o.get("midPrice")
                    or o.get("probability")
                    or o.get("p")
                )
            
            normalized.append({"name": name or "?", "price": price, "tokenId": token_id})
    elif isinstance(outcomes, dict):
        # If dict, best-effort alignment by iteration order
        for idx, (name, maybe_price) in enumerate(outcomes.items()):
            price = maybe_price
            token_id = None
            
            # Get current price (prefer CLOB prices if fetch was successful)
            if clob_fetch_successful and isinstance(outcome_prices, list) and idx < len(outcome_prices):
                price = outcome_prices[idx]
            
            # Fallback to original prices if CLOB fetch failed
            if price is None and isinstance(original_prices, list) and idx < len(original_prices):
                price = original_prices[idx]
            
            if isinstance(clob_token_ids, list) and idx < len(clob_token_ids):
                token_id = clob_token_ids[idx]
            
            # If we still don't have a price, use maybe_price from dict
            if price is None:
                price = maybe_price
            
            normalized.append({"name": str(name), "price": price, "tokenId": token_id})
    else:
        # Fallback binary representation
        yes_price = market.get("yesPrice") or market.get("yes")
        no_price = market.get("noPrice") or market.get("no")
        
        # Use CLOB prices if fetch was successful
        if clob_fetch_successful and isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
            yes_price = outcome_prices[0]
            no_price = outcome_prices[1]
        # Fallback to original prices if CLOB fetch failed
        elif isinstance(original_prices, list) and len(original_prices) >= 2:
            yes_price = original_prices[0]
            no_price = original_prices[1]
        
        if yes_price is not None or no_price is not None:
            normalized = [
                {"name": "Yes", "price": yes_price, "tokenId": None},
                {"name": "No", "price": no_price, "tokenId": None},
            ]

    if not normalized:
        print("No explicit outcomes provided by API; proceeding without outcome selection.")
        return (None, None, None)

    print("\nOutcomes:")
    for idx, o in enumerate(normalized, start=1):
        price_str = _format_price(o.get('price'))
        print(f"  {idx}) {o['name']} {price_str}")
    sel = input("\nSelect outcome to trade (or Enter to cancel): ").strip()
    if not sel:
        return None  # Signal cancellation
    try:
        sel_idx = int(sel)
    except ValueError:
        print("Invalid selection; cancelling.")
        return None  # Signal cancellation
    if sel_idx < 1 or sel_idx > len(normalized):
        print("Selection out of range; cancelling.")
        return None  # Signal cancellation
    chosen = normalized[sel_idx - 1]
    return (chosen.get("name"), chosen.get("price"), chosen.get("tokenId"))

def _place_order_now(market: dict, chosen_outcome_name: str | None = None, chosen_token_id: str | None = None, available_shares: float | None = None):
    """
    Inline order placement flow; prompts for side/size/price, shows summary, and submits with confirmation.
    User can type 'c' at any prompt to cancel.
    
    Args:
        market: Market dictionary
        chosen_outcome_name: Pre-selected outcome name
        chosen_token_id: Pre-selected token ID
        available_shares: Available shares for sell orders (to validate against)
    """
    global CURRENT_SESSION
    if not CURRENT_SESSION:
        print("No active trading session. Create a session first in the Trading Menu.")
        return
    market_id = market.get("id") or market.get("marketId") or market.get("_id")
    if not market_id:
        print("Selected market missing id.")
        return
    
    market_title = market.get("title") or market.get("question") or market.get("name") or "Unknown Market"
    # Extract neg_risk field (supports both snake_case and camelCase)
    neg_risk = market.get("neg_risk") or market.get("negRisk") or False
    if isinstance(neg_risk, str):
        neg_risk = neg_risk.lower() in ("true", "1", "yes")
    neg_risk = bool(neg_risk)
    
    print("\nPlace order (type 'c' or 'cancel' at any prompt to cancel):")
    
    # Side input with cancel option
    side_input = input("Side (buy/b/sell/s/cancel/c) [buy]: ").strip().lower()
    if side_input in ("c", "cancel"):
        print("Order cancelled.")
        return
    side = side_input or "buy"
    # Map aliases to full names
    if side in ("b", "buy"):
        side = "buy"
    elif side in ("s", "sell"):
        side = "sell"
    else:
        print("Side must be 'buy', 'b', 'sell', or 's'.")
        return
    
    # Size and price input with cancel option
    # For sell orders: check available shares, no $5 minimum
    # For buy orders: check $5 minimum
    while True:
        size_prompt = f"Size (quantity/cancel/c) [{available_shares:.2f}]: " if (side == "sell" and available_shares) else "Size (quantity/cancel/c) [1]: "
        size_str = input(size_prompt).strip()
        if size_str.lower() in ("c", "cancel"):
            print("Order cancelled.")
            return
        size_str = size_str or (f"{available_shares:.2f}" if (side == "sell" and available_shares) else "1")
        
        price_str = input("Price (0-1/cancel/c) [0.01]: ").strip()
        if price_str.lower() in ("c", "cancel"):
            print("Order cancelled.")
            return
        price_str = price_str or "0.01"
        
        try:
            size = float(size_str)
            price = float(price_str)
        except ValueError:
            print("Invalid size or price. Try again.")
            continue
        if size <= 0 or price <= 0 or price > 1:
            print("Size must be > 0 and price must be in (0, 1]. Try again.")
            continue
        
        # For sell orders: cap at exact position size from API (never round available_shares for the order)
        if side == "sell" and available_shares is not None:
            raw_available = float(available_shares)
            requested = size
            size = min(size, raw_available)
            if requested > raw_available and round(requested, 2) != round(raw_available, 2):
                print(
                    f"Sell size capped to your position: {raw_available:.2f} shares "
                    f"(requested {requested:.2f})."
                )
        
        # For buy orders: check $5 minimum
        if side == "buy":
            notional = size * price
            if notional < 5.0:
                print(f"Order notional ${notional:.2f} is below the $5 minimum. Please increase size and/or price.")
                continue
        break
    
    # SELL: CLOB EIP-712 uses round(size, 2) in micro-units; floor to 0.01-share grid so we never exceed balance
    if side == "sell":
        size = _normalize_sell_size_for_clob(size)
        if size <= 0:
            print("Sell size is below 0.01 shares after aligning to the exchange grid. Nothing to sell.")
            return
    
    # Order type input with cancel option
    print("\nOrder Type:")
    print("  GTC - Good Till Canceled: Limit order remains active until filled or cancelled")
    print("  FOK - Fill Or Kill: Market Order must be filled immediately or it's cancelled")
    print("  FAK - Fill And Kill: Market Order will be filled immediately with what is available and the rest cancelled")
    order_type_input = input("Order type (gtc/fok/fak/cancel/c) [gtc]: ").strip().upper()
    if order_type_input in ("C", "CANCEL"):
        print("Order cancelled.")
        return
    order_type = order_type_input or "GTC"
    if order_type not in ("GTC", "FOK", "FAK"):
        print("Order type must be 'GTC', 'FOK', or 'FAK'. Using default 'GTC'.")
        order_type = "GTC"
    
    side_upper = "BUY" if side == "buy" else "SELL"
    
    # Apply the same price adjustment logic that place_order() uses. Only for Market Orders (FOK, FAK).
    # This ensures the order summary shows the actual values that will be sent to the API
    if order_type in ("FOK", "FAK"):
        if side_upper == "BUY":
            adjusted_price = price + PRICE_BUFFER_ADJUSTMENT
            price_note = f"(+${PRICE_BUFFER_ADJUSTMENT} buffer for {side_upper} orders)"
        else:  # SELL
            adjusted_price = max(0.01, price - PRICE_BUFFER_ADJUSTMENT)
            price_note = f"(-${PRICE_BUFFER_ADJUSTMENT} buffer for {side_upper} orders)"
    else:
        adjusted_price = price
        price_note = ""
    
    # Calculate with adjusted price (what will actually be sent to API)
    notional = size * adjusted_price
    if side_upper == "BUY":
        fee = notional * VOLUME_FEE
        total_with_fee = notional + fee
    else:
        total_with_fee = notional

    # Display summary with actual values
    print("\n" + "="*60)
    print("Order Summary:")
    print("="*60)
    print(f"Market: {market_title}")
    if chosen_outcome_name:
        print(f"Outcome: {chosen_outcome_name}")
    print(f"Neg Risk: {neg_risk}")
    print(f"Side: {side_upper}")
    print(f"Order Type: {order_type}")
    print(f"Size: {size}")
    if adjusted_price != price:
        print(f"Requested Price: {price}")
        print(f"Adjusted Price: {adjusted_price} {price_note}")
    else:
        print(f"Price: {price}")
    print(f"Subtotal: ${notional:.2f}")
    if side_upper == "BUY":
        print(f"Platform Fee ({VOLUME_FEE*100:.1f}%): ${fee:.2f}")
    print(f"Total: ${total_with_fee:.2f}")
    print("="*60)
    
    # Final confirmation
    confirm = input("\nSubmit this order? (y/n): ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Order cancelled.")
        return
    
    # Call unified order poster
    place_order(
        market_id=market_id,
        side_upper=side_upper,
        size=size,
        price=price,
        neg_risk=neg_risk,
        order_type=order_type,
        chosen_outcome_name=chosen_outcome_name,
        chosen_token_id=chosen_token_id,
    )

def fetch_positions(filter_type: str = "all", limit: int = 100, offset: int = 0):
    """
    Fetch user's positions from the Almanac API.
    
    Args:
        filter_type: 'all', 'live', or 'closed'
        limit: Maximum number of positions to return (1-100)
        offset: Number of positions to skip
    
    Returns:
        Response data with positions or None if error
    """
    global CURRENT_SESSION
    if not CURRENT_SESSION:
        print("No active trading session. Create a session first.")
        return None
    
    session_id = CURRENT_SESSION.get("data").get("sessionId")
    wallet_address = (
        CURRENT_SESSION.get("data").get("walletAddress")
        or _get_credential("EOA_WALLET_ADDRESS")
    )
    
    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["x-session-id"] = session_id
    if wallet_address:
        headers["x-wallet-address"] = wallet_address
    
    params = {
        "filter": filter_type,
        "limit": limit,
        "offset": offset
    }
    
    try:
        response = requests.get(
            f"{ALMANAC_API_URL}/v1/trading/positions",
            headers=headers,
            params=params,
            timeout=30
        )
        
        if response.status_code != 200:
            print("Failed to fetch positions:")
            try:
                print(json.dumps(response.json(), indent=2))
            except Exception:
                print(response.text)
            return None
        
        return response.json()
    except Exception as exc:
        print(f"Error fetching positions: {exc}")
        return None

def fetch_positions_summary():
    """
    Fetch user's positions summary from the Almanac API.
    
    Returns:
        Response data with summary or None if error
    """
    global CURRENT_SESSION
    if not CURRENT_SESSION:
        print("No active trading session. Create a session first.")
        return None
    
    session_id = CURRENT_SESSION.get("data").get("sessionId")
    wallet_address = (
        CURRENT_SESSION.get("data").get("walletAddress")
        or _get_credential("EOA_WALLET_ADDRESS")
    )
    
    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["x-session-id"] = session_id
    if wallet_address:
        headers["x-wallet-address"] = wallet_address
    
    try:
        response = requests.get(
            f"{ALMANAC_API_URL}/v1/trading/positions/summary",
            headers=headers,
            timeout=30
        )
        
        if response.status_code != 200:
            print("Failed to fetch positions summary:")
            try:
                print(json.dumps(response.json(), indent=2))
            except Exception:
                print(response.text)
            return None
        
        return response.json()
    except Exception as exc:
        print(f"Error fetching positions summary: {exc}")
        return None

def fetch_orders(status: str | None = None, limit: int = 100, offset: int = 0):
    """
    Fetch user's orders from the Almanac API.
    
    Args:
        status: Order status filter ('live', 'delayed', 'matched', 'completed', 'pending', 'cancelled', 'expired', 'failed', or None for all)
        limit: Maximum number of orders to return (1-100)
        offset: Number of orders to skip
    
    Returns:
        Response data with orders or None if error
    """
    global CURRENT_SESSION
    if not CURRENT_SESSION:
        print("No active trading session. Create a session first.")
        return None
    
    session_id = CURRENT_SESSION.get("data").get("sessionId")
    wallet_address = (
        CURRENT_SESSION.get("data").get("walletAddress")
        or _get_credential("EOA_WALLET_ADDRESS")
    )
    
    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["x-session-id"] = session_id
    if wallet_address:
        headers["x-wallet-address"] = wallet_address
    
    params = {
        "limit": limit,
        "offset": offset
    }
    
    # Add status parameter if provided
    if status:
        params["status"] = status
    
    try:
        response = requests.get(
            f"{ALMANAC_API_URL}/v1/trading/orders",
            headers=headers,
            params=params,
            timeout=30
        )
        
        if response.status_code != 200:
            print("Failed to fetch orders:")
            try:
                print(json.dumps(response.json(), indent=2))
            except Exception:
                print(response.text)
            return None
        
        return response.json()
    except Exception as exc:
        print(f"Error fetching orders: {exc}")
        return None

def cancel_order(order_id: str):
    """
    Cancel an unmatched order using session-based authentication.
    
    Args:
        order_id: The order ID to cancel (e.g., "0x1234...")
    
    Returns:
        Response data with success status and order details or None if error
    """
    global CURRENT_SESSION
    if not CURRENT_SESSION:
        print("No active trading session. Create a session first.")
        return None
    
    session_id = CURRENT_SESSION.get("data").get("sessionId")
    wallet_address = (
        CURRENT_SESSION.get("data").get("walletAddress")
        or _get_credential("EOA_WALLET_ADDRESS")
    )
    
    if not session_id or not wallet_address:
        print("Missing session ID or wallet address.")
        return None
    
    headers = {
        "Content-Type": "application/json",
        "x-session-id": session_id,
        "x-wallet-address": wallet_address.lower()
    }
    
    try:
        response = requests.delete(
            f"{ALMANAC_API_URL}/v1/trading/orders/{order_id}",
            headers=headers,
            timeout=30
        )
        
        if response.status_code != 200:
            print("Failed to cancel order:")
            try:
                print(json.dumps(response.json(), indent=2))
            except Exception:
                print(response.text)
            return None
        
        return response.json()
    except Exception as exc:
        print(f"Error cancelling order: {exc}")
        return None


def _redeemable_row_current_value(p: dict) -> float:
    v = p.get("currentValue")
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _claim_plan_position_name(p: dict) -> str:
    """Human-readable line for Polymarket data-api position (claim proceeds UI)."""
    t = (p.get("title") or p.get("question") or "").strip()
    if t:
        return _truncate_text(t, 52)
    cid = str(p.get("conditionId") or "")
    return _truncate_text(cid, 18) if cid else "—"


def _claim_plan_roi_percent_str(p: dict, bet: float, won: float) -> str:
    """ROI for display; prefer API percentPnl, else derive from initial vs current."""
    pct = None
    raw = p.get("percentPnl")
    if raw is not None:
        try:
            pct = float(raw)
        except (TypeError, ValueError):
            pct = None
    if pct is None and bet > 0:
        pct = ((won - bet) / bet) * 100.0
    elif pct is None:
        pct = 0.0
    return f"{pct:+.0f}%"


def _redeemable_row_is_valid(p: dict) -> bool:
    return bool(p.get("conditionId")) and p.get("outcomeIndex") is not None


def _polymarket_get_positions_list(url: str) -> list:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else []
    except Exception as exc:
        print(f"Polymarket data-api error: {exc}")
        return []


def aggregate_claimable_bets(bets: list, include_losing_positions: bool) -> list:
    """
    Group by conditionId. Per group:
      indexSets = sorted unique of (outcomeIndex + 1) for each bet in group
      negativeRisk = any bet.negativeRisk
      if negativeRisk: yesAmount/noAmount = floor(sum(size per outcome 0/1) * 1e6) as strings
    If include_losing_positions is False: drop rows with currentValue <= VALUE_WINNER_MIN.
    """
    filtered = []
    for bet in bets:
        if not _redeemable_row_is_valid(bet):
            continue
        if not include_losing_positions and _redeemable_row_current_value(bet) <= VALUE_WINNER_MIN:
            continue
        filtered.append(bet)

    by_condition: dict[str, list] = {}
    for bet in filtered:
        cid = bet["conditionId"]
        by_condition.setdefault(cid, []).append(bet)

    redemptions = []
    for condition_id, group in by_condition.items():
        index_sets = sorted({int(b["outcomeIndex"]) + 1 for b in group})
        neg = any(b.get("negativeRisk") for b in group)
        item = {"conditionId": condition_id, "indexSets": index_sets, "negativeRisk": neg}
        if neg:
            yes_size = sum(
                float(b.get("size") or 0)
                for b in group
                if int(b.get("outcomeIndex") or 0) == 0
            )
            no_size = sum(
                float(b.get("size") or 0)
                for b in group
                if int(b.get("outcomeIndex") or 0) == 1
            )
            item["yesAmount"] = str(int(yes_size * 1_000_000))
            item["noAmount"] = str(int(no_size * 1_000_000))
        redemptions.append(item)
    return redemptions


def build_polymarket_claim_plan(proxy_lc: str) -> dict:
    """
    Fetch redeemable positions from Polymarket, apply winner gate and losing cleanup slots.
    Returns dict with has_winnings_to_claim, totals, redemptions payload fields, and slices for UI.
    """
    winners_raw = _polymarket_get_positions_list(
        POLY_POSITIONS_WINNERS_URL.format(proxy_lc=proxy_lc)
    )
    winners = [
        p
        for p in winners_raw
        if _redeemable_row_is_valid(p) and _redeemable_row_current_value(p) > VALUE_WINNER_MIN
    ]
    if not winners:
        return {
            "has_winnings_to_claim": False,
            "total_proceeds_usd": 0.0,
            "winning_position_count": 0,
            "losing_cleanup_included": 0,
            "redemption_operation_count": 0,
            "redemptions": [],
            "winners": [],
            "losing_slice": [],
            "redemptions_truncated": False,
        }

    total_proceeds_usd = sum(max(0.0, _redeemable_row_current_value(p)) for p in winners)
    winning_redemptions = aggregate_claimable_bets(winners, include_losing_positions=False)
    num_winning_conditions = len(winning_redemptions)
    slots_for_losing = max(
        0, min(MAX_LOSING_POSITIONS_TO_CLOSE, MAX_TOTAL_REDEMPTIONS - num_winning_conditions)
    )

    losing_slice = []
    if slots_for_losing > 0:
        try:
            mixed = _polymarket_get_positions_list(
                POLY_POSITIONS_MIXED_URL.format(proxy_lc=proxy_lc)
            )
            for p in mixed:
                if not _redeemable_row_is_valid(p):
                    continue
                try:
                    sz = float(p.get("size") or 0)
                except (TypeError, ValueError):
                    continue
                cv = _redeemable_row_current_value(p)
                if sz > SIZE_MIN_TOKENS and cv < VALUE_LOSER_MAX:
                    losing_slice.append(p)
            losing_slice = losing_slice[:slots_for_losing]
        except Exception:
            losing_slice = []

    all_positions = winners + losing_slice
    redemptions = aggregate_claimable_bets(all_positions, include_losing_positions=True)
    truncated = False
    if len(redemptions) > MAX_TOTAL_REDEMPTIONS:
        redemptions = redemptions[:MAX_TOTAL_REDEMPTIONS]
        truncated = True

    return {
        "has_winnings_to_claim": True,
        "total_proceeds_usd": total_proceeds_usd,
        "winning_position_count": len(winners),
        "losing_cleanup_included": len(losing_slice),
        "redemption_operation_count": len(redemptions),
        "redemptions": redemptions,
        "winners": winners,
        "losing_slice": losing_slice,
        "redemptions_truncated": truncated,
    }


def _post_almanac_redeem_prepare(redemptions: list) -> tuple[bool, str | None, dict | None]:
    """
    POST /v1/redeem. Almanac returns success with calldata; it does not broadcast on-chain.
    Returns (ok, error_message, json_payload).
    """
    global CURRENT_SESSION
    if not CURRENT_SESSION:
        return False, "No active trading session.", None
    data = CURRENT_SESSION.get("data") or {}
    session_id = data.get("sessionId")
    eoa = data.get("walletAddress") or _get_credential("EOA_WALLET_ADDRESS")
    proxy = data.get("proxyWallet") or _get_credential("EOA_PROXY_FUNDER")
    if not session_id or not eoa or not proxy:
        return False, "Missing sessionId, wallet, or proxy wallet.", None

    headers = {
        "Content-Type": "application/json",
        "x-session-id": session_id,
        "x-wallet-address": str(eoa).lower(),
    }
    body = {
        "redemptions": [
            {
                "conditionId": r["conditionId"],
                "indexSets": r["indexSets"],
                "negativeRisk": r["negativeRisk"],
                **(
                    {"yesAmount": r["yesAmount"], "noAmount": r["noAmount"]}
                    if r.get("negativeRisk")
                    else {}
                ),
            }
            for r in redemptions
        ],
        "userWalletAddress": eoa,
        "proxyWallet": proxy,
        "sessionId": session_id,
    }
    try:
        response = requests.post(
            f"{ALMANAC_API_URL}/v1/redeem",
            headers=headers,
            json=body,
            timeout=60,
        )
        if response.status_code != 200:
            try:
                return False, json.dumps(response.json(), indent=2), None
            except Exception:
                return False, response.text or "", None

        raw = (response.text or "").strip()
        if not raw:
            return False, "Empty response body from redeem API (HTTP 200).", None
        try:
            payload = response.json()
        except Exception:
            return (
                False,
                f"Redeem returned non-JSON (HTTP 200): {raw[:800]}",
                None,
            )

        if not isinstance(payload, dict):
            return False, f"Unexpected redeem JSON type: {type(payload).__name__}", None

        if not bool(payload.get("success")):
            return False, json.dumps(payload, indent=2), None

        return True, None, payload
    except Exception as exc:
        return False, str(exc), None


def _relay_submit_redeem_transactions(
    txs_raw: list, redeem_payload: dict | None = None
) -> tuple[bool, str | None]:
    """Submit redeem calldata from Almanac through Polymarket relayer (same path as withdraw)."""
    global CURRENT_SESSION
    load_dotenv(dotenv_path=str(ENV_PATH), override=True)
    client = _make_relay_client()
    if not client:
        return False, "Could not create relayer client (EOA_WALLET_PK or builder-sign)."
    proxy_addr = (CURRENT_SESSION.get("data") or {}).get("proxyWallet") if CURRENT_SESSION else None
    if not proxy_addr:
        proxy_addr = _get_credential("EOA_PROXY_FUNDER")
    onramp_addr = (
        os.environ.get("COLLATERAL_ONRAMP_ADDRESS", COLLATERAL_ONRAMP_ADDRESS) or ""
    ).strip()
    should_auto_wrap = bool(
        AUTO_WRAP_USDCE_PROCEEDS
        and onramp_addr
        and proxy_addr
        and _payload_has_usdce_redemptions(redeem_payload)
    )
    w3: Web3 | None = None
    usdce_before: int | None = None
    if should_auto_wrap:
        w3 = get_polygon_web3()
        if w3 is None:
            print(
                "Auto-wrap skipped: Polygon RPC unavailable; set POLYGON_RPC_URL and wrap manually if needed."
            )
            should_auto_wrap = False
        else:
            try:
                usdce_before = _read_erc20_balance_wei(w3, USDC_E_POLYGON, proxy_addr)
            except Exception as exc:
                print(f"Auto-wrap skipped: could not read pre-claim USDC.e balance ({exc}).")
                should_auto_wrap = False
    elif AUTO_WRAP_USDCE_PROCEEDS and _payload_has_usdce_redemptions(redeem_payload) and not onramp_addr:
        print(
            "Auto-wrap skipped: COLLATERAL_ONRAMP_ADDRESS is not set. "
            f"Set it in {ENV_PATH} to auto-convert claimed USDC.e into {PROXY_TOKEN_SYMBOL}."
        )

    relay_txs: list[SafeTransaction] = []
    for t in txs_raw:
        if not isinstance(t, dict):
            return False, f"Invalid transaction entry in redeem response: {t!r}"
        to_addr = t.get("to")
        data = t.get("data")
        if not to_addr or not data:
            return False, f"Redeem tx missing to/data: {t!r}"
        try:
            to_chk = Web3.to_checksum_address(to_addr)
        except Exception as exc:
            return False, f"Invalid redeem tx `to` {to_addr!r}: {exc}"
        ds = str(data)
        if not ds.startswith("0x"):
            ds = "0x" + ds
        try:
            op_int = int(t.get("operation", 0))
        except (TypeError, ValueError):
            op_int = 0
        op_type = (
            OperationType.DelegateCall if op_int == 1 else OperationType.Call
        )
        relay_txs.append(
            SafeTransaction(
                to=to_chk,
                operation=op_type,
                data=ds,
                value=str(t.get("value", "0")),
            )
        )
    try:
        print("\nSubmitting redemption transaction(s)...")
        resp = client.execute(relay_txs, "Redeem Polymarket proceeds")
        result = resp.wait()
        if result is not None:
            if isinstance(result, dict):
                th = result.get("transactionHash") or result.get("transaction_hash")
                if th:
                    print(f"Polygon transaction hash: {th}")
            if should_auto_wrap and w3 is not None and proxy_addr and usdce_before is not None:
                try:
                    usdce_after = _read_erc20_balance_wei(w3, USDC_E_POLYGON, proxy_addr)
                except Exception as exc:
                    return (
                        True,
                        "Claim succeeded, but could not read post-claim USDC.e balance for auto-wrap: "
                        f"{exc}",
                    )
                delta = max(0, int(usdce_after) - int(usdce_before))
                if delta > 0:
                    ok_wrap, wrap_err = _auto_wrap_usdce_proceeds(
                        client, w3, proxy_addr, onramp_addr, delta
                    )
                    if not ok_wrap:
                        return (
                            True,
                            "Claim succeeded, but auto-wrap USDC.e -> "
                            f"{PROXY_TOKEN_SYMBOL} failed: {wrap_err}",
                        )
                    print(f"Auto-wrap complete: converted claimed USDC.e into {PROXY_TOKEN_SYMBOL}.")
                else:
                    print("Auto-wrap skipped: no newly claimed USDC.e detected.")
            return True, None
        return False, "Relayer did not confirm the transaction (timeout or on-chain failure)."
    except RelayerClientException as exc:
        msg = str(exc)
        if getattr(exc, "status_code", None) == 401:
            msg += (
                " Try: Trading menu → Refresh Trading Session. "
                "Builder headers must be accepted by the Polymarket relayer."
            )
        return False, msg
    except Exception as exc:
        return False, str(exc)


def request_redeem_transactions(redemptions: list) -> tuple[bool, str | None]:
    """
    Prepare redemptions via Almanac POST /v1/redeem, then execute returned calldata
    through the Polymarket relayer (EOA-signed Safe txs). Returns (True, None) only after
    relayer success; Almanac alone only returns prepared transactions, not mined redeems.
    """
    ok, err, payload = _post_almanac_redeem_prepare(redemptions)
    if not ok or not payload:
        return False, err

    txs_raw = payload.get("transactions")
    if isinstance(txs_raw, list) and len(txs_raw) > 0:
        return _relay_submit_redeem_transactions(txs_raw, redeem_payload=payload)

    api_msg = payload.get("message") or ""
    return (
        False,
        "Almanac returned success but no `transactions` to execute on-chain.\n"
        f"API message: {api_msg!r}\n"
        f"{json.dumps(payload, indent=2)}",
    )


def claim_proceeds_menu():
    """
    On entry: fetch claimable proceeds from Polymarket data-api; POST Almanac /v1/redeem then
    execute returned txs via Polymarket relayer (EOA-signed Safe).
    """
    global CURRENT_SESSION
    if not CURRENT_SESSION:
        print("No active trading session. Create a session first.")
        return

    proxy = (CURRENT_SESSION.get("data") or {}).get("proxyWallet") or _get_credential(
        "EOA_PROXY_FUNDER"
    )
    if not proxy:
        print("No proxy wallet on session or EOA_PROXY_FUNDER; cannot resolve Polymarket positions user.")
        return

    proxy_lc = proxy.lower()

    print("\nChecking for claimable proceeds...")
    plan = build_polymarket_claim_plan(proxy_lc)

    if not plan["has_winnings_to_claim"]:
        print("No claimable proceeds right now.")
        input("\nPress Enter to continue...")
        return

    print(f"\nAmount available to claim: ${plan['total_proceeds_usd']:.2f}")

    if plan.get("redemptions_truncated"):
        print(
            f"Note: at most {MAX_TOTAL_REDEMPTIONS} markets can be included in one request."
        )

    rows = []
    for p in plan["winners"]:
        name = _claim_plan_position_name(p)
        try:
            bet = (
                float(p["initialValue"])
                if p.get("initialValue") is not None
                else 0.0
            )
        except (TypeError, ValueError):
            bet = 0.0
        won = max(0.0, _redeemable_row_current_value(p))
        roi = _claim_plan_roi_percent_str(p, bet, won)
        rows.append(
            [
                name,
                f"${bet:.2f}",
                f"${won:.2f}",
                roi,
            ]
        )
    if rows:
        print("\nPositions:")
        print(
            tabulate(
                rows,
                headers=["Position", "Amount bet", "Amount won", "ROI%"],
                tablefmt="grid",
            )
        )

    confirm = input("\nRequest claim transactions from Almanac? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        input("\nPress Enter to continue...")
        return

    ok, err = request_redeem_transactions(plan["redemptions"])
    if ok:
        print("\nClaim succeeded (relayer confirmed on-chain).")
        if err:
            print(err)
    else:
        print("\nClaim failed.")
        if err:
            print(err)
    input("\nPress Enter to continue...")


def _inject_polygon_poa_middleware(w3: Web3) -> None:
    try:
        from web3.middleware import ExtraDataToPOAMiddleware

        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    except Exception:
        try:
            from web3.middleware import geth_poa_middleware

            w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        except Exception:
            pass


def _polygon_rpc_url() -> str:
    load_dotenv(dotenv_path=str(ENV_PATH), override=True)
    u = os.environ.get("POLYGON_RPC_URL", "").strip()
    return u or DEFAULT_POLYGON_RPC_URL


def get_polygon_web3() -> Web3 | None:
    rpc = _polygon_rpc_url()
    w3 = Web3(Web3.HTTPProvider(rpc))
    if not w3.is_connected():
        return None
    _inject_polygon_poa_middleware(w3)
    return w3


POLYGON_GAS_PRICE_BUMP = 1.2


def _polygon_bumped_legacy_gas_price_wei(w3: Web3, bump_ratio: float = POLYGON_GAS_PRICE_BUMP) -> int:
    """
    Legacy gasPrice above eth_gasPrice so txs are less likely to sit in mempool
    when fees move up right after broadcast.
    """
    base = int(w3.eth.gas_price)
    bumped = int(base * bump_ratio)
    return bumped if bumped > base else base + 1


def _encode_erc20_transfer(w3: Web3, token: str, to_addr: str, amount_wei: int) -> str:
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(token), abi=ERC20_MIN_ABI
    )
    data = contract.encode_abi(
        abi_element_identifier="transfer",
        args=[Web3.to_checksum_address(to_addr), amount_wei],
    )
    if isinstance(data, bytes):
        return "0x" + data.hex()
    s = str(data)
    return s if s.startswith("0x") else "0x" + s


def _encode_erc20_approve(w3: Web3, token: str, spender: str, amount_wei: int) -> str:
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(token), abi=ERC20_MIN_ABI
    )
    data = contract.encode_abi(
        abi_element_identifier="approve",
        args=[Web3.to_checksum_address(spender), int(amount_wei)],
    )
    if isinstance(data, bytes):
        return "0x" + data.hex()
    s = str(data)
    return s if s.startswith("0x") else "0x" + s


def _encode_onramp_wrap(
    w3: Web3, onramp_addr: str, asset_addr: str, to_addr: str, amount_wei: int
) -> str:
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(onramp_addr), abi=COLLATERAL_ONRAMP_WRAP_ABI
    )
    data = contract.encode_abi(
        abi_element_identifier="wrap",
        args=[
            Web3.to_checksum_address(asset_addr),
            Web3.to_checksum_address(to_addr),
            int(amount_wei),
        ],
    )
    if isinstance(data, bytes):
        return "0x" + data.hex()
    s = str(data)
    return s if s.startswith("0x") else "0x" + s


def _read_erc20_balance_wei(w3: Web3, token: str, owner: str) -> int:
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(token), abi=ERC20_MIN_ABI
    )
    return int(contract.functions.balanceOf(Web3.to_checksum_address(owner)).call())


def _read_erc20_allowance_wei(w3: Web3, token: str, owner: str, spender: str) -> int:
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(token), abi=ERC20_MIN_ABI
    )
    return int(
        contract.functions.allowance(
            Web3.to_checksum_address(owner), Web3.to_checksum_address(spender)
        ).call()
    )


def _payload_has_usdce_redemptions(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False
    prepared = payload.get("redemptions")
    if not isinstance(prepared, list):
        return False
    for item in prepared:
        if not isinstance(item, dict):
            continue
        sel = str(item.get("selectedCollateral") or "").strip().lower()
        if sel == USDC_E_POLYGON.lower():
            return True
    return False


def _auto_wrap_usdce_proceeds(
    client: RelayClient, w3: Web3, proxy_addr: str, onramp_addr: str, amount_wei: int
) -> tuple[bool, str | None]:
    try:
        allowance = _read_erc20_allowance_wei(
            w3, USDC_E_POLYGON, proxy_addr, onramp_addr
        )
    except Exception as exc:
        return False, f"Could not read USDC.e allowance: {exc}"

    if allowance < amount_wei:
        try:
            print("Approving USDC.e for collateral onramp...")
            approve_data = _encode_erc20_approve(
                w3, USDC_E_POLYGON, onramp_addr, UINT256_MAX
            )
            approve_tx = SafeTransaction(
                to=Web3.to_checksum_address(USDC_E_POLYGON),
                operation=OperationType.Call,
                data=approve_data,
                value="0",
            )
            ar = client.execute([approve_tx], "Approve USDC.e for auto-wrap")
            if ar.wait() is None:
                return False, "Approval transaction timed out or failed."
        except Exception as exc:
            return False, f"USDC.e approval failed: {exc}"

    try:
        print(f"Wrapping {amount_wei / (10**USDC_E_DECIMALS):.6f} USDC.e into {PROXY_TOKEN_SYMBOL}...")
        wrap_data = _encode_onramp_wrap(
            w3, onramp_addr, USDC_E_POLYGON, proxy_addr, amount_wei
        )
        wrap_tx = SafeTransaction(
            to=Web3.to_checksum_address(onramp_addr),
            operation=OperationType.Call,
            data=wrap_data,
            value="0",
        )
        wr = client.execute([wrap_tx], f"Auto-wrap USDC.e to {PROXY_TOKEN_SYMBOL}")
        if wr.wait() is None:
            return False, "Wrap transaction timed out or failed."
        return True, None
    except Exception as exc:
        return False, f"USDC.e wrap failed: {exc}"


def fetch_auth_balances_raw() -> dict | None:
    """GET /v1/auth/balances/{eoa} with session headers. Returns parsed JSON dict or None."""
    global CURRENT_SESSION
    if not CURRENT_SESSION:
        print("No active trading session. Create a session first.")
        return None
    data = CURRENT_SESSION.get("data") or {}
    session_id = data.get("sessionId")
    eoa = data.get("walletAddress") or _get_credential("EOA_WALLET_ADDRESS")
    if not session_id or not eoa:
        print("Missing session or wallet address.")
        return None
    headers = {
        "Content-Type": "application/json",
        "x-session-id": session_id,
        "x-wallet-address": str(eoa).lower(),
    }
    try:
        r = requests.get(
            f"{ALMANAC_API_URL}/v1/auth/balances/{eoa}",
            headers=headers,
            timeout=30,
        )
        if r.status_code != 200:
            try:
                print(json.dumps(r.json(), indent=2))
            except Exception:
                print(r.text or f"HTTP {r.status_code}")
            return None
        return r.json()
    except Exception as exc:
        print(f"Error fetching balances: {exc}")
        return None


def _extract_proxy_token_formatted(payload: dict) -> str | None:
    if not isinstance(payload, dict):
        return None
    root = payload.get("data")
    if root is None:
        root = payload
    if not isinstance(root, dict):
        return None
    proxy = root.get("proxy")
    if not isinstance(proxy, dict):
        return None
    bal = proxy.get("balance")
    if isinstance(bal, dict):
        f = bal.get("formatted")
        if f is not None:
            return str(f)
        amt = bal.get("amount")
        if amt is not None:
            return str(amt)
    if isinstance(bal, str):
        return bal
    return None


def funds_show_balance() -> None:
    raw = fetch_auth_balances_raw()
    if raw is None:
        return
    formatted = _extract_proxy_token_formatted(raw)
    print("\nSafe/Proxy wallet balance")
    if formatted is not None:
        print(f"  {PROXY_TOKEN_SYMBOL}: {formatted}")
    else:
        print("  Could not read balance from API. Response keys:")
        print(json.dumps(list(raw.keys()) if isinstance(raw, dict) else raw, indent=2))
    input("\nPress Enter to continue...")


def _print_eoa_balances_for_deposit(w3: Web3, eoa: str) -> None:
    """pUSD, USDC.e, native USDC, and POL on the EOA on Polygon (chain 137)."""
    try:
        cs = Web3.to_checksum_address(eoa)
        pusd = w3.eth.contract(
            address=Web3.to_checksum_address(PROXY_TOKEN_POLYGON), abi=ERC20_MIN_ABI
        )
        usdc_e = w3.eth.contract(
            address=Web3.to_checksum_address(USDC_E_POLYGON), abi=ERC20_MIN_ABI
        )
        native_u = w3.eth.contract(
            address=Web3.to_checksum_address(NATIVE_USDC_POLYGON), abi=ERC20_MIN_ABI
        )
        p_wei = pusd.functions.balanceOf(cs).call()
        e_wei = usdc_e.functions.balanceOf(cs).call()
        n_wei = native_u.functions.balanceOf(cs).call()
        pol_wei = w3.eth.get_balance(cs)
        print(f"\nYour Polygon EVM wallet ({eoa})")
        print(f"  {PROXY_TOKEN_SYMBOL}:        {p_wei / 10**PROXY_TOKEN_DECIMALS:.6f}")
        print(f"  USDC.e:      {e_wei / 10**USDC_E_DECIMALS:.6f}")
        print(f"  Native USDC: {n_wei / 10**USDC_E_DECIMALS:.6f}")
        print(f"  POL (gas):   {pol_wei / 10**18:.6f}")
    except Exception as exc:
        print(f"\nCould not load EOA balances: {exc}")


def _transfer_erc20_eoa_to_proxy(
    w3: Web3,
    safe: str,
    account: Account,
    token_address: str,
    token_symbol: str,
    token_decimals: int,
) -> None:
    eoa = account.address
    amt_s = input(f"\n{token_symbol} amount to send to proxy (human units, e.g. 10.5): ").strip()
    try:
        amt_human = float(amt_s)
    except ValueError:
        print("Invalid amount.")
        return
    if amt_human <= 0:
        print("Amount must be positive.")
        return
    amount_wei = int(amt_human * (10**token_decimals))
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(token_address), abi=ERC20_MIN_ABI
    )
    bal = contract.functions.balanceOf(Web3.to_checksum_address(eoa)).call()
    if bal < amount_wei:
        print(
            f"Insufficient {token_symbol} on your wallet. Have {bal / 10**token_decimals:.6f}, need {amt_human:.6f}."
        )
        return
    gas_bal = w3.eth.get_balance(Web3.to_checksum_address(eoa))
    if gas_bal == 0:
        print("Your wallet has 0 POL; you need POL on Polygon for gas.")
        return
    print(f"\nSend {amt_human} {token_symbol} from your wallet to proxy {safe}?")
    if input("Confirm [y/N]: ").strip().lower() != "y":
        print("Cancelled.")
        return
    try:
        nonce = w3.eth.get_transaction_count(Web3.to_checksum_address(eoa))
        gas_price = _polygon_bumped_legacy_gas_price_wei(w3)
        tx = contract.functions.transfer(
            Web3.to_checksum_address(safe), amount_wei
        ).build_transaction(
            {
                "from": Web3.to_checksum_address(eoa),
                "nonce": nonce,
                "gas": 120000,
                "gasPrice": gas_price,
                "chainId": POLYGON_CHAIN_ID,
            }
        )
        signed = account.sign_transaction(tx)
        raw = getattr(signed, "raw_transaction", None) or getattr(
            signed, "rawTransaction", None
        )
        if raw is None:
            print("Signing failed.")
            return
        h = w3.eth.send_raw_transaction(raw)
        print(f"Submitted. Tx hash: {Web3.to_hex(h)}")
    except Exception as exc:
        print(f"Deposit failed: {exc}")


def _fetch_polymarket_bridge_evm_address(safe: str) -> tuple[str | None, str | None]:
    """
    POST bridge deposit API; return (checksummed EVM deposit address, optional API note) or (None, None).
    """
    try:
        r = requests.post(
            POLYMARKET_BRIDGE_DEPOSIT_URL,
            json={"address": Web3.to_checksum_address(safe)},
            timeout=30,
        )
        if not r.ok:
            print(f"Bridge API HTTP {r.status_code}")
            try:
                print(json.dumps(r.json(), indent=2))
            except Exception:
                print(r.text)
            return None, None
        j = r.json()
    except Exception as exc:
        print(f"Bridge request failed: {exc}")
        return None, None

    evm = None
    if isinstance(j, dict):
        addr = j.get("address")
        if isinstance(addr, dict):
            evm = addr.get("evm")
        evm = evm or j.get("evm")
    if not evm:
        print("Unexpected bridge response (no EVM address):")
        print(json.dumps(j, indent=2) if isinstance(j, dict) else j)
        return None, None
    try:
        chk = Web3.to_checksum_address(evm)
    except Exception:
        print(f"Invalid EVM address from bridge: {evm!r}")
        return None, None
    note = j.get("note") if isinstance(j, dict) else None
    note_s = str(note) if note else None
    return chk, note_s


def _bridge_deposit_execute(w3: Web3, safe: str, account: Account) -> None:
    """Sign and broadcast native USDC or POL from this EOA to Polymarket bridge (credits proxy)."""
    bridge_to, api_note = _fetch_polymarket_bridge_evm_address(safe)
    if not bridge_to:
        return
    eoa = account.address
    eoa_cs = Web3.to_checksum_address(eoa)

    print("\nPolymarket bridge (send from this wallet)")
    print(f"  Your proxy will be credited: {safe}")
    print(f"  On-chain recipient: {bridge_to}")
    if api_note:
        print(f"  {api_note}")

    print("\n  1) Native USDC (ERC-20 on Polygon)")
    print("  2) POL (native token)")
    print("  3) Cancel")
    asset = input("\nChoose what to send: ").strip()
    if asset == "3" or not asset:
        print("Cancelled.")
        return
    if asset not in ("1", "2"):
        print("Invalid choice.")
        return

    amt_s = input("Amount to send (human units): ").strip()
    try:
        amt_human = float(amt_s)
    except ValueError:
        print("Invalid amount.")
        return
    if amt_human <= 0:
        print("Amount must be positive.")
        return

    gas_price = _polygon_bumped_legacy_gas_price_wei(w3)
    nonce = w3.eth.get_transaction_count(eoa_cs)

    if asset == "1":
        amount_wei = int(amt_human * (10**USDC_E_DECIMALS))
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(NATIVE_USDC_POLYGON), abi=ERC20_MIN_ABI
        )
        bal = contract.functions.balanceOf(eoa_cs).call()
        if bal < amount_wei:
            print(
                f"Insufficient native USDC. Have {bal / 10**USDC_E_DECIMALS:.6f}, need {amt_human:.6f}."
            )
            return
        pol_bal = w3.eth.get_balance(eoa_cs)
        est_gas_cost = gas_price * 120000
        if pol_bal < est_gas_cost:
            print(
                f"Insufficient POL for gas (need ~{est_gas_cost / 10**18:.6f} POL). "
                f"Have {pol_bal / 10**18:.6f}."
            )
            return
        print(
            f"\nSend {amt_human} native USDC to bridge → proxy {safe}? "
            f"Tx to {bridge_to}"
        )
        if input("Confirm [y/N]: ").strip().lower() != "y":
            print("Cancelled.")
            return
        try:
            tx = contract.functions.transfer(bridge_to, amount_wei).build_transaction(
                {
                    "from": eoa_cs,
                    "nonce": nonce,
                    "gas": 120000,
                    "gasPrice": gas_price,
                    "chainId": POLYGON_CHAIN_ID,
                }
            )
            signed = account.sign_transaction(tx)
            raw = getattr(signed, "raw_transaction", None) or getattr(
                signed, "rawTransaction", None
            )
            if raw is None:
                print("Signing failed.")
                return
            h = w3.eth.send_raw_transaction(raw)
            print(f"Submitted. Tx hash: {Web3.to_hex(h)}")
            print("After the tx confirms, the bridge may take a few minutes to credit your proxy.")
        except Exception as exc:
            print(f"Bridge deposit failed: {exc}")
        return

    # POL native transfer
    amount_wei = int(amt_human * 10**18)
    gas_limit = 21000
    pol_bal = w3.eth.get_balance(eoa_cs)
    gas_cost = gas_price * gas_limit
    if pol_bal < amount_wei + gas_cost:
        print(
            f"Insufficient POL. Need {amt_human:.6f} + ~{gas_cost / 10**18:.6f} gas; "
            f"have {pol_bal / 10**18:.6f}."
        )
        return
    print(
        f"\nSend {amt_human} POL to bridge → proxy {safe}? Tx to {bridge_to}"
    )
    if input("Confirm [y/N]: ").strip().lower() != "y":
        print("Cancelled.")
        return
    try:
        tx = {
            "from": eoa_cs,
            "to": bridge_to,
            "value": amount_wei,
            "nonce": nonce,
            "gas": gas_limit,
            "gasPrice": gas_price,
            "chainId": POLYGON_CHAIN_ID,
        }
        signed = account.sign_transaction(tx)
        raw = getattr(signed, "raw_transaction", None) or getattr(
            signed, "rawTransaction", None
        )
        if raw is None:
            print("Signing failed.")
            return
        h = w3.eth.send_raw_transaction(raw)
        print(f"Submitted. Tx hash: {Web3.to_hex(h)}")
        print("After the tx confirms, the bridge may take a few minutes to credit your proxy.")
    except Exception as exc:
        print(f"Bridge deposit failed: {exc}")


def funds_deposit_to_proxy() -> None:
    """Polygon EVM only: deposit pUSD/USDC.e from EOA, or bridge native USDC/POL (EVM deposit address)."""
    global CURRENT_SESSION
    if not CURRENT_SESSION:
        print("No active trading session.")
        return
    safe = (CURRENT_SESSION.get("data") or {}).get("proxyWallet") or _get_credential(
        "EOA_PROXY_FUNDER"
    )
    if not safe:
        print("No proxy wallet on session.")
        return
    eoa = (CURRENT_SESSION.get("data") or {}).get("walletAddress") or _get_credential(
        "EOA_WALLET_ADDRESS"
    )
    pk = _get_credential("EOA_WALLET_PK")
    if not pk:
        print("EOA_WALLET_PK missing.")
        return
    if not pk.startswith("0x"):
        pk = "0x" + pk
    try:
        account = Account.from_key(pk)
    except Exception as exc:
        print(f"Invalid private key: {exc}")
        return
    if account.address.lower() != (eoa or "").lower():
        print("Warning: private key does not match session / EOA_WALLET_ADDRESS.")

    print("\nDeposit to proxy (Polygon)")
    print(f"  Proxy (EVM): {safe}")
    print("  Use your Polygon EVM private key / wallet; balances below are on chain ID 137.")

    w3 = get_polygon_web3()
    if w3 and eoa:
        _print_eoa_balances_for_deposit(w3, eoa)
    else:
        print(
            f"\n(RPC not available — could not load Polygon balances. "
            f"Endpoint: {_polygon_rpc_url()!r}.)"
        )

    print(f"\n  1) Transfer {PROXY_TOKEN_SYMBOL} from this EOA to the proxy (POL on Polygon for gas)")
    print("  2) Transfer USDC.e from this EOA to the proxy (POL on Polygon for gas)")
    print("  3) Polymarket bridge — send native USDC or POL from this wallet (on-chain)")
    print("  4) Back")
    sub = input("\nEnter choice: ").strip()
    if sub == "1":
        if not w3:
            print(
                f"{PROXY_TOKEN_SYMBOL} transfer needs a working Polygon RPC. "
                "Check POLYGON_RPC_URL or your network."
            )
            input("\nPress Enter to continue...")
            return
        _transfer_erc20_eoa_to_proxy(
            w3,
            safe,
            account,
            token_address=PROXY_TOKEN_POLYGON,
            token_symbol=PROXY_TOKEN_SYMBOL,
            token_decimals=PROXY_TOKEN_DECIMALS,
        )
        input("\nPress Enter to continue...")
    elif sub == "2":
        if not w3:
            print(
                "USDC.e transfer needs a working Polygon RPC. "
                "Check POLYGON_RPC_URL or your network."
            )
            input("\nPress Enter to continue...")
            return
        _transfer_erc20_eoa_to_proxy(
            w3,
            safe,
            account,
            token_address=USDC_E_POLYGON,
            token_symbol="USDC.e",
            token_decimals=USDC_E_DECIMALS,
        )
        input("\nPress Enter to continue...")
    elif sub == "3":
        if not w3:
            print(
                "Bridge deposit from this wallet needs a working Polygon RPC. "
                "Check POLYGON_RPC_URL or your network."
            )
            input("\nPress Enter to continue...")
            return
        _bridge_deposit_execute(w3, safe, account)
        input("\nPress Enter to continue...")
    elif sub == "4":
        return
    else:
        print("Invalid choice.")
        input("\nPress Enter to continue...")


class AlmanacRelayClient(RelayClient):
    """
    Default RelayClient passes str(dict) into builder-sign; the relayer POST uses
    the same dict encoded with requests' json.dumps(..., allow_nan=False). Polymarket
    verifies the HMAC against that JSON, so builder-sign must sign the identical
    string (see Almanac builder-sign + @polymarket/builder-signing-sdk contract).
    """

    def poll_until_state(
        self,
        transaction_id: str,
        states: list,
        fail_state: str,
        max_polls: int | None = None,
        poll_frequency: int | None = None,
    ):
        """Same as RelayClient.poll_until_state but with a short user-facing status line."""
        target_states = set(list(states))
        poll_limit = max_polls if max_polls is not None else 10
        poll_frequency_ms = 2000
        if poll_frequency is not None and poll_frequency >= 1000:
            poll_frequency_ms = poll_frequency

        print("Waiting for confirmation...")

        for _ in range(poll_limit):
            transactions = self.get_transaction(transaction_id)
            if transactions:
                txn = transactions[0]
                txn_state = txn.get("state")
                if (
                    txn_state
                    and isinstance(txn_state, str)
                    and txn_state in target_states
                ):
                    return txn
                if fail_state is not None and txn_state == fail_state:
                    txn_hash = txn.get("transactionHash")
                    self.logger.error(
                        f"txn {transaction_id} failed onchain, transaction_hash: {txn_hash}!"
                    )
                    return None
            time.sleep(poll_frequency_ms / 1000)

        self.logger.info(
            f"Transaction {transaction_id} not found or not in given states, timing out!"
        )
        return None

    def _generate_builder_headers(
        self, method: str, request_path: str, body: dict | None = None
    ) -> dict | None:
        body_for_sign: str | None = None
        if body is not None:
            body_for_sign = requests_json.dumps(body, allow_nan=False)
        headers = self.builder_config.generate_builder_headers(
            method, request_path, body_for_sign
        )
        return headers.to_dict() if headers is not None else None


@dataclass
class _BuilderHeaders:
    POLY_BUILDER_API_KEY: str
    POLY_BUILDER_TIMESTAMP: str
    POLY_BUILDER_PASSPHRASE: str
    POLY_BUILDER_SIGNATURE: str

    def to_dict(self) -> dict[str, str]:
        return {
            "POLY_BUILDER_API_KEY": self.POLY_BUILDER_API_KEY,
            "POLY_BUILDER_TIMESTAMP": self.POLY_BUILDER_TIMESTAMP,
            "POLY_BUILDER_PASSPHRASE": self.POLY_BUILDER_PASSPHRASE,
            "POLY_BUILDER_SIGNATURE": self.POLY_BUILDER_SIGNATURE,
        }


class AlmanacRelayBuilderConfig:
    """
    Calls Almanac POST /v1/auth/builder-sign and returns relayer builder headers.
    The stock RemoteBuilderConfig path returns a plain dict, but RelayClient expects .to_dict().
    Pass trading session headers so Almanac can return relayer-valid builder credentials.
    """

    def __init__(self, sign_url: str, session_headers: dict | None = None):
        self._sign_url = sign_url.rstrip("/")
        self._session_headers = dict(session_headers) if session_headers else {}

    def generate_builder_headers(
        self,
        method: str,
        path: str,
        body: str | None = None,
        timestamp: int | None = None,
    ):
        ts = int(time.time()) if timestamp is None else int(timestamp)
        payload = {
            "method": method,
            "path": path,
            "body": body,
            "timestamp": ts,
        }
        req_headers = {"Content-Type": "application/json", **self._session_headers}
        try:
            r = requests.post(
                self._sign_url,
                json=payload,
                headers=req_headers,
                timeout=30,
            )
            if not r.ok:
                return None
            d = r.json()
            if not isinstance(d, dict):
                return None
        except Exception:
            return None
        def _g(*keys: str) -> str | None:
            for k in keys:
                if k in d and d[k] is not None:
                    return str(d[k])
            return None

        key = _g("POLY_BUILDER_API_KEY", "polyBuilderApiKey", "apiKey")
        hdr_ts = _g("POLY_BUILDER_TIMESTAMP", "polyBuilderTimestamp", "timestamp")
        phrase = _g("POLY_BUILDER_PASSPHRASE", "polyBuilderPassphrase", "passphrase")
        sig = _g("POLY_BUILDER_SIGNATURE", "polyBuilderSignature", "signature")
        if not all((key, hdr_ts, phrase, sig)):
            return None
        return _BuilderHeaders(
            POLY_BUILDER_API_KEY=key,
            POLY_BUILDER_TIMESTAMP=hdr_ts,
            POLY_BUILDER_PASSPHRASE=phrase,
            POLY_BUILDER_SIGNATURE=sig,
        )


def _make_relay_client() -> RelayClient | None:
    global CURRENT_SESSION
    pk = _get_credential("EOA_WALLET_PK")
    if not pk:
        print("EOA_WALLET_PK missing.")
        return None
    if not pk.startswith("0x"):
        pk = "0x" + pk
    session_headers: dict[str, str] = {}
    if CURRENT_SESSION:
        sdata = CURRENT_SESSION.get("data") or {}
        sid = sdata.get("sessionId")
        waddr = sdata.get("walletAddress") or _get_credential("EOA_WALLET_ADDRESS")
        if sid:
            session_headers["x-session-id"] = sid
        if waddr:
            session_headers["x-wallet-address"] = str(waddr).lower()
    try:
        builder = AlmanacRelayBuilderConfig(
            f"{ALMANAC_API_URL}/v1/auth/builder-sign",
            session_headers=session_headers,
        )
        return AlmanacRelayClient(
            POLYMARKET_RELAYER_URL,
            POLYGON_CHAIN_ID,
            pk,
            builder,
        )
    except Exception as exc:
        print(f"Could not create relay client: {exc}")
        return None


def funds_withdraw_usdc_e() -> None:
    """Withdraw proxy token from proxy/Safe via Polymarket relayer (gasless)."""
    global CURRENT_SESSION
    if not CURRENT_SESSION:
        print("No active trading session.")
        return
    session_proxy = (CURRENT_SESSION.get("data") or {}).get("proxyWallet") or _get_credential(
        "EOA_PROXY_FUNDER"
    )
    default_dest = (
        (CURRENT_SESSION.get("data") or {}).get("walletAddress")
        or _get_credential("EOA_WALLET_ADDRESS")
    )

    print(f"\nWithdraw {PROXY_TOKEN_SYMBOL} from proxy")
    if session_proxy:
        print(f"  Proxy: {session_proxy}")
    raw_bal = fetch_auth_balances_raw()
    if raw_bal:
        fb = _extract_proxy_token_formatted(raw_bal)
        if fb is not None:
            print(f"  {PROXY_TOKEN_SYMBOL} (trading balance): {fb}")
        else:
            print(f"  {PROXY_TOKEN_SYMBOL} (trading balance): unavailable from API response.")
    else:
        print(f"  {PROXY_TOKEN_SYMBOL} (trading balance): not loaded (API error above, if any).")

    client = _make_relay_client()
    if not client:
        return
    try:
        derived_safe = client.get_expected_safe()
    except Exception as exc:
        print(f"Could not derive Safe: {exc}")
        input("\nPress Enter to continue...")
        return
    if (
        session_proxy
        and derived_safe
        and session_proxy.lower() != derived_safe.lower()
    ):
        print(
            f"Note: session proxy {session_proxy} != relayer derived Safe {derived_safe}; using relayer derivation."
        )

    dest = input(f"\nDestination address [{default_dest}]: ").strip() or default_dest
    try:
        dest_chk = Web3.to_checksum_address(dest)
    except Exception:
        print("Invalid destination address.")
        return
    amt_s = input(f"{PROXY_TOKEN_SYMBOL} amount to withdraw (human units): ").strip()
    try:
        amt_human = float(amt_s)
    except ValueError:
        print("Invalid amount.")
        return
    if amt_human <= 0:
        print("Amount must be positive.")
        return
    amount_wei = int(amt_human * (10**PROXY_TOKEN_DECIMALS))

    w3 = get_polygon_web3()
    if not w3:
        print(
            f"Could not connect to {_polygon_rpc_url()!r} for calldata encoding; "
            "using offline Web3. Set POLYGON_RPC_URL if encoding fails."
        )
        w3 = Web3()
    try:
        data = _encode_erc20_transfer(w3, PROXY_TOKEN_POLYGON, dest_chk, amount_wei)
    except Exception as exc:
        print(f"Encode failed: {exc}")
        input("\nPress Enter to continue...")
        return

    txn = SafeTransaction(
        to=Web3.to_checksum_address(PROXY_TOKEN_POLYGON),
        operation=OperationType.Call,
        data=data,
        value="0",
    )
    print(f"\nWithdraw {amt_human} {PROXY_TOKEN_SYMBOL} to {dest_chk} via relayer?")
    if input("Confirm [y/N]: ").strip().lower() != "y":
        print("Cancelled.")
        input("\nPress Enter to continue...")
        return
    try:
        resp = client.execute([txn], f"Withdraw {PROXY_TOKEN_SYMBOL}")
        result = resp.wait()
        if result is not None:
            print("Withdraw succeeded.")
        else:
            print("Withdraw failed or timed out (check relayer / Polygon explorer).")
    except RelayerClientException as exc:
        print(f"Relayer error: {exc}")
        if getattr(exc, "status_code", None) == 401:
            print(
                "  Builder auth was rejected by the relayer. Try: Trading menu → "
                "Refresh Trading Session, then withdraw again. If it still fails, "
                "Almanac may need to align builder-sign with Polymarket’s relayer, or "
                "your account may need Builder Program access on Polymarket’s side."
            )
    except Exception as exc:
        print(f"Withdraw failed: {exc}")
    input("\nPress Enter to continue...")


def funds_menu() -> None:
    while True:
        print("\nFunds (balance / deposit / withdraw):")
        print("  1) Proxy wallet balance")
        print("  2) Deposit to proxy")
        print(f"  3) Withdraw {PROXY_TOKEN_SYMBOL} from proxy/safe")
        print("  4) Back to Trading Menu")
        c = input("\nEnter choice: ").strip()
        if c == "1":
            funds_show_balance()
        elif c == "2":
            funds_deposit_to_proxy()
        elif c == "3":
            funds_withdraw_usdc_e()
        elif c == "4":
            break
        else:
            print("Invalid choice. Enter 1–4.\n")


def _format_position_value(value, is_pnl=False, is_current_value=False):
    """Format a position value (price, size, etc.) for display."""
    if value is None:
        return "-"
    try:
        f = float(value)
        # Special handling for current_value = 0
        if is_current_value and f == 0:
            return "0"
        if is_pnl:
            # For P&L, show with sign
            sign = "+" if f >= 0 else ""
            return f"{sign}{f:.2f}"
        return f"{f:.2f}" if f >= 0.01 else f"{f:.6f}"
    except Exception:
        return str(value) if value is not None else "-"

def _truncate_text(text, max_length=50):
    """Truncate text to max_length, adding ellipsis if needed."""
    if not text:
        return "-"
    text_str = str(text)
    if len(text_str) <= max_length:
        return text_str
    return text_str[:max_length-3] + "..."

def _format_timestamp(timestamp):
    """Format timestamp to a shorter, readable format."""
    if not timestamp:
        return "-"
    try:
        # Try parsing ISO format
        if isinstance(timestamp, str):
            if 'T' in timestamp:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.strftime("%Y-%m-%d %H:%M")
        # Try parsing as timestamp
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M")
        return str(timestamp)
    except Exception:
        return str(timestamp)

def _display_positions(positions_data):
    """
    Display positions in a user-friendly format.
    
    Args:
        positions_data: Response data from fetch_positions (dict with 'data' key containing positions list)
    
    Returns:
        dict with pagination info: {'has_more': bool, 'offset': int, 'limit': int, 'total': int}
    """
    if not positions_data:
        print("No positions data received.")
        return None, []
    
    # Extract positions list from response
    positions = []
    pagination_info = None
    
    if isinstance(positions_data, dict):
        positions = positions_data.get("data", [])
        if not isinstance(positions, list):
            # Try other common response shapes
            positions = positions_data.get("positions", [])
            if not isinstance(positions, list):
                positions = []
        
        # Extract pagination info
        pagination = positions_data.get("pagination", {})
        if pagination:
            pagination_info = {
                "has_more": pagination.get("hasMore", False),
                "offset": pagination.get("offset", 0),
                "limit": pagination.get("limit", DEFAULT_POSITIONS_LIMIT),
                "total": pagination.get("total", len(positions))
            }
    elif isinstance(positions_data, list):
        positions = positions_data
    
    if not positions:
        print("\nNo positions found.")
        return pagination_info, []
    
    # Build table rows
    table_rows = []
    for idx, pos in enumerate(positions, start=1):
        # Market information
        market_title = pos.get("title") or "Unknown Market"
        
        # Outcome information
        outcome = pos.get("outcome") or "-"
        
        # Position details (all come as strings from API)
        size_str = pos.get("size") or "0"
        avg_price_str = pos.get("avg_price") or "0"
        initial_value_str = pos.get("initial_value") or "0"
        current_value_str = pos.get("current_value") or "0"
        
        # Convert strings to floats for formatting
        try:
            size = float(size_str)
        except (ValueError, TypeError):
            size = 0.0
        
        try:
            avg_price = float(avg_price_str)
        except (ValueError, TypeError):
            avg_price = 0.0
        
        try:
            initial_value = float(initial_value_str)
        except (ValueError, TypeError):
            initial_value = 0.0
        
        try:
            current_value = float(current_value_str)
        except (ValueError, TypeError):
            current_value = 0.0
        
        # P&L information (strings from API)
        cash_pnl_str = pos.get("cash_pnl") or "0"
        percent_pnl_str = pos.get("percent_pnl") or "0"
        
        try:
            cash_pnl = float(cash_pnl_str)
        except (ValueError, TypeError):
            cash_pnl = 0.0
        
        try:
            percent_pnl = float(percent_pnl_str)
        except (ValueError, TypeError):
            percent_pnl = 0.0
        
        # Status
        is_completed = pos.get("is_completed", False)
        status = "Completed" if is_completed else "Open"
        
        # Timestamps
        completed_at = pos.get("completed_at")
        
        # Build row with index as first column
        row = [
            str(idx),
            _truncate_text(market_title, 35),
            outcome,
            f"{size:.2f}",
            _format_position_value(avg_price),
            _format_position_value(initial_value),
            _format_position_value(current_value, is_current_value=True),
            _format_position_value(cash_pnl, is_pnl=True),
            f"{percent_pnl:.1f}%",
            status,
            _format_timestamp(completed_at) if completed_at else "-",
        ]
        table_rows.append(row)
    
    # Define table headers with # as first column
    headers = [
        "#",
        "Market",
        "Outcome",
        "Size",
        "Avg Price",
        "Initial Value",
        "Current Value",
        "P&L",
        "ROI",
        "Status",
        "Completed At"
    ]
    
    # Calculate summary statistics
    total_initial_value = sum(float(pos.get("initial_value") or "0") for pos in positions)
    total_current_value = sum(float(pos.get("current_value") or "0") for pos in positions)
    total_cash_pnl = sum(float(pos.get("cash_pnl") or "0") for pos in positions)
    completed_count = sum(1 for pos in positions if pos.get("is_completed", False))
    open_count = len(positions) - completed_count
    
    # Print table
    if pagination_info:
        total_display = pagination_info.get("total", len(positions))
        current_range_start = pagination_info.get("offset", 0) + 1
        current_range_end = pagination_info.get("offset", 0) + len(positions)
        print(f"\nPositions (showing {current_range_start}-{current_range_end} of {total_display}):")
    else:
        print(f"\nPositions ({len(positions)} found):")
    
    print(tabulate(table_rows, headers=headers, tablefmt="grid", stralign="left"))
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Total Positions: {len(positions)} ({open_count} Open, {completed_count} Completed)")
    print(f"Total Initial Value: {total_initial_value:.2f}")
    print(f"Total Current Value: {total_current_value:.2f}")
    print(f"Total P&L: {total_cash_pnl:+.2f}")
    if total_initial_value > 0:
        total_percent_pnl = (total_cash_pnl / total_initial_value) * 100
        print(f"Total ROI: {total_percent_pnl:+.1f}%")
    print("=" * 80)
    
    return pagination_info, positions

def _display_orders(orders_data):
    """
    Display orders in a user-friendly format.
    
    Args:
        orders_data: Response data from fetch_orders (dict with 'data' key containing orders list)
    
    Returns:
        tuple: (pagination_info dict, orders list)
    """
    if not orders_data:
        print("No orders data received.")
        return None, []
    
    # Extract orders list from response
    orders = []
    pagination_info = None
    
    if isinstance(orders_data, dict):
        orders = orders_data.get("data", [])
        if not isinstance(orders, list):
            # Try other common response shapes
            orders = orders_data.get("orders", [])
            if not isinstance(orders, list):
                orders = []
        
        # Extract pagination info
        pagination = orders_data.get("pagination", {})
        if pagination:
            pagination_info = {
                "has_more": pagination.get("hasMore", False),
                "offset": pagination.get("offset", 0),
                "limit": pagination.get("limit", DEFAULT_POSITIONS_LIMIT),
                "total": pagination.get("total", len(orders))
            }
    elif isinstance(orders_data, list):
        orders = orders_data
    
    if not orders:
        print("\nNo orders found.")
        return pagination_info, []
    
    # Build table rows
    table_rows = []
    for idx, order in enumerate(orders, start=1):
        # Market information - use marketQuestion from API
        market_question = order.get("marketQuestion") or order.get("title") or order.get("marketTitle") or "Unknown Market"
        
        # Outcome information
        outcome = order.get("outcome") or "-"
        
        # Side (BUY/SELL)
        side = order.get("side") or order.get("orderSide") or "-"
        if isinstance(side, str):
            side = side.upper()
        
        # Order details
        # API returns price and size as floats already, but handle both cases
        size_val = order.get("size") or order.get("quantity") or 0
        try:
            size = float(size_val)
        except (ValueError, TypeError):
            size = 0.0
        
        price_val = order.get("price") or 0
        try:
            price = float(price_val)
        except (ValueError, TypeError):
            price = 0.0
        
        order_type = order.get("orderType") or order.get("order_type") or order.get("type") or "GTC"
        
        # Status
        status = order.get("status") or order.get("orderStatus") or "unknown"
        if isinstance(status, str):
            status = status.capitalize()
        
        # Filled size (if partially filled) - API returns as float
        filled_size_val = order.get("filledSize") or order.get("filled_size") or order.get("filled") or 0
        try:
            filled_size = float(filled_size_val)
        except (ValueError, TypeError):
            filled_size = 0.0
        
        # Timestamps
        created_at = order.get("createdAt") or order.get("created_at") or order.get("timestamp")
        completed_at = order.get("completedAt") or order.get("completed_at")
        
        # Build row with index as first column
        row = [
            str(idx),
            _truncate_text(market_question, 35),
            outcome,
            side,
            f"{size:.2f}",
            _format_position_value(price),
            f"{filled_size:.2f}" if filled_size > 0 else "-",
            order_type,
            status,
            _format_timestamp(created_at),
            _format_timestamp(completed_at) if completed_at else "-",
        ]
        table_rows.append(row)
    
    # Define table headers
    headers = [
        "#",
        "Market",
        "Outcome",
        "Side",
        "Size",
        "Price",
        "Filled",
        "Type",
        "Status",
        "Created",
        "Completed At"
    ]
    
    # Calculate summary statistics using actual API status values
    status_counts = {}
    for o in orders:
        status = (o.get("status") or "").lower()
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Build summary string
    status_summary_parts = []
    for status_key in ["live", "delayed", "matched", "completed", "pending", "cancelled", "expired", "failed"]:
        count = status_counts.get(status_key, 0)
        if count > 0:
            status_summary_parts.append(f"{count} {status_key.capitalize()}")
    
    # Include any other statuses not in the standard list
    for status_key, count in status_counts.items():
        if status_key not in ["live", "delayed", "matched", "completed", "pending", "cancelled", "expired", "failed"]:
            if count > 0:
                status_summary_parts.append(f"{count} {status_key.capitalize()}")
    
    status_summary = ", ".join(status_summary_parts) if status_summary_parts else "0"
    
    # Print table
    if pagination_info:
        total_display = pagination_info.get("total", len(orders))
        current_range_start = pagination_info.get("offset", 0) + 1
        current_range_end = pagination_info.get("offset", 0) + len(orders)
        print(f"\nOrders (showing {current_range_start}-{current_range_end} of {total_display}):")
    else:
        print(f"\nOrders ({len(orders)} found):")
    
    print(tabulate(table_rows, headers=headers, tablefmt="grid", stralign="left"))
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Total Orders: {len(orders)} ({status_summary})")
    print("=" * 80)
    
    return pagination_info, orders

def _select_order_for_cancel(orders: list):
    """
    Allow user to select an order from the list and cancel it.
    
    Args:
        orders: List of order dicts
    """
    if not orders:
        print("No orders available.")
        return
    
    # Prompt for order selection
    print("\nCancel an order (or Enter to cancel):")
    try:
        choice = input("Order number: ").strip()
        if not choice:
            print("Cancelled.")
            return
        
        order_idx = int(choice) - 1
        if order_idx < 0 or order_idx >= len(orders):
            print("Invalid order number.")
            return
        
        selected_order = orders[order_idx]
        
        # Get order details
        order_id = selected_order.get("orderId") or selected_order.get("order_id") or selected_order.get("id")
        if not order_id:
            print("Order missing order ID.")
            return
        
        # Check if order can be cancelled (should be live or delayed)
        status = (selected_order.get("status") or "").lower()
        if status not in ("live", "delayed"):
            print(f"Cannot cancel order with status: {status}")
            print("Only live or delayed orders can be cancelled.")
            return
        
        # Show order details
        market_question = selected_order.get("marketQuestion") or "Unknown Market"
        outcome = selected_order.get("outcome") or "-"
        side = selected_order.get("side") or "-"
        size = selected_order.get("size") or 0
        price = selected_order.get("price") or 0
        
        print(f"\nSelected order:")
        print(f"  Market: {market_question}")
        print(f"  Outcome: {outcome}")
        print(f"  Side: {side}")
        print(f"  Size: {size}")
        print(f"  Price: {price}")
        print(f"  Status: {status}")
        
        # Confirm cancellation
        confirm = input("\nAre you sure you want to cancel this order? (y/n): ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Cancellation cancelled.")
            return
        
        # Cancel the order
        print("\nCancelling order...")
        result = cancel_order(order_id)
        
        if result:
            if result.get("success"):
                print("✓ Order cancelled successfully")
                if result.get("data"):
                    print(json.dumps(result.get("data"), indent=2))
            else:
                error_msg = result.get("error", "Unknown error")
                user_msg = result.get("userMessage", error_msg)
                print(f"✗ Error: {error_msg}")
                print(f"  Message: {user_msg}")
        else:
            print("Failed to cancel order.")
        
    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as exc:
        print(f"Error: {exc}")

def _position_to_market_dict(position: dict) -> dict:
    """
    Convert a position dict to a market dict format for use with order placement.
    
    Args:
        position: Position dict from API
        
    Returns:
        Market dict with required fields for order placement
    """
    # Try to get market ID - positions might have event_id which could be the market ID
    # or we might need to look it up. For now, use event_id as market_id.
    market_id = position.get("marketId") or position.get("market_id") or position.get("event_id") or position.get("eventId")
    
    market_dict = {
        "id": market_id,
        "title": position.get("title") or "Unknown Market",
        "neg_risk": position.get("neg_risk") or position.get("negRisk") or False,
    }
    return market_dict

def _select_position_for_trade(positions: list):
    """
    Allow user to select a position from the list and go to order flow.
    
    Args:
        positions: List of position dicts
    """
    if not positions:
        print("No positions available.")
        return
    
    # Prompt for position selection
    print("\nSell or Add to your position (or Enter to cancel):")
    try:
        choice = input("Position number: ").strip()
        if not choice:
            print("Cancelled.")
            return
        
        position_idx = int(choice) - 1
        if position_idx < 0 or position_idx >= len(positions):
            print("Invalid position number.")
            return
        
        selected_position = positions[position_idx]
        
        # Check if position is open
        is_completed = selected_position.get("is_completed", False)
        if is_completed:
            print("Cannot trade a completed position.")
            return
        
        # Get position details
        size_str = selected_position.get("size") or "0"
        outcome = selected_position.get("outcome") or None
        token_id = selected_position.get("token_id") or selected_position.get("tokenId")
        event_id = selected_position.get("event_id") or selected_position.get("eventId")
        
        try:
            position_size = float(size_str)
        except (ValueError, TypeError):
            print("Invalid position size.")
            return
        
        if position_size <= 0:
            print("Position size must be greater than 0.")
            return
        
        # Show position details
        market_title = selected_position.get("title") or "Unknown Market"
        print(f"\nSelected position:")
        print(f"  Market: {market_title}")
        print(f"  Outcome: {outcome}")
        print(f"  Current Size: {position_size:.2f}")
        
        # Fetch market details from API to get outcomes and clob_token_ids
        print("\nFetching latest market prices...")
        market_dict = None
        if event_id:
            try:
                event_resp = requests.get(f"{ALMANAC_API_URL}/markets/events/{event_id}", timeout=30)
                if event_resp.status_code == 200:
                    event_data = event_resp.json()
                    # Handle response wrapped in 'data' or direct event object
                    full_event = event_data.get("data") or event_data
                    markets = full_event.get("markets") or []
                    
                    # Find the market that matches this position (by token_id or outcome)
                    for market in markets:
                        market_token_ids = market.get("clob_token_ids") or []
                        if token_id and str(token_id) in [str(tid) for tid in market_token_ids]:
                            market_dict = market
                            break
                        # Fallback: use first market if we can't match by token
                        if not market_dict:
                            market_dict = market
                    
                    # If we found a market, fetch and display prices
                    if market_dict:
                        # Update market with latest prices from CLOB
                        market_dict = _update_market_prices_from_clob(market_dict)
                        
                        # Display outcomes with prices
                        outcomes = market_dict.get("outcomes") or []
                        outcome_prices = market_dict.get("outcome_prices") or []
                        clob_token_ids = market_dict.get("clob_token_ids") or []
                        clob_prices = market_dict.get("_clob_prices")
                        
                        if outcomes:
                            print("\nCurrent Market Prices:")
                            for idx, outcome_name in enumerate(outcomes):
                                outcome_name_str = outcome_name if isinstance(outcome_name, str) else (outcome_name.get("name") if isinstance(outcome_name, dict) else str(outcome_name))
                                price = None
                                if isinstance(outcome_prices, list) and idx < len(outcome_prices):
                                    price = outcome_prices[idx]
                                
                                # Show BUY/SELL prices if available
                                token_id_for_outcome = clob_token_ids[idx] if idx < len(clob_token_ids) else None
                                buy_price = None
                                sell_price = None
                                if clob_prices and token_id_for_outcome:
                                    token_id_str = str(token_id_for_outcome)
                                    token_price_data = clob_prices.get(token_id_str) or clob_prices.get(str(int(token_id_for_outcome)))
                                    if token_price_data:
                                        buy_price = token_price_data.get("BUY")
                                        sell_price = token_price_data.get("SELL")
                                
                                if price is not None:
                                    price_str = _format_price(price)
                                    if buy_price and sell_price:
                                        print(f"  {outcome_name_str}: {price_str} (Buy: {_format_price(buy_price)}, Sell: {_format_price(sell_price)})")
                                    else:
                                        print(f"  {outcome_name_str}: {price_str}")
                                else:
                                    print(f"  {outcome_name_str}: -")
                        else:
                            print("  (No outcomes found)")
                    else:
                        print("  (Could not fetch market details)")
            except Exception as exc:
                print(f"  (Could not fetch market prices: {exc})")
        
        # Convert position to market dict if we didn't get one from API
        if not market_dict:
            market_dict = _position_to_market_dict(selected_position)
            # Ensure we have a market ID - use event_id if market_id is not available
            if not market_dict.get("id") and event_id:
                market_dict["id"] = event_id
        
        # Go to order flow - user can choose buy (add to position) or sell
        # Pass available_shares for sell order validation
        _place_order_now(market_dict, chosen_outcome_name=outcome, chosen_token_id=token_id, available_shares=position_size)
        
    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as exc:
        print(f"Error: {exc}")

def _display_positions_summary(summary_data):
    """
    Display positions summary in a user-friendly format.
    
    Args:
        summary_data: Response data from fetch_positions_summary
    """
    if not summary_data:
        print("No summary data received.")
        return
    
    # Extract summary data from response
    summary = None
    if isinstance(summary_data, dict):
        if summary_data.get("success") and summary_data.get("data"):
            summary = summary_data.get("data")
        elif "data" in summary_data:
            summary = summary_data.get("data")
        else:
            summary = summary_data
    
    if not summary:
        print("\nNo summary data found.")
        return
    
    # Extract values (handle both camelCase and snake_case, and both string and numeric types)
    total_value = summary.get("totalValue") or summary.get("total_value") or 0
    total_volume = summary.get("totalVolume") or summary.get("total_volume") or 0
    total_pnl = summary.get("totalPnL") or summary.get("total_pnl") or 0
    total_pnl_percent = summary.get("totalPnLPercent") or summary.get("total_pnl_percent") or 0
    total_predictions = summary.get("totalPredictions") or summary.get("total_predictions") or 0
    completed_positions = summary.get("completedPositions") or summary.get("completed_positions") or 0
    winning_trades = summary.get("winningTrades") or summary.get("winning_trades") or 0
    win_rate = summary.get("winRate") or summary.get("win_rate") or 0
    
    # Convert to floats if strings
    try:
        total_value = float(total_value)
    except (ValueError, TypeError):
        total_value = 0.0
    
    try:
        total_volume = float(total_volume)
    except (ValueError, TypeError):
        total_volume = 0.0
    
    try:
        total_pnl = float(total_pnl)
    except (ValueError, TypeError):
        total_pnl = 0.0
    
    try:
        total_pnl_percent = float(total_pnl_percent)
    except (ValueError, TypeError):
        total_pnl_percent = 0.0
    
    try:
        total_predictions = int(total_predictions)
    except (ValueError, TypeError):
        total_predictions = 0
    
    try:
        completed_positions = int(completed_positions)
    except (ValueError, TypeError):
        completed_positions = 0
    
    try:
        winning_trades = int(winning_trades)
    except (ValueError, TypeError):
        winning_trades = 0
    
    try:
        win_rate = float(win_rate)
    except (ValueError, TypeError):
        win_rate = 0.0
    
    # Display summary
    print("\n" + "=" * 80)
    print("Positions Summary")
    print("=" * 80)
    
    # Build summary table
    summary_rows = [
        ["Total Value", f"{total_value:.2f}"],
        ["Total Volume", f"{total_volume:.2f}"],
        ["Total P&L", f"{total_pnl:+.2f}"],
        ["Total ROI", f"{total_pnl_percent:+.2f}%"],
        ["", ""],  # Separator
        ["Total Predictions", str(total_predictions)],
        ["Completed Positions", str(completed_positions)],
        ["Winning Trades", str(winning_trades)],
        ["Win Rate", f"{win_rate:.2f}%"],
    ]
    
    print(tabulate(summary_rows, headers=["Metric", "Value"], tablefmt="grid", stralign="left"))
    print("=" * 80)

def positions_menu():
    """
    Submenu for viewing positions: Open, Closed, or All.
    Supports pagination with navigation options.
    """
    current_filter = None
    current_offset = 0
    current_limit = DEFAULT_POSITIONS_LIMIT
    pagination_info = None
    exit_menu = False
    
    while True:
        if exit_menu:
            break
        if current_filter is None:
            # Main menu
            print("\nPositions Menu:")
            print("  1) Open Positions")
            print("  2) Closed Positions")
            print("  3) All Positions")
            print("  4) Positions Summary")
            print("  5) Back to Trading Menu")
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == "1":
                current_filter = "live"
                current_offset = 0
                filter_name = "open"
            elif choice == "2":
                current_filter = "closed"
                current_offset = 0
                filter_name = "closed"
            elif choice == "3":
                current_filter = "all"
                current_offset = 0
                filter_name = "all"
            elif choice == "4":
                print("\nFetching positions summary...")
                summary_data = fetch_positions_summary()
                _display_positions_summary(summary_data)
                # Return to menu after displaying summary
                input("\nPress Enter to continue...")
                continue
            elif choice == "5":
                break
            else:
                print("Invalid choice. Please enter a number from 1 to 5.\n")
                continue
            
            # Fetch and display positions
            print(f"\nFetching {filter_name} positions...")
            positions_data = fetch_positions(filter_type=current_filter, limit=current_limit, offset=current_offset)
            pagination_info, positions = _display_positions(positions_data)
            
            # For open positions, loop: prompt for selection, then show menu
            if current_filter == "live" and positions:
                while True:
                    _select_position_for_trade(positions)
                    # Show menu after order attempt
                    print("\nOptions:")
                    print("  1) Refresh Open Positions")
                    print("  2) Back to Positions Menu")
                    print("  3) Back to Trading Menu")
                    choice = input("\nEnter choice (1/2/3 or Enter to refresh): ").strip()
                    if choice == "2":
                        current_filter = None
                        current_offset = 0
                        pagination_info = None
                        break  # Exit the while loop, go back to main menu
                    elif choice == "3":
                        exit_menu = True  # Signal to exit positions_menu entirely
                        break  # Exit the while loop
                    else:
                        # Default: refresh positions and loop again to prompt for selection
                        print(f"\nRefreshing {filter_name} positions...")
                        positions_data = fetch_positions(filter_type=current_filter, limit=current_limit, offset=current_offset)
                        pagination_info, positions = _display_positions(positions_data)
                        if not positions:
                            # No positions left, go back to main menu
                            current_filter = None
                            break
                        # Loop will continue and prompt for selection again
                # After breaking from open positions loop, continue outer loop
                if current_filter is None:
                    continue
        else:
            # Pagination menu
            print("\nNavigation:")
            nav_options = []
            nav_num = 1
            
            if pagination_info and pagination_info.get("has_more"):
                nav_options.append(("next", "Next Page"))
                print(f"  {nav_num}) Next Page")
                nav_num += 1
            
            if current_offset > 0:
                nav_options.append(("prev", "Previous Page"))
                print(f"  {nav_num}) Previous Page")
                nav_num += 1
            
            nav_options.append(("refresh", "Refresh Current Page"))
            print(f"  {nav_num}) Refresh Current Page")
            nav_num += 1
            
            nav_options.append(("back", "Back to Positions Menu"))
            print(f"  {nav_num}) Back to Positions Menu")
            nav_num += 1
            
            nav_options.append(("main", "Back to Trading Menu"))
            print(f"  {nav_num}) Back to Trading Menu")
            
            choice = input("\nEnter choice: ").strip()
            
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(nav_options):
                    action, _ = nav_options[choice_num - 1]
                    
                    if action == "next":
                        current_offset += current_limit
                        print(f"\nFetching positions (offset {current_offset})...")
                        positions_data = fetch_positions(filter_type=current_filter, limit=current_limit, offset=current_offset)
                        pagination_info, positions = _display_positions(positions_data)
                        # For open positions, immediately prompt for position selection
                        if current_filter == "live" and positions:
                            _select_position_for_trade(positions)
                            # Show menu after order attempt
                            print("\nOptions:")
                            print("  1) Refresh Open Positions")
                            print("  2) Back to Positions Menu")
                            print("  3) Back to Trading Menu")
                            choice = input("\nEnter choice (1/2/3 or Enter to refresh): ").strip()
                            if choice == "2":
                                current_filter = None
                                current_offset = 0
                                pagination_info = None
                                continue
                            elif choice == "3":
                                break
                            else:
                                # Default: refresh positions
                                positions_data = fetch_positions(filter_type=current_filter, limit=current_limit, offset=current_offset)
                                pagination_info, positions = _display_positions(positions_data)
                    elif action == "prev":
                        current_offset = max(0, current_offset - current_limit)
                        print(f"\nFetching positions (offset {current_offset})...")
                        positions_data = fetch_positions(filter_type=current_filter, limit=current_limit, offset=current_offset)
                        pagination_info, positions = _display_positions(positions_data)
                        # For open positions, immediately prompt for position selection
                        if current_filter == "live" and positions:
                            _select_position_for_trade(positions)
                            # Show menu after order attempt
                            print("\nOptions:")
                            print("  1) Refresh Open Positions")
                            print("  2) Back to Positions Menu")
                            print("  3) Back to Trading Menu")
                            choice = input("\nEnter choice (1/2/3 or Enter to refresh): ").strip()
                            if choice == "2":
                                current_filter = None
                                current_offset = 0
                                pagination_info = None
                                continue
                            elif choice == "3":
                                break
                            else:
                                # Default: refresh positions
                                positions_data = fetch_positions(filter_type=current_filter, limit=current_limit, offset=current_offset)
                                pagination_info, positions = _display_positions(positions_data)
                    elif action == "refresh":
                        print(f"\nRefreshing positions (offset {current_offset})...")
                        positions_data = fetch_positions(filter_type=current_filter, limit=current_limit, offset=current_offset)
                        pagination_info, positions = _display_positions(positions_data)
                        # For open positions, immediately prompt for position selection
                        if current_filter == "live" and positions:
                            _select_position_for_trade(positions)
                            # Show menu after order attempt
                            print("\nOptions:")
                            print("  1) Refresh Open Positions")
                            print("  2) Back to Positions Menu")
                            print("  3) Back to Trading Menu")
                            choice = input("\nEnter choice (1/2/3 or Enter to refresh): ").strip()
                            if choice == "2":
                                current_filter = None
                                current_offset = 0
                                pagination_info = None
                                continue
                            elif choice == "3":
                                break
                            else:
                                # Default: refresh positions
                                positions_data = fetch_positions(filter_type=current_filter, limit=current_limit, offset=current_offset)
                                pagination_info, positions = _display_positions(positions_data)
                    elif action == "back":
                        current_filter = None
                        current_offset = 0
                        pagination_info = None
                    elif action == "main":
                        break
                else:
                    print("Invalid choice. Please enter a number from the menu.\n")
            except ValueError:
                print("Invalid input. Please enter a number.\n")

def orders_menu():
    """
    Submenu for viewing orders: Live, Delayed, Matched, Completed, or All.
    Supports pagination with navigation options.
    """
    current_status = None
    current_offset = 0
    current_limit = DEFAULT_POSITIONS_LIMIT
    pagination_info = None
    exit_menu = False
    
    while True:
        if exit_menu:
            break
        if current_status is None:
            # Main menu
            print("\nOrders Menu:")
            print("  1) Live Orders")
            print("  2) Delayed Orders")
            print("  3) Matched Orders")
            print("  4) Completed Orders")
            print("  5) All Orders")
            print("  6) Back to Trading Menu")
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == "1":
                current_status = "live"
                current_offset = 0
                status_name = "live"
            elif choice == "2":
                current_status = "delayed"
                current_offset = 0
                status_name = "delayed"
            elif choice == "3":
                current_status = "matched"
                current_offset = 0
                status_name = "matched"
            elif choice == "4":
                current_status = "completed"
                current_offset = 0
                status_name = "completed"
            elif choice == "5":
                current_status = None  # None means fetch all orders
                current_offset = 0
                status_name = "all"
            elif choice == "6":
                break
            else:
                print("Invalid choice. Please enter a number from 1 to 6.\n")
                continue
            
            # Fetch and display orders
            print(f"\nFetching {status_name} orders...")
            orders_data = fetch_orders(status=current_status, limit=current_limit, offset=current_offset)
            pagination_info, orders = _display_orders(orders_data)
            
            # For live and delayed orders, loop: prompt for selection, then show menu
            if current_status in ("live", "delayed") and orders:
                while True:
                    _select_order_for_cancel(orders)
                    # Show menu after cancel attempt
                    print("\nOptions:")
                    print("  1) Refresh Orders")
                    print("  2) Back to Orders Menu")
                    print("  3) Back to Trading Menu")
                    choice = input("\nEnter choice (1/2/3 or Enter to refresh): ").strip()
                    if choice == "2":
                        current_status = None
                        current_offset = 0
                        pagination_info = None
                        break  # Exit the while loop, go back to main menu
                    elif choice == "3":
                        exit_menu = True  # Signal to exit orders_menu entirely
                        break  # Exit the while loop
                    else:
                        # Default: refresh orders and loop again to prompt for selection
                        print(f"\nRefreshing {status_name} orders...")
                        orders_data = fetch_orders(status=current_status, limit=current_limit, offset=current_offset)
                        pagination_info, orders = _display_orders(orders_data)
                        if not orders:
                            # No orders left, go back to main menu
                            current_status = None
                            break
                        # Loop will continue and prompt for selection again
                # After breaking from live/delayed orders loop, continue outer loop
                if current_status is None:
                    continue
        else:
            # Pagination menu
            print("\nNavigation:")
            nav_options = []
            nav_num = 1
            
            if pagination_info and pagination_info.get("has_more"):
                nav_options.append(("next", "Next Page"))
                print(f"  {nav_num}) Next Page")
                nav_num += 1
            
            if current_offset > 0:
                nav_options.append(("prev", "Previous Page"))
                print(f"  {nav_num}) Previous Page")
                nav_num += 1
            
            nav_options.append(("refresh", "Refresh Current Page"))
            print(f"  {nav_num}) Refresh Current Page")
            nav_num += 1
            
            nav_options.append(("back", "Back to Orders Menu"))
            print(f"  {nav_num}) Back to Orders Menu")
            nav_num += 1
            
            nav_options.append(("main", "Back to Trading Menu"))
            print(f"  {nav_num}) Back to Trading Menu")
            
            choice = input("\nEnter choice: ").strip()
            
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(nav_options):
                    action, _ = nav_options[choice_num - 1]
                    
                    if action == "next":
                        current_offset += current_limit
                        print(f"\nFetching orders (offset {current_offset})...")
                        orders_data = fetch_orders(status=current_status, limit=current_limit, offset=current_offset)
                        pagination_info, orders = _display_orders(orders_data)
                        # For live and delayed orders, immediately prompt for order selection
                        if current_status in ("live", "delayed") and orders:
                            _select_order_for_cancel(orders)
                            # Show menu after cancel attempt
                            print("\nOptions:")
                            print("  1) Refresh Orders")
                            print("  2) Back to Orders Menu")
                            print("  3) Back to Trading Menu")
                            choice = input("\nEnter choice (1/2/3 or Enter to refresh): ").strip()
                            if choice == "2":
                                current_status = None
                                current_offset = 0
                                pagination_info = None
                                continue
                            elif choice == "3":
                                break
                            else:
                                # Default: refresh orders
                                orders_data = fetch_orders(status=current_status, limit=current_limit, offset=current_offset)
                                pagination_info, orders = _display_orders(orders_data)
                    elif action == "prev":
                        current_offset = max(0, current_offset - current_limit)
                        print(f"\nFetching orders (offset {current_offset})...")
                        orders_data = fetch_orders(status=current_status, limit=current_limit, offset=current_offset)
                        pagination_info, orders = _display_orders(orders_data)
                        # For live and delayed orders, immediately prompt for order selection
                        if current_status in ("live", "delayed") and orders:
                            _select_order_for_cancel(orders)
                            # Show menu after cancel attempt
                            print("\nOptions:")
                            print("  1) Refresh Orders")
                            print("  2) Back to Orders Menu")
                            print("  3) Back to Trading Menu")
                            choice = input("\nEnter choice (1/2/3 or Enter to refresh): ").strip()
                            if choice == "2":
                                current_status = None
                                current_offset = 0
                                pagination_info = None
                                continue
                            elif choice == "3":
                                break
                            else:
                                # Default: refresh orders
                                orders_data = fetch_orders(status=current_status, limit=current_limit, offset=current_offset)
                                pagination_info, orders = _display_orders(orders_data)
                    elif action == "refresh":
                        print(f"\nRefreshing orders (offset {current_offset})...")
                        orders_data = fetch_orders(status=current_status, limit=current_limit, offset=current_offset)
                        pagination_info, orders = _display_orders(orders_data)
                        # For live and delayed orders, immediately prompt for order selection
                        if current_status in ("live", "delayed") and orders:
                            _select_order_for_cancel(orders)
                            # Show menu after cancel attempt
                            print("\nOptions:")
                            print("  1) Refresh Orders")
                            print("  2) Back to Orders Menu")
                            print("  3) Back to Trading Menu")
                            choice = input("\nEnter choice (1/2/3 or Enter to refresh): ").strip()
                            if choice == "2":
                                current_status = None
                                current_offset = 0
                                pagination_info = None
                                continue
                            elif choice == "3":
                                break
                            else:
                                # Default: refresh orders
                                orders_data = fetch_orders(status=current_status, limit=current_limit, offset=current_offset)
                                pagination_info, orders = _display_orders(orders_data)
                    elif action == "back":
                        current_status = None
                        current_offset = 0
                        pagination_info = None
                    elif action == "main":
                        break
                else:
                    print("Invalid choice. Please enter a number from the menu.\n")
            except ValueError:
                print("Invalid input. Please enter a number.\n")

def start_trading_flow():
    """
    Submenu for trading: auto-creates session if needed, search markets, place orders.
    Keeps state in CURRENT_SESSION and SELECTED_MARKET.
    """
    global CURRENT_SESSION, SELECTED_MARKET
    
    # Check if account exists before creating session
    wallet_address = _get_credential("EOA_WALLET_ADDRESS")
    if wallet_address:
        account_exists = check_account_exists(wallet_address)
        if account_exists is False:
            print("\n✗ Account not found")
            print("="*60)
            print("You need to complete your account creation on Almanac first.")
            print("Please visit https://almanac.market to create your account,")
            print("then try trading again.")
            print("="*60)
            return
        elif account_exists is None:
            # Error checking account, but proceed anyway
            pass
    
    # Auto-create session if none exists
    if not CURRENT_SESSION:
        print("\nNo active trading session detected. Creating one now...")
        try:
            session = initiate_trading_session()
            if session:
                print("Trading session created successfully.")
                CURRENT_SESSION = session
            else:
                print("Failed to create trading session. Please check your configuration.")
                return
        except Exception as exc:
            print(f"Failed to create trading session: {exc}")
            return
    
    while True:
        print("\nTrading Menu:")
        print("  1) Search and Trade Markets")
        print("  2) See Positions")
        print("  3) See Orders")
        print("  4) Claim Polymarket Proceeds")
        print("  5) Funds (balance / deposit / withdraw)")
        print("  6) Refresh Trading Session")
        print("  7) Back to Main Menu")
        choice = input("\nEnter choice: ").strip()

        if choice == "1":
            search_markets()
        elif choice == "2":
            positions_menu()
        elif choice == "3":
            orders_menu()
        elif choice == "4":
            claim_proceeds_menu()
        elif choice == "5":
            funds_menu()
        elif choice == "6":
            try:
                session = initiate_trading_session()
                if session:
                    print("\nTrading session refreshed.")
                    CURRENT_SESSION = session
                else:
                    print("Trading session could not be refreshed.")
            except Exception as exc:
                print(f"Failed to refresh trading session: {exc}")
        elif choice == "7":
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 7.\n")

def initiate_trading_session():
    """
    Initiate a trading session with the Almanac API.

    {
        "signature": "0x...",
        "message": "Create Almanac trading session",  # human-readable action text
        "walletAddress": "0x...",
        "apiCredentials": {
            "apiKey": "string",
            "secret": "base64-string",
            "passphrase": "string"
        },
        "userAgent": "optional string"
    }

    Returns:
    {
        'success': True,
        'data': {
            'sessionId': '...',
            'walletAddress': '0x...',
            'proxyWallet': '0x...',
            'expiresAt': '2025-11-17T15:10:50.983Z'
        },
        'timestamp': '2025-11-16T15:10:50.983Z'
    }
    """
    load_dotenv(dotenv_path=str(ENV_PATH))

    # Load wallet address and private key
    wallet_address = _get_credential("EOA_WALLET_ADDRESS")
    if not wallet_address:
        print(f"EOA_WALLET_ADDRESS not found in {ENV_PATH}. Please set it and try again.")
        return
    
    private_key = _get_credential("EOA_WALLET_PK")
    if not private_key:
        print(f"EOA_WALLET_PK not found in {ENV_PATH}. Please set it and try again.")
        return
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key
    
    # Validate address derives cleanly (optional)
    try:
        addr = Account.from_key(private_key).address.lower()
        if addr != wallet_address.lower():
            print(f"Private key does not match wallet address: {addr} != {wallet_address}")
            return
    except Exception as exc:
        print(f"Invalid private key: {exc}")
        return

    # Prepare EIP-191 message (personal_sign). Include nonce/timestamp to prevent replay.
    action = "Create Almanac trading session"
    nonce = secrets.token_hex(16)
    timestamp = int(time.time())
    #message = f"{action}\nwallet:{wallet_address}\nchainId:{POLYGON_CHAIN_ID}\nnonce:{nonce}\ntimestamp:{timestamp}"
    message = action
    msg = encode_defunct(text=message)
    signed = Account.from_key(private_key).sign_message(msg)
    signature = signed.signature.hex() if hasattr(signed.signature, "hex") else signed.signature
    if not isinstance(signature, str):
        signature = str(signature)
    if not signature.startswith("0x"):
        signature = "0x" + signature

    api_keys = {
        "apiKey": _get_credential("POLYMARKET_API_KEY"),
        "secret": _get_credential("POLYMARKET_API_SECRET"),
        "passphrase": _get_credential("POLYMARKET_API_PASSPHRASE")
    }

    response = requests.post(f'{ALMANAC_API_URL}/v1/trading/sessions', 
        headers={'Content-Type': 'application/json'},
        json={
        'signature': signature,
        'message': message,  # full message that was signed (contains action+nonce+timestamp)
        'walletAddress': wallet_address,
        'nonce': nonce,
        'timestamp': timestamp,
        'apiCredentials': api_keys
    })
    if response.status_code != 200:
        print(f"Failed to create trading session:")
        print(json.dumps(response.json(), indent=2))
        return None
    return response.json()


def _extract_signed_order_dict(signed_order) -> dict:
    """
    Normalize SDK signed-order objects to a plain dict for Almanac API payloads.
    """
    if isinstance(signed_order, dict):
        data = dict(signed_order)
    elif hasattr(signed_order, "model_dump"):
        data = signed_order.model_dump()  # pydantic v2
    elif hasattr(signed_order, "dict"):
        data = signed_order.dict()  # pydantic v1
    elif hasattr(signed_order, "__dict__"):
        data = dict(signed_order.__dict__)
    else:
        raise TypeError("Unsupported signed order type from py-clob-client-v2")

    # Convert enum-like values from SDK models into JSON-friendly primitives.
    for key, value in list(data.items()):
        if hasattr(value, "value"):
            data[key] = value.value

    # Our backend expects side as BUY/SELL in the wire payload.
    side_val = data.get("side")
    if isinstance(side_val, str):
        upper = side_val.upper()
        data["side"] = "BUY" if upper.endswith("BUY") else ("SELL" if upper.endswith("SELL") else upper)
    elif isinstance(side_val, int):
        data["side"] = "BUY" if side_val == 0 else "SELL"

    return data


def _normalize_sell_size_for_clob(size: float) -> float:
    """
    Polymarket-style orders encode sell `makerAmount` from round(size, 2) * 1e6.
    A position like 5.5555 rounds to 5.56 on-chain and can exceed the wallet → INSUFFICIENT_BALANCE.
    Floor to the 0.01-share grid in micro-units so the signed amount never exceeds `size`.
    """
    micro = math.floor(float(size) * 1_000_000 + 1e-6)
    aligned = (micro // 10_000) * 10_000
    return aligned / 1_000_000


def place_order(
    market_id: str,
    side_upper: str,
    size: float,
    price: float,
    neg_risk: bool = False,
    order_type: str = "GTC",
    chosen_outcome_name: str | None = None,
    chosen_token_id: str | None = None,
):
    """
    Place an order for the selected market.

    Expected headers (if supported by server):
      - x-session-id
      - x-wallet-address

    Request body (signed flow — must match Almanac + CLOB v2 signed order shape):
      - marketId, orderType, userWalletAddress
      - signedOrder: v2 order fields from py-clob-client-v2 (includes timestamp/metadata/builder)
      All order fields and signature are on the SAME object (not signedOrder.orderPayload).
    """
    global CURRENT_SESSION
    if not CURRENT_SESSION:
        print("No active trading session. Create a session first.")
        return

    session_id = (
        CURRENT_SESSION.get("data").get("sessionId")
    )
    wallet_address = (
        CURRENT_SESSION.get("data").get("walletAddress")
        or _get_credential("EOA_WALLET_ADDRESS")
    )
    proxy_address = (
        CURRENT_SESSION.get("data").get("proxyWallet")
        or _get_credential("EOA_PROXY_FUNDER")
    )

    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["x-session-id"] = session_id
    if wallet_address:
        headers["x-wallet-address"] = wallet_address

    if side_upper == "SELL":
        size = _normalize_sell_size_for_clob(size)
        if size <= 0:
            print("Sell size is below 0.01 shares after aligning to the exchange grid. Nothing to sell.")
            return

    private_key = _get_credential("EOA_WALLET_PK")
    if not private_key:
        print("EOA_WALLET_PK is required to sign CLOB v2 orders.")
        return
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    if chosen_token_id is None:
        print("No token ID selected for this order. Please choose a market outcome first.")
        return

    token_id = str(chosen_token_id)
    if order_type in ("FOK", "FAK"):
        adjusted_price = price + PRICE_BUFFER_ADJUSTMENT if side_upper == "BUY" else max(0.01, price - PRICE_BUFFER_ADJUSTMENT)
    else:
        adjusted_price = price

    use_proxy_funder = bool(proxy_address and wallet_address and proxy_address.lower() != wallet_address.lower())
    signature_type_candidates = [1, 2] if use_proxy_funder else [None]
    last_response = None

    for idx, sig_type in enumerate(signature_type_candidates):
        signed_flow_payload = None
        try:
            client_kwargs = {
                "host": POLYMARKET_CLOB_HOST,
                "chain_id": POLYGON_CHAIN_ID,
                "key": private_key,
            }
            if use_proxy_funder:
                client_kwargs["funder"] = proxy_address
                if sig_type is not None:
                    client_kwargs["signature_type"] = sig_type

            client = ClobClient(**client_kwargs)

            tick_size = "0.01"
            try:
                dynamic_tick_size = client.get_tick_size(token_id)
                if dynamic_tick_size:
                    tick_size = str(dynamic_tick_size)
            except Exception:
                pass

            try:
                dynamic_neg_risk = client.get_neg_risk(token_id)
                if isinstance(dynamic_neg_risk, bool):
                    neg_risk = dynamic_neg_risk
            except Exception:
                pass

            options = PartialCreateOrderOptions(tick_size=tick_size, neg_risk=bool(neg_risk))
            if order_type in ("FOK", "FAK"):
                # CLOB v2 market orders require BUY amount in USDC (2dp max) and enforce stricter precision.
                market_amount = round(float(size) * float(adjusted_price), 2) if side_upper == "BUY" else round(float(size), 4)
                market_order_args = MarketOrderArgs(
                    token_id=token_id,
                    amount=market_amount,
                    side=BUY if side_upper == "BUY" else SELL,
                    price=float(adjusted_price),
                    order_type=OrderType.FOK if order_type == "FOK" else OrderType.FAK,
                )
                signed_order = client.create_market_order(order_args=market_order_args, options=options)
            else:
                order_args = OrderArgs(
                    token_id=token_id,
                    price=float(adjusted_price),
                    size=float(size),
                    side=BUY if side_upper == "BUY" else SELL,
                )
                signed_order = client.create_order(order_args=order_args, options=options)
            order_fields_for_api = _extract_signed_order_dict(signed_order)

            for key in ("salt", "tokenId", "makerAmount", "takerAmount", "expiration", "timestamp"):
                if key in order_fields_for_api and order_fields_for_api[key] is not None:
                    order_fields_for_api[key] = str(order_fields_for_api[key])
            for key in ("metadata", "builder"):
                if key in order_fields_for_api and isinstance(order_fields_for_api[key], bytes):
                    order_fields_for_api[key] = "0x" + order_fields_for_api[key].hex()

            signed_flow_payload = {
                "marketId": market_id,
                "signedOrder": order_fields_for_api,
                "orderType": order_type,
                "userWalletAddress": wallet_address,
            }
        except Exception:
            signed_flow_payload = None
            import traceback
            traceback.print_exc()

        if signed_flow_payload is None:
            if idx == len(signature_type_candidates) - 1:
                print("Failed to build signed order payload. Aborting without sending.")
                return
            continue

        try:
            resp = requests.post(
                f"{ALMANAC_API_URL}/v1/trading/orders",
                headers=headers,
                json=signed_flow_payload,
                timeout=30,
            )
            last_response = resp
        except Exception as exc:
            if idx == len(signature_type_candidates) - 1:
                print(f"Order error: {exc}")
                return
            continue

        if resp.status_code == 200:
            print("Order placed:")
            print(json.dumps(resp.json(), indent=2))
            return

        # Some backends still verify proxy orders using signatureType=2; retry once.
        should_retry_invalid_sig = False
        try:
            body = resp.json() or {}
            details = body.get("details") if isinstance(body, dict) else {}
            error_text = str((details or {}).get("error") or body.get("error") or "").lower()
            should_retry_invalid_sig = "invalid signature" in error_text
        except Exception:
            should_retry_invalid_sig = False

        if should_retry_invalid_sig and idx < len(signature_type_candidates) - 1:
            continue

        print("Failed to place order:")
        try:
            print(json.dumps(resp.json(), indent=2))
        except Exception:
            print(resp.text)
        return

    if last_response is not None:
        print("Failed to place order:")
        try:
            print(json.dumps(last_response.json(), indent=2))
        except Exception:
            print(last_response.text)

def search_markets():
    """
    Prompt for query, fetch events, let user pick an event, then a market,
    then optionally an outcome, and optionally place an order immediately.
    Also stores the chosen market into SELECTED_MARKET for later use if desired.
    """
    global SELECTED_MARKET
    query = input("\nEnter market search query: ").strip()
    if not query:
        print("Empty query. Cancelled.")
        return
    try:
        resp = requests.get(
            f"{ALMANAC_API_URL}/markets/search",
            params={"q": query, "limit": 10},
            timeout=30,
        )
        if resp.status_code != 200:
            print("Search failed:")
            try:
                print(json.dumps(resp.json(), indent=2))
            except Exception:
                print(resp.text)
            return
        payload = resp.json() or []
        results = _normalize_search_results(payload)
        events = _extract_event_list(results)
        
        # Inject debug market if DEBUG is enabled
        if DEBUG:
            debug_event = {
                "id": "debug-event",
                "title": "[DEBUG] Non-Eligible Market Test",
                "name": "[DEBUG] Non-Eligible Market Test",
                "markets": [DEBUG_STATIC_MARKET]
            }
            events = [debug_event] + events
        
        if not events:
            print("No events found. Please try again with a different query.")
            return
        print("\nSearch results (Events):")
        for idx, ev in enumerate(events, start=1):
            ev_title = ev.get("title") or ev.get("question") or ev.get("name") or "Untitled Event"
            ev_id = ev.get("id") or ev.get("eventId") or ev.get("_id") or "unknown"
            
            # Check if event has "Games" tag (tags are list of dicts with 'label' field)
            tags = ev.get("tags", [])
            has_games_tag = False
            if isinstance(tags, list):
                has_games_tag = any(
                    isinstance(tag, dict) and tag.get("label") == "Games"
                    for tag in tags
                )
            
            # Check markets for game_start_time (it's on markets, not events)
            game_start_time = None
            markets = ev.get("markets", [])
            if isinstance(markets, list) and len(markets) > 0:
                # Get game_start_time from first market (they should all have the same time)
                first_market = markets[0]
                game_start_time = first_market.get("game_start_time")
            
            # Format event title with game start time if available
            if has_games_tag and game_start_time:
                formatted_time = _format_game_start_time(game_start_time)
                if formatted_time:
                    ev_title = f"{ev_title} -- {formatted_time}"
            
            print(f"  {idx}) {ev_title} [{ev_id}]")
        sel = input("\nChoose an event by number (or Enter to cancel): ").strip()
        if not sel:
            print("Cancelled.")
            return
        try:
            sel_idx = int(sel)
        except ValueError:
            print("Invalid selection.")
            return
        if sel_idx < 1 or sel_idx > len(events):
            print("Selection out of range.")
            return
        chosen_event = events[sel_idx - 1]
        
        # Fetch full event details from API to get child events and complete market data
        event_id = chosen_event.get("id") or chosen_event.get("eventId") or chosen_event.get("_id")
        if event_id:
            try:
                event_resp = requests.get(f"{ALMANAC_API_URL}/markets/events/{event_id}", timeout=30)
                if event_resp.status_code == 200:
                    event_data = event_resp.json()
                    # Use the full event data (may include child events and more complete market data)
                    if isinstance(event_data, dict):
                        # Handle response wrapped in 'data' or direct event object
                        full_event = event_data.get("data") or event_data
                        chosen_event = full_event
                    else:
                        print("⚠ Unexpected event data format")
                else:
                    print(f"⚠ Could not fetch full event details (status {event_resp.status_code}), using search result data")
            except Exception as exc:
                print(f"⚠ Could not fetch full event details: {exc}, using search result data")
        
        # Get markets from event before displaying
        markets = chosen_event.get("markets") or []
        
        # Check for child events and include their markets
        child_events = chosen_event.get("childEvents") or chosen_event.get("child_events") or chosen_event.get("children")
        if child_events and isinstance(child_events, list):
            for child_event in child_events:
                child_markets = child_event.get("markets") or []
                if child_markets:
                    markets = markets + child_markets
        
        # Check for parent event and include its markets
        parent_event_id = chosen_event.get("parentEventId") or chosen_event.get("parent_event_id")        
        if parent_event_id:
            try:
                parent_resp = requests.get(f"{ALMANAC_API_URL}/events/{parent_event_id}", timeout=30)
                if parent_resp.status_code == 200:
                    parent_event_data = parent_resp.json()
                    # Handle response wrapped in 'data' or direct event object
                    parent_event = parent_event_data.get("data") or parent_event_data
                    parent_markets = parent_event.get("markets") or []
                    if parent_markets:
                        markets = markets + parent_markets
                        print(f"\nNote: Found {len(parent_markets)} additional markets from parent event")
            except Exception as exc:
                print(f"Warning: Could not fetch parent event: {exc}")
        
        if not isinstance(markets, list) or not markets:
            print("No markets found for this event.")
            return
        
        # Batch fetch latest prices from CLOB API for all markets in the event
        print("\nFetching latest prices from CLOB API for all markets...")
        markets = _update_all_markets_prices_from_clob(markets)
        clob_fetch_successful = any(m.get("_clob_prices") for m in markets)
        if clob_fetch_successful:
            print("✓ Latest prices fetched successfully")
        else:
            print("⚠ Could not fetch latest prices (using cached prices)")
        
        # Now display the markets with updated prices
        markets = _display_markets_for_event({"markets": markets})
        if not markets:
            return
        
        # Auto-select if only one market, otherwise prompt
        if len(markets) == 1:
            SELECTED_MARKET = markets[0]
            title = SELECTED_MARKET.get("title") or SELECTED_MARKET.get("question") or SELECTED_MARKET.get("name") or "Untitled"
            market_id = SELECTED_MARKET.get("id") or SELECTED_MARKET.get("marketId") or SELECTED_MARKET.get("_id") or "unknown"
            print(f"\nSelected market: {title} [{market_id}]")
        else:
            sel_m = input("\nChoose a market by number (or Enter to cancel): ").strip()
            if not sel_m:
                print("Cancelled.")
                return
            try:
                sel_m_idx = int(sel_m)
            except ValueError:
                print("Invalid selection.")
                return
            if sel_m_idx < 1 or sel_m_idx > len(markets):
                print("Selection out of range.")
                return
            SELECTED_MARKET = markets[sel_m_idx - 1]
            title = SELECTED_MARKET.get("title") or SELECTED_MARKET.get("question") or SELECTED_MARKET.get("name") or "Untitled"
            market_id = SELECTED_MARKET.get("id") or SELECTED_MARKET.get("marketId") or SELECTED_MARKET.get("_id") or "unknown"
            print(f"\nSelected market: {title} [{market_id}]")

        # Prices have already been fetched for all markets when the event was selected
        # No need to fetch again here - the selected market already has updated prices

        result = _display_outcomes_and_choose(SELECTED_MARKET)
        if result is None:
            print("Cancelled.")
            return
        chosen_outcome_name, chosen_outcome_price, chosen_token_id = result
        # Go straight to placing an order (single confirmation flow is order inputs)
        _place_order_now(SELECTED_MARKET, chosen_outcome_name, chosen_token_id)
    except Exception as exc:
        print(f"Search error: {exc}")

def _prompt_yes_no(prompt: str) -> bool:
    while True:
        choice = input(f"{prompt} [y/n]: ").strip().lower()
        if choice in ("y", "yes"):
            return True
        if choice in ("n", "no"):
            return False
        print("Please answer with 'y' or 'n'.")

def _display_credentials(credentials) -> None:
    print("\nYour Polymarket API credentials:")
    # Convenient copy-paste .env format
    print("Copy as .env lines:")
    print(f"POLYMARKET_API_KEY={credentials.api_key}")
    print(f"POLYMARKET_API_SECRET={credentials.api_secret}")
    print(f"POLYMARKET_API_PASSPHRASE={credentials.api_passphrase}")
    print("")

def initiate_wallet_session():
    """
    Initiate a wallet session with the Almanac API.
    
    Returns:
    {
        'success': True,
        'data': {
            'sessionId': '...',
            'address': '0x...',
            ...
        },
        'timestamp': '...'
    }
    """
    load_dotenv(dotenv_path=str(ENV_PATH))

    # Load wallet address and private key
    wallet_address = _get_credential("EOA_WALLET_ADDRESS")
    if not wallet_address:
        print(f"EOA_WALLET_ADDRESS not found in {ENV_PATH}. Please set it and try again.")
        return None
    
    private_key = _get_credential("EOA_WALLET_PK")
    if not private_key:
        print(f"EOA_WALLET_PK not found in {ENV_PATH}. Please set it and try again.")
        return None
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key
    
    # Validate address derives cleanly (optional)
    try:
        addr = Account.from_key(private_key).address.lower()
        if addr != wallet_address.lower():
            print(f"Private key does not match wallet address: {addr} != {wallet_address}")
            return None
    except Exception as exc:
        print(f"Invalid private key: {exc}")
        return None

    # Prepare EIP-191 message (personal_sign)
    message = "Create Almanac wallet session"
    msg = encode_defunct(text=message)
    signed = Account.from_key(private_key).sign_message(msg)
    signature = signed.signature.hex() if hasattr(signed.signature, "hex") else signed.signature
    if not isinstance(signature, str):
        signature = str(signature)
    if not signature.startswith("0x"):
        signature = "0x" + signature

    response = requests.post(
        f'{ALMANAC_API_URL}/wallet/session',
        headers={'Content-Type': 'application/json'},
        json={
            'signature': signature,
            'message': message,
            'walletAddress': wallet_address,
            'userAgent': 'Python/1.0'
        }
    )
    if response.status_code != 200:
        print(f"Failed to create wallet session:")
        try:
            print(json.dumps(response.json(), indent=2))
        except Exception:
            print(response.text)
        return None
    return response.json()

def unlink_bittensor_hotkey(session_id: str, wallet_address: str):
    """
    Unlink the Bittensor hotkey from the Almanac account.
    
    Args:
        session_id: The wallet session ID
        wallet_address: The wallet address
    """
    print("\nUnlinking Bittensor hotkey...")
    
    # Show disclaimer
    print("\n" + "="*60)
    print("IMPORTANT NOTE:")
    print("="*60)
    print("Unlinking your Bittensor hotkey will:")
    print("- Your trading activity will no longer be tracked for this miner")
    print("- You'll no longer be eligible for subnet alpha rewards")
    print("- This unlinks your EOA wallet from your Bittensor identity")
    print("\nYou will need to link a hotkey again to be eligible for subnet alpha rewards.")
    print("="*60)
    
    confirm = input("\nAre you sure you want to unlink your Bittensor hotkey? (yes/no): ").strip().lower()
    if confirm not in ("yes", "y"):
        print("Unlinking cancelled.")
        return
    
    try:
        response = requests.post(
            f"{ALMANAC_API_URL}/subnet/unlink-hotkey",
            headers={
                "x-session-id": session_id,
                "x-wallet-address": wallet_address,
                "Content-Type": "application/json"
            },
            timeout=30
        )
        
        if response.status_code != 200:
            print("Failed to unlink hotkey:")
            try:
                print(json.dumps(response.json(), indent=2))
            except Exception:
                print(response.text)
            return
        
        data = response.json()
        if data.get('success'):
            print("✓ Bittensor integration unlinked")
        else:
            error_msg = data.get('error', 'Unknown error')
            user_msg = data.get('userMessage', error_msg)
            print(f"✗ Error: {error_msg}")
            print(f"  Message: {user_msg}")
    except Exception as exc:
        print(f"Failed to unlink hotkey: {exc}")
        import traceback
        traceback.print_exc()

def check_account_exists(wallet_address: str) -> bool:
    """
    Check if an account exists for the given wallet address.
    
    Returns:
        True if account exists, False if 404 (account doesn't exist), None if other error
    """
    try:
        response = requests.get(
            f"{ALMANAC_API_URL}/accounts/{wallet_address}",
            headers={
                "Content-Type": "application/json"
            },
            timeout=30
        )
        
        if response.status_code == 404:
            return False  # Account doesn't exist
        elif response.status_code == 200:
            return True  # Account exists
        else:
            return None  # Other error
    except Exception:
        return None  # Error occurred

def check_account_bittensor_status(wallet_address: str):
    """
    Check if an account already has a linked Bittensor hotkey.
    
    Returns:
        dict with 'has_link' (bool), 'hotkey' (str or None), 'uid' (int or None), 'error' (str or None)
    """
    try:
        response = requests.get(
            f"{ALMANAC_API_URL}/accounts/{wallet_address}",
            headers={
                "Content-Type": "application/json"
            },
            timeout=30
        )
        
        if response.status_code == 404:
            return {"has_link": False, "hotkey": None, "uid": None, "error": "account_not_found"}
        
        if response.status_code != 200:
            return {"has_link": False, "hotkey": None, "uid": None, "error": "Account lookup failed"}
        
        data = response.json()
        if data.get('success') and data.get('data'):
            account = data['data']
            bittensor_hotkey = account.get('bittensor_hotkey')
            bittensor_uid = account.get('bittensor_uid')
            
            if bittensor_hotkey:
                return {
                    "has_link": True,
                    "hotkey": bittensor_hotkey,
                    "uid": bittensor_uid
                }
            else:
                return {"has_link": False, "hotkey": None, "uid": None}
        else:
            return {"has_link": False, "hotkey": None, "uid": None, "error": "Account not found"}
    except Exception as exc:
        return {"has_link": False, "hotkey": None, "uid": None, "error": str(exc)}

def link_bittensor_uid():
    """
    Link Bittensor hotkey/uid to Almanac account.
    Steps:
    1. Create a wallet session
    2. Check existing Bittensor link status
    3. Sign hotkey address with Bittensor keypair
    4. Link hotkey to account via API
    """
    print("\nLinking Bittensor UID to Almanac account...")
    
    # Step 1: Create wallet session
    print("\nStep 1: Creating wallet session...")
    try:
        session_response = initiate_wallet_session()
        if not session_response:
            print("Failed to create wallet session. Please check your configuration.")
            return
        
        session_data = session_response.get("data", {})
        session_id = session_data.get("sessionId")
        wallet_address = session_data.get("address")
        
        if not session_id or not wallet_address:
            print("Wallet session created but missing required data.")
            print(json.dumps(session_response, indent=2))
            return
        
        print(f"✓ Wallet session created successfully")
        print(f"  Wallet Address: {wallet_address}")
    except Exception as exc:
        print(f"Failed to create wallet session: {exc}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Check existing Bittensor link status
    print("\nStep 2: Checking existing Bittensor link status...")
    status = check_account_bittensor_status(wallet_address)
    
    if status.get("error") == "account_not_found":
        print("\n✗ Account not found")
        print("="*60)
        print("You need to complete your account creation on Almanac first.")
        print("Please visit https://almanac.market to create your account,")
        print("then try linking your Bittensor UID again.")
        print("="*60)
        return
    
    if status.get("error"):
        print(f"⚠ Warning: Could not check account status: {status['error']}")
        print("Proceeding with linking...")
    elif status.get("has_link"):
        print("✓ Bittensor integration found")
        print(f"  Hotkey: {status['hotkey']}")
        if status.get('uid') is not None:
            print(f"  UID: {status['uid']}")
        
        print("\nOptions:")
        print("  1) Link a different hotkey")
        print("  2) Unlink current hotkey")
        print("  3) Cancel")
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == "1":
            # Proceed with linking a different hotkey
            pass
        elif choice == "2":
            # Unlink the current hotkey
            unlink_bittensor_hotkey(session_id, wallet_address)
            return
        else:
            print("Cancelled.")
            return
    else:
        print("No Bittensor integration found. Proceeding with linking...")
    
    # Step 3: Sign hotkey with Bittensor keypair
    print("\nStep 3: Signing hotkey with Bittensor keypair...")
    
    # Prompt for wallet name and hotkey name
    wallet_name = input("Enter Bittensor wallet name: ").strip()
    if not wallet_name:
        print("Wallet name is required. Cancelled.")
        return
    
    hotkey_name = input("Enter Bittensor hotkey name: ").strip()
    if not hotkey_name:
        print("Hotkey name is required. Cancelled.")
        return
    
    try:
        # Initialize Bittensor wallet
        print(f"Loading wallet: {wallet_name}/{hotkey_name}...")
        wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
        
        # Get hotkey address (SS58 format)
        hotkey_address = wallet.hotkey.ss58_address
        print(f"  Hotkey address: {hotkey_address}")
        
        # Sign the hotkey address (the message being signed is the hotkey itself)
        message = hotkey_address
        signature = wallet.hotkey.sign(message)
        signature_hex = "0x" + signature.hex()
        
        print(f"✓ Hotkey signed successfully")
    except Exception as exc:
        print(f"Failed to sign hotkey: {exc}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Link hotkey to account
    print("\nStep 4: Linking hotkey to Almanac account...")
    
    try:
        response = requests.post(
            f"{ALMANAC_API_URL}/subnet/select-hotkey",
            headers={
                "x-session-id": session_id,
                "x-wallet-address": wallet_address,
                "Content-Type": "application/json"
            },
            json={
                "hotkey": hotkey_address,
                "signature": signature_hex
            },
            timeout=30
        )
        
        if response.status_code != 200:
            print("Failed to link hotkey:")
            try:
                print(json.dumps(response.json(), indent=2))
            except Exception:
                print(response.text)
            return
        
        data = response.json()
        if data.get('success'):
            print("✓ Hotkey linked successfully")
            selected_hotkey = data.get('data', {}).get('selectedHotkey', {})
            uid = selected_hotkey.get('uid')
            coldkey = data.get('data', {}).get('coldkey')
            if uid is not None:
                print(f"  UID: {uid}")
            if coldkey:
                print(f"  Coldkey: {coldkey}")
        else:
            error_msg = data.get('error', 'Unknown error')
            user_msg = data.get('userMessage', error_msg)
            print(f"✗ Error: {error_msg}")
            print(f"  Message: {user_msg}")
    except Exception as exc:
        print(f"Failed to link hotkey: {exc}")
        import traceback
        traceback.print_exc()

def generate_polymarket_credentials():
    """
    Generate Polymarket CLOB API credentials using py-clob-client with EOA_WALLET_PK from api_trading.env.
    """
    global CREDENTIAL_SETS, SELECTED_CREDENTIAL_SET, CURRENT_SESSION, SELECTED_MARKET
    
    print("\nGenerating Polymarket API credentials...")
    load_dotenv(dotenv_path=str(ENV_PATH))
    private_key = _get_credential("EOA_WALLET_PK")
    if not private_key:
        print(f"EOA_WALLET_PK not found in {ENV_PATH}. Please set it and try again.")
        return

    # Validate address derives cleanly (optional)
    try:
        addr = Account.from_key(private_key).address
        print(f"Using wallet: {addr}")
    except Exception as exc:
        print(f"Invalid private key: {exc}")
        return

    proxy_funder_address = _get_credential("EOA_PROXY_FUNDER")
    if not proxy_funder_address:
        print(f"EOA_PROXY_FUNDER not found in {ENV_PATH}. Please set it and try again.")
        return

    # Create client and generate credentials
    client = ClobClient(host=POLYMARKET_CLOB_HOST, key=private_key, chain_id=POLYGON_CHAIN_ID)
    try:
        credentials = client.create_or_derive_api_key()
    except Exception as exc:
        print(f"Failed to create Polymarket API credentials: {exc}")
        return

    _display_credentials(credentials)
    
    # Prompt to reload environment file
    print("="*60)
    print("Next Steps:")
    print("1. Copy the credentials above and add them to your api_trading.env file")
    if SELECTED_CREDENTIAL_SET and SELECTED_CREDENTIAL_SET != "default":
        prefix = f"{SELECTED_CREDENTIAL_SET}_"
        print(f"   (Use prefix '{prefix}' for this credential set)")
    print("2. Reload the environment file to use the new credentials")
    print("="*60)
    
    reload = input("\nReload environment file now? (y/n): ").strip().lower()
    if reload in ("y", "yes"):
        # Reload environment and refresh credential sets
        load_dotenv(dotenv_path=str(ENV_PATH), override=True)
        CREDENTIAL_SETS = _detect_credential_sets()
        print("✓ Environment file reloaded. Credential sets refreshed.")
        
        # Validate current selection still exists
        if SELECTED_CREDENTIAL_SET and SELECTED_CREDENTIAL_SET not in CREDENTIAL_SETS:
            print(f"⚠ Previously selected credential set '{SELECTED_CREDENTIAL_SET}' no longer exists.")
            SELECTED_CREDENTIAL_SET = None
            CURRENT_SESSION = None
            SELECTED_MARKET = None
    else:
        print("Environment file not reloaded. You can reload it later by selecting a credential set.")

def interactive_setup():
    """
    Interactive setup process for trading session.
    """
    # ASCII Art Banner
    ascii_banner = """

     $$$$$$\  $$\       $$\      $$\  $$$$$$\  $$\   $$\  $$$$$$\   $$$$$$\  
    $$  __$$\ $$ |      $$$\    $$$ |$$  __$$\ $$$\  $$ |$$  __$$\ $$  __$$\ 
    $$ /  $$ |$$ |      $$$$\  $$$$ |$$ /  $$ |$$$$\ $$ |$$ /  $$ |$$ /  \__|
    $$$$$$$$ |$$ |      $$\$$\$$ $$ |$$$$$$$$ |$$ $$\$$ |$$$$$$$$ |$$ |      
    $$  __$$ |$$ |      $$ \$$$  $$ |$$  __$$ |$$ \$$$$ |$$  __$$ |$$ |      
    $$ |  $$ |$$ |      $$ |\$  /$$ |$$ |  $$ |$$ |\$$$ |$$ |  $$ |$$ |  $$\ 
    $$ |  $$ |$$$$$$$$\ $$ | \_/ $$ |$$ |  $$ |$$ | \$$ |$$ |  $$ |\$$$$$$  |
    \__|  \__|\________|\__|     \__|\__|  \__|\__|  \__|\__|  \__| \______/ 
                                                                                                                                           
                               Powered by
                   ╔═╗╔═╗╔═╗╦═╗╔╦╗╔═╗╔╦╗╔═╗╔╗╔╔═╗╔═╗╦═╗
                   ╚═╗╠═╝║ ║╠╦╝ ║ ╚═╗ ║ ║╣ ║║║╚═╗║ ║╠╦╝
                   ╚═╝╩  ╚═╝╩╚═ ╩ ╚═╝ ╩ ╚═╝╝╚╝╚═╝╚═╝╩╚═

    ________________________________________________________________________

    """
    print(ascii_banner)
    print("This script will help you generate Polymarket API credentials and provide a basic interactive flow.")
    
    # Initialize credential sets at startup
    global CREDENTIAL_SETS, SELECTED_CREDENTIAL_SET, CURRENT_SESSION, SELECTED_MARKET
    CREDENTIAL_SETS = _detect_credential_sets()
    
    # Validate existing selection if credential sets were loaded
    if CREDENTIAL_SETS:
        if SELECTED_CREDENTIAL_SET and SELECTED_CREDENTIAL_SET not in CREDENTIAL_SETS:
            # Previously selected set no longer exists, clear everything
            SELECTED_CREDENTIAL_SET = None
            CURRENT_SESSION = None
            SELECTED_MARKET = None
        elif len(CREDENTIAL_SETS) > 1 and not SELECTED_CREDENTIAL_SET:
            # If multiple sets found and no selection, default to "default" if it exists
            if "default" in CREDENTIAL_SETS:
                SELECTED_CREDENTIAL_SET = "default"
            else:
                # Otherwise select the first one
                SELECTED_CREDENTIAL_SET = sorted(CREDENTIAL_SETS.keys())[0]
    else:
        # No credential sets found, clear everything
        SELECTED_CREDENTIAL_SET = None
        CURRENT_SESSION = None
        SELECTED_MARKET = None

    while True:
        print("\nPlease choose an option:")
        print("  1) Trading, Positions, Claims, Funds")
        print("  2) Generate Polymarket API credentials")
        print("  3) Link Bittensor UID to Almanac account")
        if len(CREDENTIAL_SETS) > 1:
            current_set = SELECTED_CREDENTIAL_SET or "default"
            print(f"  4) Select Credential Set (current: {current_set})")
            print("  5) Exit")
            max_choice = 5
        else:
            print("  4) Exit")
            max_choice = 4
        choice = input("\nEnter choice: ").strip()

        if choice == "1":
            start_trading_flow()
        elif choice == "2":
            try:
                generate_polymarket_credentials()
            except Exception as exc:
                print(f"Failed to generate credentials: {exc}")
        elif choice == "3":
            try:
                link_bittensor_uid()
            except Exception as exc:
                print(f"Failed to link Bittensor UID: {exc}")
        elif choice == "4":
            if len(CREDENTIAL_SETS) > 1:
                try:
                    select_credential_set()
                except Exception as exc:
                    print(f"Failed to select credential set: {exc}")
            else:
                print("Ciao!")
                break
        elif choice == "5" and len(CREDENTIAL_SETS) > 1:
            print("Ciao!")
            break
        else:
            print(f"Invalid choice. Please enter 1 through {max_choice}.\n")

# Run the miner.
if __name__ == "__main__":
    interactive_setup()