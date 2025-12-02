#!/usr/bin/env python3
"""
Almanac API Interactive Trading Client

This script provides an interactive trading session with the Almanac API.
It allows you to:
- Generate Polymarket API credentials
- Initiate a trading session
- View trading session details
- Exit the program

Requirements:
- Python 3.10+
- Pip
- CPU
- Almanac account (setup at https://almanac.market)
- EOA wallet private key for signing transactions

Python dependencies:
- requests
- dotenv
- py-clob-client
- eth-account

pip install requests dotenv py-clob-client eth-account
"""

import os
import json
from pathlib import Path
import requests
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_account.messages import encode_typed_data as EIP712_ENCODE  # type: ignore
import time
import secrets
from constants import VOLUME_FEE

ALMANAC_API_URL = "https://api.almanac.market/api"
#ALMANAC_API_URL = "http://localhost:3001/api"
POLYMARKET_CLOB_HOST = "https://clob.polymarket.com"
POLYGON_CHAIN_ID = 137
# EIP-712 domain contract for Polymarket CTF Exchange
EIP712_DOMAIN_CONTRACT = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
EIP712_DOMAIN_NEGRISK_CONTRACT = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
ENV_PATH = Path("api_trading.env")

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

def _format_price(value):
    try:
        f = float(value)
        # Clip to [0,1] range for probabilities if out-of-range but close
        if -0.05 <= f <= 1.05:
            f = min(max(f, 0.0), 1.0)
        return f"{f:.2f}"
    except Exception:
        return "-"

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
    Print markets for a single event with concise outcomes summary.
    Returns the list of markets for further selection.
    """
    markets = event.get("markets") or []
    if not isinstance(markets, list) or not markets:
        print("No markets found for this event.")
        return []
    # Order by first outcome price (index-0, commonly 'Yes') descending
    def _first_yes_price(m):
        prices = m.get("outcome_prices") or []
        try:
            if isinstance(prices, list) and len(prices) > 0:
                return float(prices[0])
        except Exception:
            pass
        # Fallbacks
        try:
            return float(m.get("yesPrice") or 0.0)
        except Exception:
            return 0.0
    markets = sorted(markets, key=_first_yes_price, reverse=True)
    print("\nMarkets:")
    for idx, m in enumerate(markets, start=1):
        title = m.get("title") or m.get("question") or m.get("name") or "Untitled"
        market_id = m.get("id") or m.get("marketId") or m.get("_id") or "unknown"
        summary = _extract_outcomes_summary(m)
        if summary:
            print(f"  {idx}) {title} [{market_id}]\n      {summary}")
        else:
            print(f"  {idx}) {title} [{market_id}]")
    return markets

def _display_outcomes_and_choose(market: dict):
    """
    Show detailed outcomes for a market (if available) and let the user pick one.
    Returns a tuple (chosen_outcome_name, chosen_outcome_price, chosen_token_id).
    """
    outcomes = market.get("outcomes")
    outcome_prices = market.get("outcome_prices")
    clob_token_ids = market.get("clob_token_ids")

    # Normalize into list of dicts {name, price?, tokenId?} with index alignment
    normalized = []
    if isinstance(outcomes, list):
        for idx, o in enumerate(outcomes):
            name = o if isinstance(o, str) else (o.get("name") if isinstance(o, dict) else str(o))
            price = None
            token_id = None
            if isinstance(outcome_prices, list) and idx < len(outcome_prices):
                price = outcome_prices[idx]
            if isinstance(clob_token_ids, list) and idx < len(clob_token_ids):
                token_id = clob_token_ids[idx]
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
            if isinstance(outcome_prices, list) and idx < len(outcome_prices):
                price = outcome_prices[idx]
            if isinstance(clob_token_ids, list) and idx < len(clob_token_ids):
                token_id = clob_token_ids[idx]
            normalized.append({"name": str(name), "price": price, "tokenId": token_id})
    else:
        # Fallback binary representation
        yes_price = market.get("yesPrice") or market.get("yes")
        no_price = market.get("noPrice") or market.get("no")
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
        print(f"  {idx}) {o['name']} {_format_price(o.get('price'))}")
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

def _place_order_now(market: dict, chosen_outcome_name: str | None = None, chosen_token_id: str | None = None):
    """
    Inline order placement flow; prompts for side/size/price, shows summary, and submits with confirmation.
    User can type 'c' at any prompt to cancel.
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
    
    print("\nPlace order (type 'c' at any prompt to cancel):")
    
    # Side input with cancel option
    side_input = input("Side (buy/sell/c) [buy]: ").strip().lower()
    if side_input == "c":
        print("Order cancelled.")
        return
    side = side_input or "buy"
    if side not in ("buy", "sell"):
        print("Side must be 'buy' or 'sell'.")
        return
    
    # Size and price input with cancel option and $5 minimum validation
    while True:
        size_str = input("Size (quantity/c) [1]: ").strip()
        if size_str.lower() == "c":
            print("Order cancelled.")
            return
        size_str = size_str or "1"
        
        price_str = input("Price (0-1/c) [0.01]: ").strip()
        if price_str.lower() == "c":
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
        notional = size * price
        if notional < 5.0:
            print(f"Order notional ${notional:.2f} is below the $5 minimum. Please increase size and/or price.")
            continue
        break
    
    # Order type input with cancel option
    print("\nOrder Type:")
    print("  FOK - Fill Or Kill: Market Order must be filled immediately or it's cancelled")
    print("  FAK - Fill And Kill: Market Order will be filled immediately with what is available and the rest cancelled")
    # GTC and GTD are not supported yet
    #print("  GTC - Good Till Canceled: Limit order remains active until filled or cancelled")
    order_type_input = input("Order type (fok/fak/c) [fok]: ").strip().upper()
    if order_type_input == "C":
        print("Order cancelled.")
        return
    order_type = order_type_input or "FOK"
    if order_type not in ("FOK", "FAK"):
        print("Order type must be 'FOK' or 'FAK'. Using default 'FOK'.")
        order_type = "FOK"
    
    side_upper = "BUY" if side == "buy" else "SELL"
    notional = size * price
    # Fee calculation is for display only - we send original size/price to API
    fee = notional * VOLUME_FEE
    total_with_fee = notional + fee
    
    # Display summary
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
    print(f"Price: {price}")
    print(f"Subtotal: ${notional:.2f}")
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

def start_trading_flow():
    """
    Submenu for trading: auto-creates session if needed, search markets, place orders.
    Keeps state in CURRENT_SESSION and SELECTED_MARKET.
    """
    global CURRENT_SESSION, SELECTED_MARKET
    
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
        print("  2) Refresh Trading Session")
        print("  3) Back to Main Menu")
        choice = input("\nEnter choice: ").strip()

        if choice == "1":
            search_markets()
        elif choice == "2":
            try:
                session = initiate_trading_session()
                if session:
                    print("\nTrading session refreshed.")
                    CURRENT_SESSION = session
                else:
                    print("Trading session could not be refreshed.")
            except Exception as exc:
                print(f"Failed to refresh trading session: {exc}")
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 3.\n")

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
    wallet_address = os.getenv("EOA_WALLET_ADDRESS")
    if not wallet_address:
        print(f"EOA_WALLET_ADDRESS not found in {ENV_PATH}. Please set it and try again.")
        return
    
    private_key = os.getenv("EOA_WALLET_PK")
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
        "apiKey": os.getenv("POLYMARKET_API_KEY"),
        "secret": os.getenv("POLYMARKET_API_SECRET"),
        "passphrase": os.getenv("POLYMARKET_API_PASSPHRASE")
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

def place_order(
    market_id: str,
    side_upper: str,
    size: float,
    price: float,
    neg_risk: bool = False,
    order_type: str = "FOK",
    chosen_outcome_name: str | None = None,
    chosen_token_id: str | None = None,
):
    """
    Place an order for the selected market.

    Expected headers (if supported by server):
      - x-session-id
      - x-wallet-address

    Request body supported by server (simple flow):
      - marketId: string (required)
      - tokenId: string (optional)
      - side: "BUY" | "SELL" (optional)
      - orderType: "FOK" | "FAK" (optional; default "FOK"; "GTC" and "GTD" are not supported yet)
      - price: float between 0.01 and 0.99 (optional)
      - size: float >= POLYMARKET_MIN_ORDER (optional)
      - signature: string (optional)
      - signedOrder: object (optional; advanced flow)
      - signedOrder.signature: string (optional)
      - signedOrder.orderPayload: object (optional)
      - userWalletAddress: string (optional)
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
        or os.getenv("EOA_WALLET_ADDRESS")
    )
    proxy_address = (
        CURRENT_SESSION.get("data").get("proxyWallet")
        or os.getenv("EOA_PROXY_FUNDER")
    )

    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["x-session-id"] = session_id
    if wallet_address:
        headers["x-wallet-address"] = wallet_address

    # Attempt EIP-712 signed order flow if config present
    exchange_address = EIP712_DOMAIN_NEGRISK_CONTRACT if neg_risk else EIP712_DOMAIN_CONTRACT
    private_key = os.getenv("EOA_WALLET_PK")
    signed_flow_payload = None
    try:
        if exchange_address and private_key and wallet_address:
            if not private_key.startswith("0x"):
                private_key = "0x" + private_key
            # Numeric fields as ints for EIP-712
            def to_6d_int(x: float) -> int:
                return int(round(float(x) * 1_000_000))
            # Build order payload
            side_num = 0 if side_upper == "BUY" else 1
            # use BigInt(Date.now()) for salt
            salt = int(time.time() * 1000)
            maker_amount = to_6d_int(size * price if side_num == 0 else size)
            taker_amount = to_6d_int(size if side_num == 0 else size * price)
            # Frontend sets expiration and nonce to 0, feeRateBps to 0
            expiration = 0
            nonce = 0
            fee_bps = 0
            # tokenId parse
            token_id_int = 0
            if chosen_token_id is not None:
                try:
                    token_id_int = int(str(chosen_token_id), 10)
                except Exception:
                    token_id_int = 0
            order_payload = {
                "salt": salt,
                "maker": proxy_address,
                "signer": wallet_address,
                "taker": "0x0000000000000000000000000000000000000000",
                "tokenId": token_id_int,
                "makerAmount": maker_amount,
                "takerAmount": taker_amount,
                "expiration": expiration,
                "nonce": nonce,
                "feeRateBps": fee_bps,
                "side": side_num,
                "signatureType": 2,
            }
            # EIP-712 typed data (Polymarket-esque order schema)
            typed_data = {
                "types": {
                    "EIP712Domain": [
                        {"name": "name", "type": "string"},
                        {"name": "version", "type": "string"},
                        {"name": "chainId", "type": "uint256"},
                        {"name": "verifyingContract", "type": "address"},
                    ],
                    "Order": [
                        {"name": "salt", "type": "uint256"},
                        {"name": "maker", "type": "address"},
                        {"name": "signer", "type": "address"},
                        {"name": "taker", "type": "address"},
                        {"name": "tokenId", "type": "uint256"},
                        {"name": "makerAmount", "type": "uint256"},
                        {"name": "takerAmount", "type": "uint256"},
                        {"name": "expiration", "type": "uint256"},
                        {"name": "nonce", "type": "uint256"},
                        {"name": "feeRateBps", "type": "uint256"},
                        {"name": "side", "type": "uint8"},
                        {"name": "signatureType", "type": "uint8"},
                    ],
                },
                "primaryType": "Order",
                "domain": {
                    "name": "Polymarket CTF Exchange",
                    "version": "1",
                    "chainId": POLYGON_CHAIN_ID,
                    "verifyingContract": exchange_address,
                },
                "message": order_payload,
            }
            # eth-account 0.13.x: pass full_message=typed_data
            eip712_msg = EIP712_ENCODE(full_message=typed_data)
            signed = Account.from_key(private_key).sign_message(eip712_msg)
            signature_hex = signed.signature.hex()
            if not signature_hex.startswith("0x"):
                signature_hex = "0x" + signature_hex
            # Convert numeric fields to strings for backend
            order_payload_str = {
                "salt": str(order_payload["salt"]),
                "maker": order_payload["maker"],
                "signer": order_payload["signer"],
                "taker": order_payload["taker"],
                "tokenId": str(order_payload["tokenId"]),
                "makerAmount": str(order_payload["makerAmount"]),
                "takerAmount": str(order_payload["takerAmount"]),
                "expiration": str(order_payload["expiration"]),
                "nonce": str(order_payload["nonce"]),
                "feeRateBps": str(order_payload["feeRateBps"]),
                "side": order_payload["side"],
                "signatureType": order_payload["signatureType"],
            }
            signed_flow_payload = {
                "marketId": market_id,
                "signedOrder": {
                    "signature": signature_hex,
                    "orderPayload": order_payload_str,
                },
                "orderType": order_type,
                "userWalletAddress": wallet_address,
            }
    except Exception as _exc:
        # Fallback to simple flow below if signing fails
        signed_flow_payload = None
        import traceback
        traceback.print_exc()

    if signed_flow_payload is None:
        print("Failed to build signed order payload. Aborting without sending.")
        return
    payload = signed_flow_payload

    try:
        resp = requests.post(
            f"{ALMANAC_API_URL}/v1/trading/orders",
            headers=headers,
            json=payload,
            timeout=30,
        )
        if resp.status_code != 200:
            print("Failed to place order:")
            try:
                print(json.dumps(resp.json(), indent=2))
            except Exception:
                print(resp.text)
            return
        print("Order placed:")
        print(json.dumps(resp.json(), indent=2))
    except Exception as exc:
        print(f"Order error: {exc}")

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
        markets = _display_markets_for_event(chosen_event)
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

def generate_polymarket_credentials():
    """
    Generate Polymarket CLOB API credentials using py-clob-client with EOA_WALLET_PK from api_trading.env.
    """
    print("\nGenerating Polymarket API credentials...")
    load_dotenv(dotenv_path=str(ENV_PATH))
    private_key = os.getenv("EOA_WALLET_PK")
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

    proxy_funder_address = os.getenv("EOA_PROXY_FUNDER")
    if not proxy_funder_address:
        print(f"EOA_PROXY_FUNDER not found in {ENV_PATH}. Please set it and try again.")
        return

    # Create client and generate credentials
    client = ClobClient(host=POLYMARKET_CLOB_HOST, key=private_key, chain_id=POLYGON_CHAIN_ID)
    try:
        credentials = client.create_or_derive_api_creds()
    except Exception as exc:
        print(f"Failed to create Polymarket API credentials: {exc}")
        return

    _display_credentials(credentials)

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

    while True:
        print("\nPlease choose an option:")
        print("  1) Start Trading")
        print("  2) Generate Polymarket API credentials")
        print("  3) Exit")
        choice = input("\nEnter choice: ").strip()

        if choice == "1":
            start_trading_flow()
        elif choice == "2":
            try:
                generate_polymarket_credentials()
            except Exception as exc:
                print(f"Failed to generate credentials: {exc}")
        elif choice == "3":
            print("Ciao!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.\n")

# Run the miner.
if __name__ == "__main__":
    interactive_setup()