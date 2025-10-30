import os
import contextlib
from typing import Optional
import threading
from pathlib import Path
import numpy as np
import bittensor as bt

# Optional PostgreSQL imports - only load if available
try:
    import psycopg2
    import psycopg2.extras
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None
    #psycopg2.extras = None

# Optional dotenv import - only load if available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = None

from constants import MINER_WEIGHT_PERCENTAGE, GENERAL_POOL_WEIGHT_PERCENTAGE


class PostgresValidatorStorage():
    _instance: Optional['PostgresValidatorStorage'] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'PostgresValidatorStorage':
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
        
    def __init__(self):
        self._initialized = False
        self.continuous_connection_do_not_reuse: Optional[object] = None
        self.lock = threading.RLock()
        self.postgres_available = POSTGRES_AVAILABLE
        self.connection_params = None
        self.env_loaded = False
        
        # Try to load connection parameters, but don't fail if it doesn't work
        try:
            self.connection_params = self._get_connection_params()
            self.env_loaded = True
        except Exception as e:
            bt.logging.warning(f"Failed to load connection parameters: {e}")
            self.env_loaded = False

    def _load_env_file(self):
        """Load environment variables from storage.env file."""
        if not DOTENV_AVAILABLE:
            bt.logging.warning("python-dotenv not available. Install with: pip install python-dotenv")
            return False
        
        # Look for storage.env in the same directory as this file
        env_file = Path(__file__).parent / "storage.env"
        
        if not env_file.exists():
            bt.logging.error(f"Environment file not found: {env_file}")
            return False
        
        try:
            load_dotenv(env_file)
            bt.logging.info(f"Loaded environment variables from {env_file}")
            return True
        except Exception as e:
            bt.logging.error(f"Failed to load environment file {env_file}: {e}")
            return False

    def _get_connection_params(self):
        """Get PostgreSQL connection parameters from environment variables."""
        # Try to load from storage.env file
        self._load_env_file()
        
        # Required environment variables
        required_vars = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            bt.logging.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            return {
                'host': os.getenv('DB_HOST'),
                'port': int(os.getenv('DB_PORT')),
                'database': os.getenv('DB_NAME'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD')
            }
        except ValueError as e:
            error_msg = f"Invalid environment variable format: {e}"
            bt.logging.error(error_msg)
            raise RuntimeError(error_msg)

    def initialize(self):
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            if not self.postgres_available:
                raise RuntimeError(
                    "PostgreSQL dependencies not available. "
                    "Install psycopg2 to enable PostgreSQL functionality: "
                    "pip install psycopg2-binary"
                )
            
            if not self.env_loaded or not self.connection_params:
                raise RuntimeError(
                    "Database connection parameters not loaded. "
                    "Check that storage.env file exists and contains required variables: "
                    "DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD"
                )
            
            self._initialized = True
            
            # Initialize database tables
            self._initialize_database()
            
            self.continuous_connection_do_not_reuse = self._create_connection()

    def _initialize_database(self):
        """Create database tables if they don't exist."""
        try:
            with contextlib.closing(self._create_connection()) as connection:
                with connection.cursor() as cursor:
                    # Create epoch_scores table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS epoch_scores (
                            id SERIAL PRIMARY KEY,
                            epoch_date DATE NOT NULL UNIQUE,
                            total_budget DECIMAL(15, 2) NOT NULL,
                            mp_budget DECIMAL(15, 2) NOT NULL,
                            gp_budget DECIMAL(15, 2) NOT NULL,
                            total_volume DECIMAL(15, 2) NOT NULL,
                            mp_volume DECIMAL(15, 2) NOT NULL,
                            gp_volume DECIMAL(15, 2) NOT NULL,
                            total_fees DECIMAL(15, 2) NOT NULL,
                            mp_fees DECIMAL(15, 2) NOT NULL,
                            gp_fees DECIMAL(15, 2) NOT NULL,
                            total_payouts DECIMAL(15, 2) NOT NULL,
                            mp_payouts DECIMAL(15, 2) NOT NULL,
                            gp_payouts DECIMAL(15, 2) NOT NULL,
                            mp_kappa DECIMAL(10, 6) NOT NULL,
                            gp_kappa DECIMAL(10, 6) NOT NULL,
                            mp_weight_percentage DECIMAL(10, 6) NOT NULL,
                            gp_weight_percentage DECIMAL(10, 6) NOT NULL,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Index for epoch_scores table - primary lookup by date
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_epoch_scores_epoch_date 
                        ON epoch_scores(epoch_date)
                    """)

                    # Create epoch_trader_scores table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS epoch_trader_scores (
                            id SERIAL PRIMARY KEY,
                            epoch_date DATE NOT NULL,
                            account_id INTEGER NOT NULL,
                            miner_uid INTEGER,
                            miner_hotkey VARCHAR(255),
                            is_general_pool BOOLEAN NOT NULL,
                            num_predictions INTEGER NOT NULL,
                            num_correct_predictions INTEGER NOT NULL,
                            volume DECIMAL(15, 2) NOT NULL,
                            qualified_volume DECIMAL(15, 2) NOT NULL,
                            fees DECIMAL(15, 2) NOT NULL,
                            pnl DECIMAL(15, 2) NOT NULL,
                            roi DECIMAL(10, 6) NOT NULL,
                            payout DECIMAL(15, 2) NOT NULL,
                            subnet_weight DECIMAL(10, 6) NOT NULL,
                            is_payout_distributed BOOLEAN NOT NULL DEFAULT FALSE,
                            payout_distributed_at TIMESTAMP,
                            payout_transaction_id VARCHAR(255),
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Create unique index that acts as a constraint for general pool entries
                    cursor.execute("""
                        CREATE UNIQUE INDEX IF NOT EXISTS uniq_epoch_general_pool ON epoch_trader_scores(epoch_date, account_id) WHERE is_general_pool = TRUE
                    """)

                    # Create unique index that acts as a constraint for miner pool entries
                    cursor.execute("""
                        CREATE UNIQUE INDEX IF NOT EXISTS uniq_epoch_miner_pool ON epoch_trader_scores(epoch_date, account_id, miner_uid, miner_hotkey) WHERE is_general_pool = FALSE
                    """)

                    # Primary lookup: Find all traders for a specific epoch
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_epoch_trader_scores_epoch_date ON epoch_trader_scores(epoch_date)
                    """)
                    
                    # Lookup by miner: Find all epochs for a specific miner
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_epoch_trader_scores_miner_uid ON epoch_trader_scores(miner_uid)
                    """)
                    
                    # Lookup by hotkey: Alternative miner identification
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_epoch_trader_scores_miner_hotkey ON epoch_trader_scores(miner_hotkey)
                    """)
                    
                    # Lookup by account: Find all traders for a specific account
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_epoch_trader_scores_account_id ON epoch_trader_scores(account_id)
                    """)
                    
                    # Composite index: Find specific trader's performance in specific epoch
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_epoch_trader_scores_epoch_miner ON epoch_trader_scores(epoch_date, miner_uid)
                    """)
                    
                    # Composite index (general pool only): Find general pool rows for specific epoch/account
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_epoch_trader_scores_epoch_account_pool ON epoch_trader_scores(epoch_date, account_id) WHERE is_general_pool = TRUE
                    """)
                    
                    # Performance ranking: Order by payout (descending) for leaderboards
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_epoch_trader_scores_payout_desc ON epoch_trader_scores(epoch_date, payout DESC)
                    """)
                    
                    # Performance ranking: Order by ROI (descending) for performance analysis
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_epoch_trader_scores_roi_desc ON epoch_trader_scores(epoch_date, roi DESC)
                    """)
                    
                    # Performance ranking: Order by volume (descending) for volume analysis
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_epoch_trader_scores_volume_desc ON epoch_trader_scores(epoch_date, volume DESC)
                    """)
                    
                    # Performance ranking: Order by PnL (descending) for profit analysis
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_epoch_trader_scores_pnl_desc ON epoch_trader_scores(epoch_date, pnl DESC)
                    """)
                    
                    connection.commit()
                    bt.logging.info("Database tables initialized successfully")
        except Exception as e:
            bt.logging.error(f"Failed to initialize database: {e}")
            raise

    def _create_connection(self):
        """Create a PostgreSQL connection."""
        if not self.postgres_available:
            raise RuntimeError("PostgreSQL dependencies not available")
        
        try:
            connection = psycopg2.connect(**self.connection_params)
            connection.autocommit = False
            return connection
        except Exception as e:
            bt.logging.error(f"Failed to create PostgreSQL connection: {e}")
            raise
    
    def get_connection(self):
        if not self._initialized:
            raise RuntimeError("PostgresValidatorStorage has not been initialized")
        return self.continuous_connection_do_not_reuse
    
    def is_postgres_available(self) -> bool:
        """Check if PostgreSQL dependencies are available."""
        return self.postgres_available
    
    def is_environment_loaded(self) -> bool:
        """Check if environment variables are properly loaded."""
        return self.env_loaded and self.connection_params is not None
    
    def get_connection_status(self) -> dict:
        """Get detailed status of the storage system."""
        return {
            'postgres_available': self.postgres_available,
            'environment_loaded': self.is_environment_loaded(),
            'initialized': self._initialized,
            'dotenv_available': DOTENV_AVAILABLE
        }
    

    def insert_epoch_scores(self, epoch_scores: dict):
        """
        Inserts the epoch scores into the database.
        
        Args:
            epoch_scores: Dictionary containing:
                - epoch_date: Date of the epoch
                - total_budget: Total budget for the epoch
                - mp_budget: Miner pool budget
                - gp_budget: General pool budget
                - total_volume: Total volume traded
                - mp_volume: Miner pool volume
                - gp_volume: General pool volume
                - total_fees: Total fees collected
                - mp_fees: Miner pool fees
                - gp_fees: General pool fees
                - total_payouts: Total payouts made
                - mp_payouts: Miner pool payouts
                - gp_payouts: General pool payouts
                - mp_kappa: Miner pool kappa value
                - gp_kappa: General pool kappa value
                - mp_weight_percentage: Miner pool weight percentage
                - gp_weight_percentage: General pool weight percentage
        """
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO epoch_scores (
                            epoch_date, total_budget, mp_budget, gp_budget,
                            total_volume, mp_volume, gp_volume,
                            total_fees, mp_fees, gp_fees,
                            total_payouts, mp_payouts, gp_payouts,
                            mp_kappa, gp_kappa, mp_weight_percentage, gp_weight_percentage
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (epoch_date) DO NOTHING
                    """, (
                        epoch_scores["epoch_date"],
                        epoch_scores["total_budget"],
                        epoch_scores["mp_budget"],
                        epoch_scores["gp_budget"],
                        epoch_scores["total_volume"],
                        epoch_scores["mp_volume"],
                        epoch_scores["gp_volume"],
                        epoch_scores["total_fees"],
                        epoch_scores["mp_fees"],
                        epoch_scores["gp_fees"],
                        epoch_scores["total_payouts"],
                        epoch_scores["mp_payouts"],
                        epoch_scores["gp_payouts"],
                        epoch_scores["mp_kappa"],
                        epoch_scores["gp_kappa"],
                        epoch_scores["mp_weight_percentage"],
                        epoch_scores["gp_weight_percentage"]
                    ))
                    connection.commit()

    def insert_epoch_trader_scores(self, trader_scores: dict):
        """
        Inserts the epoch trader scores into the database.
        
        Args:
            trading_scores: Dictionary containing:
                - epoch_date: Date of the epoch
                - account_id: Unique identifier for the account
                - miner_uid: Unique identifier for the miner
                - miner_hotkey: Hotkey string for the miner
                - is_general_pool: Boolean indicating if the trader is a general pool trader
                - num_predictions: Number of predictions made
                - num_correct_predictions: Number of correct predictions
                - volume: Trading volume
                - qualified_volume: Qualified trading volume
                - fees: Fees paid by the miner
                - pnl: Profit and loss
                - roi: Return on investment
                - payout: Payout amount
                - subnet_weight: Weight in the subnet
        """
        with self.lock:
            with contextlib.closing(self._create_connection()) as connection:
                with connection.cursor() as cursor:
                    # Determine which constraint to use based on the data
                    if trader_scores.get("miner_uid") is not None and trader_scores.get("miner_hotkey") is not None and trader_scores.get("is_general_pool") is False:
                        # This is a miner entry - use miner constraint
                        cursor.execute("""
                            INSERT INTO epoch_trader_scores (
                                epoch_date, account_id, miner_uid, miner_hotkey, is_general_pool,
                                num_predictions, num_correct_predictions,
                                volume, qualified_volume, fees, pnl, roi, payout, subnet_weight
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                        """, (
                            trader_scores["epoch_date"],
                            trader_scores["account_id"],
                            trader_scores["miner_uid"],
                            trader_scores["miner_hotkey"],
                            trader_scores["is_general_pool"],
                            trader_scores["num_predictions"],
                            trader_scores["num_correct_predictions"],
                            trader_scores["volume"],
                            trader_scores["qualified_volume"],
                            trader_scores["fees"],
                            trader_scores["pnl"],
                            trader_scores["roi"],
                            trader_scores["payout"],
                            trader_scores["subnet_weight"]
                        ))
                    elif trader_scores.get("is_general_pool") is True:
                        # This is a general pool entry - use pool constraint
                        cursor.execute("""
                            INSERT INTO epoch_trader_scores (
                                epoch_date, account_id, miner_uid, miner_hotkey, is_general_pool,
                                num_predictions, num_correct_predictions,
                                volume, qualified_volume, fees, pnl, roi, payout, subnet_weight
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                        """, (
                            trader_scores["epoch_date"],
                            trader_scores["account_id"],
                            trader_scores["miner_uid"],
                            trader_scores["miner_hotkey"],
                            trader_scores["is_general_pool"],
                            trader_scores["num_predictions"],
                            trader_scores["num_correct_predictions"],
                            trader_scores["volume"],
                            trader_scores["qualified_volume"],
                            trader_scores["fees"],
                            trader_scores["pnl"],
                            trader_scores["roi"],
                            trader_scores["payout"],
                            trader_scores["subnet_weight"]
                        ))
                    else:
                        # Fallback - try to insert without conflict handling
                        cursor.execute("""
                            INSERT INTO epoch_trader_scores (
                                epoch_date, account_id, miner_uid, miner_hotkey, is_general_pool,
                                num_predictions, num_correct_predictions,
                                volume, qualified_volume, fees, pnl, roi, payout, subnet_weight
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            trader_scores["epoch_date"],
                            trader_scores["account_id"],
                            trader_scores["miner_uid"],
                            trader_scores["miner_hotkey"],
                            trader_scores["is_general_pool"],
                            trader_scores["num_predictions"],
                            trader_scores["num_correct_predictions"],
                            trader_scores["volume"],
                            trader_scores["qualified_volume"],
                            trader_scores["fees"],
                            trader_scores["pnl"],
                            trader_scores["roi"],
                            trader_scores["payout"],
                            trader_scores["subnet_weight"]
                        ))
                    connection.commit()


# Global accessor function
def get_storage() -> PostgresValidatorStorage:
    return PostgresValidatorStorage.get_instance()


def log_scores_to_database(miner_history, general_pool_history, miners_scores, general_pool_scores, miner_budget, general_pool_budget, all_hotkeys, weights):
    """
    Utility function to log scores to the database.
    This function handles the conversion from scoring data to database format.
    
    Args:
        miner_history: History of miner performance data (epoch-based matrices)
        general_pool_history: History of general pool performance data (epoch-based matrices)
        miners_scores: Calculated scores for miners (from score_with_epochs)
        general_pool_scores: Calculated scores for general pools (from score_with_epochs)
        miner_budget: Budget allocated to miners
        general_pool_budget: Budget allocated to general pools
        all_hotkeys: List of all hotkeys for the miners
        weights: List of final weights for all UIDs in the subnet
    """
    try:
        # Get storage instance and initialize
        storage = get_storage()
        storage.initialize()
        
        # Get current date for epoch
        from datetime import date, timedelta, datetime
        # The epoch we're scoring should be yesterday timezone UTC with 00:00:00
        yesterday = date.today() - timedelta(days=1)
        epoch_date = datetime.combine(yesterday, datetime.min.time()).isoformat() + "Z"
        
        # Extract current epoch data from history matrices
        # Current epoch is the last row in the matrices (index -1)
        current_epoch_idx = miner_history["n_epochs"] - 1
        
        # Extract current epoch data for miners
        miner_current_volume = miner_history["volume_prev"][current_epoch_idx] if miner_history["n_entities"] > 0 else np.array([])
        miner_current_qualified = miner_history["qualified_prev"][current_epoch_idx] if miner_history["n_entities"] > 0 else np.array([])
        miner_current_profit = miner_history["profit_prev"][current_epoch_idx] if miner_history["n_entities"] > 0 else np.array([])
        miner_current_fees = miner_history["fees_prev"][current_epoch_idx] if miner_history["n_entities"] > 0 else np.array([])
        miner_current_trades = miner_history["trade_counts"][current_epoch_idx] if miner_history["n_entities"] > 0 else np.array([])
        miner_current_correct_trades = miner_history["correct_trade_counts"][current_epoch_idx] if miner_history["n_entities"] > 0 else np.array([])

        # Extract current epoch data for general pools
        gp_current_volume = general_pool_history["volume_prev"][current_epoch_idx] if general_pool_history["n_entities"] > 0 else np.array([])
        gp_current_qualified = general_pool_history["qualified_prev"][current_epoch_idx] if general_pool_history["n_entities"] > 0 else np.array([])
        gp_current_profit = general_pool_history["profit_prev"][current_epoch_idx] if general_pool_history["n_entities"] > 0 else np.array([])
        gp_current_fees = general_pool_history["fees_prev"][current_epoch_idx] if general_pool_history["n_entities"] > 0 else np.array([])
        gp_current_trades = general_pool_history["trade_counts"][current_epoch_idx] if general_pool_history["n_entities"] > 0 else np.array([])
        gp_current_correct_trades = general_pool_history["correct_trade_counts"][current_epoch_idx] if general_pool_history["n_entities"] > 0 else np.array([])
        
        # Calculate totals for current epoch (round to 2 decimal places for currency)
        total_budget = round(float(miner_budget + general_pool_budget), 2)
        total_volume = round(float(np.sum(miner_current_volume) + np.sum(gp_current_volume)), 2)
        total_fees = round(float(np.sum(miner_current_fees) + np.sum(gp_current_fees)), 2)
        total_payouts = round(float(np.sum(miners_scores.get('tokens', [])) + np.sum(general_pool_scores.get('tokens', []))), 2)
        
        # Calculate MP (Market Prediction) and GP (General Pool) breakdowns
        mp_budget = round(float(miner_budget), 2)
        gp_budget = round(float(general_pool_budget), 2)
        mp_volume = round(float(np.sum(miner_current_volume)), 2)
        gp_volume = round(float(np.sum(gp_current_volume)), 2)
        mp_fees = round(float(np.sum(miner_current_fees)), 2)
        gp_fees = round(float(np.sum(gp_current_fees)), 2)
        mp_payouts = round(float(np.sum(miners_scores.get('tokens', []))), 2)
        gp_payouts = round(float(np.sum(general_pool_scores.get('tokens', []))), 2)
        
        # Extract kappa values from the scoring results (round to 6 decimal places for ratios)
        mp_kappa = round(float(miners_scores.get('kappa_bar', 0.0)), 6)
        gp_kappa = round(float(general_pool_scores.get('kappa_bar', 0.0)), 6)
        
        # Insert epoch-level data
        epoch_data = {
            "epoch_date": epoch_date,
            "total_budget": total_budget,
            "mp_budget": mp_budget,
            "gp_budget": gp_budget,
            "total_volume": total_volume,
            "mp_volume": mp_volume,
            "gp_volume": gp_volume,
            "total_fees": total_fees,
            "mp_fees": mp_fees,
            "gp_fees": gp_fees,
            "total_payouts": total_payouts,
            "mp_payouts": mp_payouts,
            "gp_payouts": gp_payouts,
            "mp_kappa": mp_kappa,
            "gp_kappa": gp_kappa,
            "mp_weight_percentage": MINER_WEIGHT_PERCENTAGE,
            "gp_weight_percentage": GENERAL_POOL_WEIGHT_PERCENTAGE
        }
        storage.insert_epoch_scores(epoch_data)
        
        # Insert trader-level data for miners
        miner_entity_ids = miners_scores.get('entity_ids', [])
        miner_tokens = miners_scores.get('tokens', [])
        miner_total_volume = miners_scores.get('total_volume', [])
        miner_total_profit = miners_scores.get('total_profit', [])
        miner_roi_trailing = miners_scores.get('roi_trailing', [])
        
        for i, miner_uid in enumerate(miner_entity_ids):
            # Get current epoch data for this miner
            current_volume = miner_current_volume[i] if i < len(miner_current_volume) else 0.0
            current_qualified = miner_current_qualified[i] if i < len(miner_current_qualified) else 0.0
            current_profit = miner_current_profit[i] if i < len(miner_current_profit) else 0.0
            current_fees = miner_current_fees[i] if i < len(miner_current_fees) else 0.0
            current_trades = int(miner_current_trades[i]) if i < len(miner_current_trades) else 0
            current_correct_trades = int(miner_current_correct_trades[i]) if i < len(miner_current_correct_trades) else 0

            # Calculate current epoch ROI
            current_roi = current_profit / current_qualified if current_qualified > 0 else 0.0
            
            # Get miner hotkey - all_hotkeys is a list where index = UID
            miner_hotkey = all_hotkeys[miner_uid] if miner_uid < len(all_hotkeys) else None

            # Get account ID from account map
            account_id = miner_history["account_map"][miner_uid] if miner_uid in miner_history["account_map"] else None

            subnet_weight = weights[miner_uid] if miner_uid < len(weights) else 0.0
            
            trader_data = {
                "epoch_date": epoch_date,
                "account_id": account_id,
                "miner_uid": miner_uid,
                "miner_hotkey": miner_hotkey,
                "is_general_pool": False,  # Miners are not general pool traders
                "num_predictions": current_trades,
                "num_correct_predictions": current_correct_trades,
                "volume": round(float(current_volume), 2),
                "qualified_volume": round(float(current_qualified), 2),
                "fees": round(float(current_fees), 2),
                "pnl": round(float(current_profit), 2),
                "roi": round(float(current_roi), 6),
                "payout": round(float(miner_tokens[i]) if i < len(miner_tokens) else 0.0, 2),
                "subnet_weight": round(float(subnet_weight), 6)
            }
            storage.insert_epoch_trader_scores(trader_data)
        
        # Insert trader-level data for general pools
        gp_entity_ids = general_pool_scores.get('entity_ids', [])
        gp_tokens = general_pool_scores.get('tokens', [])
        gp_total_volume = general_pool_scores.get('total_volume', [])
        gp_total_profit = general_pool_scores.get('total_profit', [])
        gp_roi_trailing = general_pool_scores.get('roi_trailing', [])
        
        for i, pool_id in enumerate(gp_entity_ids):
            # Get current epoch data for this pool
            current_volume = gp_current_volume[i] if i < len(gp_current_volume) else 0.0
            current_qualified = gp_current_qualified[i] if i < len(gp_current_qualified) else 0.0
            current_profit = gp_current_profit[i] if i < len(gp_current_profit) else 0.0
            current_fees = gp_current_fees[i] if i < len(gp_current_fees) else 0.0
            current_trades = int(gp_current_trades[i]) if i < len(gp_current_trades) else 0
            current_correct_trades = int(gp_current_correct_trades[i]) if i < len(gp_current_correct_trades) else 0

            # Calculate current epoch ROI
            current_roi = current_profit / current_qualified if current_qualified > 0 else 0.0
            
            # Get account ID from account map
            account_id = general_pool_history["account_map"][pool_id] if pool_id in general_pool_history["account_map"] else None
            
            trader_data = {
                "epoch_date": epoch_date,
                "account_id": account_id,
                "miner_uid": None,  # General pools don't have miner UIDs
                "miner_hotkey": None,
                "is_general_pool": True,
                "num_predictions": current_trades,
                "num_correct_predictions": current_correct_trades,
                "volume": round(float(current_volume), 2),
                "qualified_volume": round(float(current_qualified), 2),
                "fees": round(float(current_fees), 2),
                "pnl": round(float(current_profit), 2),
                "roi": round(float(current_roi), 6),
                "payout": round(float(gp_tokens[i]) if i < len(gp_tokens) else 0.0, 2),
                "subnet_weight": 0.0 # always 0 for general pool
            }
            storage.insert_epoch_trader_scores(trader_data)
        
        bt.logging.info(f"Successfully logged scores to database for epoch {epoch_date}")
        
    except Exception as e:
        bt.logging.error(f"Failed to log scores to database: {e}")
        raise