import os
import argparse
import bittensor as bt
import traceback
import datetime
import time
import wandb
import json
from typing import Optional, Dict, List
import requests
from requests.auth import HTTPBasicAuth
from subprocess import Popen, PIPE
from substrateinterface import SubstrateInterface
import numpy as np
import random

from metadata_manager import MetadataManager
from scoring import score_miners, calculate_weights, print_pool_stats
from storage.postgres_validator_storage import log_scores_to_database, get_storage
from constants import (
    ROLLING_HISTORY_IN_DAYS,
    TOTAL_MINER_ALPHA_PER_DAY,
)


class Validator:
    def __init__(self):
        self.config = self.get_config()
        self.setup_logging()
        self.setup_bittensor_objects()
        self.last_update = 0
        self.current_block = 0
        self.node = SubstrateInterface(url=self.config.subtensor.chain_endpoint)
        self.tempo = self.node_query('SubtensorModule', 'Tempo', [self.config.netuid])
        #self.moving_avg_scores = [1.0] * len(self.metagraph.S)
        #self.alpha = 0.1

        self.trading_history_endpoint = "https://api.almanac.market/api/v1/trading/trading-history"
        if self.config.subtensor.network == "test":
            self.trading_history_endpoint = "https://test-api.almanac.market/api/v1/trading/trading-history"
            #self.trading_history_endpoint = "http://localhost:3001/api/v1/trading/trading-history"
        self.rolling_history_in_days = ROLLING_HISTORY_IN_DAYS
        self.trading_history_batch_limit = 1000
        # Set up auto update.
        self.last_update_check = datetime.datetime.now()
        self.update_check_interval = 60 * 60 * 24  # 24 hours
        
        # Set up metadata manager
        self.metadata_manager = MetadataManager(
            netuid=self.config.netuid,
            network=self.config.subtensor.network,
            state_file=f"validator_state_{self.config.netuid}.json"
        )
        if not self.config.metadata_manager.off:
            self.metadata_manager.start()
        
        # Set up wandb.
        self.wandb_run = None
        self.wandb_run_start = None
        if not self.config.wandb.off:
            if os.getenv("WANDB_API_KEY"):
                self.new_wandb_run()
            else:
                bt.logging.exception(
                    "WANDB_API_KEY not found. Set it with `export WANDB_API_KEY=<your API key>`. Alternatively, you can disable W&B with --wandb.off, but it is strongly recommended to run with W&B enabled."
                )
                self.config.wandb.off = True
        else:
            bt.logging.warning(
                "Running with --wandb.off. It is strongly recommended to run with W&B enabled."
            )
        
        # Use correct endpoint based on network
        if self.config.subtensor.network == "test":
            endpoint = "wss://test.finney.opentensor.ai:443"
        else:
            # Use the configured endpoint for mainnet (finney)
            endpoint = self.config.subtensor.chain_endpoint
        
        self.node = SubstrateInterface(url=endpoint)

        # If DB score logging is enabled, initialize and test storage connection
        if getattr(self.config, 'db_score_logging', False):
            try:
                bt.logging.info("Initializing PostgreSQL storage (db_score_logging)...")
                storage = get_storage()
                storage.initialize()
                bt.logging.success("✅ PostgreSQL storage initialized and tables ensured.")
            except Exception as e:
                bt.logging.error(f"❌ Failed to initialize PostgreSQL storage: {e}")
                bt.logging.warning("Disabling db_score_logging for this run to avoid runtime failures.")
                # Disable DB logging for the remainder of the run
                self.config.db_score_logging = False

    def get_config(self):
        # Set up the configuration parser.
        parser = argparse.ArgumentParser()
        # Adds override arguments for network and netuid.
        parser.add_argument('--netuid', type=int, default=1, help="The chain subnet uid.")
        # Adds subtensor specific arguments.
        bt.subtensor.add_args(parser)
        # Adds logging specific arguments.
        bt.logging.add_args(parser)
        # Adds wallet specific arguments.
        bt.wallet.add_args(parser)
        # Adds wandb arguments
        parser.add_argument('--wandb.off', action='store_true', help="Disable wandb logging.")
        # Adds auto-update arguments.
        parser.add_argument('--auto_update', action='store_true', help="Enable auto-update of the validator.")
        # Adds metadata manager arguments.
        parser.add_argument('--metadata_manager.off', action='store_true', help="Disable metadata manager.")
        # Adds postgres database score logging.
        parser.add_argument('--db_score_logging', action='store_true', help="Enable postgres database score logging.")
        # Adds a flag to use synthetic data for testing
        parser.add_argument('--use_synthetic_data', action='store_true', help="Use synthetic data for testing.")
        # Adds a flag to write trading history to a local JSON file
        parser.add_argument('--write_trading_history', action='store_true', help="Write trading history to local trading_history.json file.")
        # Parse the config.
        config = bt.config(parser)
        # Set up logging directory.
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey_str,
                config.netuid,
                'validator',
            )
        )
        # Ensure the logging directory exists.
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        # Set up logging.
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.network} with config:")
        bt.logging.info(self.config)

    def setup_bittensor_objects(self):
        # Build Bittensor validator objects.
        bt.logging.info("Setting up Bittensor objects.")

        # Initialize wallet.
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        # Initialize subtensor.
        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        # Initialize dendrite.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Initialize metagraph.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        # Connect the validator to the network.
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(f"\nYour validator: {self.wallet} is not registered to chain connection: {self.subtensor} \nRun 'btcli register' and try again.")
            exit()
        else:
            # Each validator gets a unique identity (UID) in the network.
            self.my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            bt.logging.info(f"Running validator on uid: {self.my_uid}")

        # Set up initial scoring weights for validation.
        bt.logging.info("Building initial validation weights.")
        self.scores = [0] * len(self.metagraph.S)
        #bt.logging.info(f"Weights: {self.scores}")

    def node_query(self, module, method, params):
        try:
            result = self.node.query(module, method, params).value

        except Exception:
            # reinitilize node
            if self.config.subtensor.network == "test":
                endpoint = "wss://test.finney.opentensor.ai:443"
            else:
                endpoint = self.config.subtensor.chain_endpoint
            self.node = SubstrateInterface(url=endpoint)
            result = self.node.query(module, method, params).value
        
        return result

    def is_git_latest(self) -> bool:
        p = Popen(["git", "rev-parse", "HEAD"], stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        if err:
            return False
        current_commit = out.decode().strip()
        p = Popen(["git", "ls-remote", "origin", "HEAD"], stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        if err:
            return False
        latest_commit = out.decode().split()[0]
        bt.logging.info(
            f"Current commit: {current_commit}, Latest commit: {latest_commit}"
        )
        return current_commit == latest_commit

    def should_restart(self) -> bool:
        # Check if enough time has elapsed since the last update check, if not assume we are up to date.
        if (
            datetime.datetime.now() - self.last_update_check
        ).seconds < self.update_check_interval:
            return False

        self.last_update_check = datetime.datetime.now()
        return not self.is_git_latest()
    
    def get_miner_metadata(self, uid: int) -> Optional[str]:
        """Get metadata for a specific miner UID."""
        return self.metadata_manager.get_miner_metadata(uid)
    
    def get_all_miner_metadata(self) -> Dict[int, str]:
        """Get all miner metadata."""
        return self.metadata_manager.get_all_miner_metadata()

    def new_wandb_run(self):
        """Creates a new wandb run to save information to."""
        # Create a unique run id for this run.
        now = datetime.datetime.now()
        self.wandb_run_start = now
        run_id = now.strftime("%Y-%m-%d_%H-%M-%S")
        name = "validator-" + str(self.my_uid) + "-" + run_id
        self.wandb_run = wandb.init(
            name=name,
            project="sportstensor-vali-logs",
            entity="sportstensor",
            config={
                "uid": self.my_uid,
                "hotkey": self.wallet.hotkey.ss58_address,
                "run_name": run_id,
                "type": "validator",
            },
            allow_val_change=True,
            anonymous="allow",
        )
        bt.logging.debug(f"Started a new wandb run: {name}")

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Generic retry wrapper with exponential backoff."""
        max_retries = 3
        base_delay = 3  # 3 seconds
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    bt.logging.error(f"❌ {func.__name__} failed after {max_retries} attempts: {str(e)}")
                    raise e
                else:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff: 3, 6, 12 seconds
                    bt.logging.warning(f"⚠️ {func.__name__} attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)

    def fetch_trading_history(self) -> List[Dict]:
        def _fetch():
            if self.config.use_synthetic_data:
                # Let's load in synthetic data for now
                with open("tests/advanced_mock_data.json", "r") as f:
                    trading_history = json.load(f)

                # we need to look through the trading history and replace all the hotkeys with the actual hotkeys based on the uid
                for trade in trading_history:
                    if trade['miner_id'] == 170:
                        trade['miner_id'] = 17
                    if trade['miner_id'] is not None and trade['miner_id'] < len(self.metagraph.hotkeys) and trade['is_general_pool'] is False:
                        trade['miner_hotkey'] = self.metagraph.hotkeys[trade['miner_id']]
                    else:
                        # Skip trades with invalid miner_id
                        continue
                return trading_history

            else:
                bt.logging.info(f"Fetching trading history from {self.trading_history_endpoint} for {self.rolling_history_in_days} days")
                
                keypair = self.dendrite.keypair
                hotkey = keypair.ss58_address
                signature = f"0x{keypair.sign(hotkey).hex()}"
                
                # Function to fetch a single batch with offset
                def _fetch_batch(offset: Optional[int] = None):
                    url = f"{self.trading_history_endpoint}?days={self.rolling_history_in_days}&limit={self.trading_history_batch_limit}"
                    if offset is not None:
                        url += f"&offset={offset}"
                    
                    response = requests.get(
                        url, 
                        auth=HTTPBasicAuth(hotkey, signature),
                        timeout=20
                    )
                    response.raise_for_status()
                    api_response = response.json()
                    
                    # Validate response structure
                    if not isinstance(api_response, dict) or "data" not in api_response:
                        raise ValueError(f"Unexpected API response format: {type(api_response)}. Expected dict with 'data' field.")
                    
                    trading_history_batch = api_response["data"]
                    if not isinstance(trading_history_batch, list):
                        raise ValueError(f"Expected 'data' field to be a list, got {type(trading_history_batch)}")
                    
                    # Extract pagination info
                    meta = api_response.get("meta", {})
                    pagination = meta.get("pagination", {})
                    has_more = pagination.get("has_more", False)
                    next_offset = pagination.get("next_offset")
                    
                    return trading_history_batch, has_more, next_offset
                
                # Fetch all batches with pagination
                all_trading_history = []
                offset = None
                batch_num = 1
                
                while True:
                    bt.logging.info(f"Fetching trading history batch {batch_num} (offset={offset})")
                    
                    # Fetch batch with retry logic
                    trading_history_batch, has_more, next_offset = self._retry_with_backoff(_fetch_batch, offset)
                    
                    if not trading_history_batch:
                        bt.logging.warning(f"Empty batch received at offset {offset}")
                        break
                    
                    all_trading_history.extend(trading_history_batch)
                    bt.logging.info(f"Batch {batch_num}: fetched {len(trading_history_batch)} trades (total so far: {len(all_trading_history)})")
                    
                    # Check if there are more pages
                    if not has_more:
                        bt.logging.info(f"Finished fetching all trading history. Total trades: {len(all_trading_history)}")
                        break
                    
                    # Update offset for next batch
                    if next_offset is None:
                        bt.logging.warning("has_more is True but next_offset is None. Stopping pagination.")
                        break
                    
                    offset = next_offset
                    batch_num += 1
                
                return all_trading_history
        
        return self._retry_with_backoff(_fetch)

    def fetch_tao_price(self) -> float:
        def _fetch():
            # Fetch the $TAO price from the API
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()['bittensor']['usd']
        
        return self._retry_with_backoff(_fetch)

    def run(self):
        # The Main Validation Loop.
        bt.logging.info("=========== STARTING SN41 VALIDATOR LOOP ===========")
        while True:
            current_time = datetime.datetime.now(datetime.timezone.utc)
            minutes = current_time.minute

            # Get the current block number and the last update time.
            try:
                should_score_and_set_weights = False
                # Score and set weights every hour between the 5th and 15th minute of the hour
                if minutes == random.randint(5, 15):
                    should_score_and_set_weights = True

                # If metadata manager last full sync is more than 2 hours ago, skip scoring and setting weights
                metadata_stats = self.metadata_manager.get_stats()
                last_full_sync_str = metadata_stats.get("last_full_sync")
                if last_full_sync_str:
                    last_full_sync = datetime.datetime.fromisoformat(last_full_sync_str)
                    # Make timezone-aware if naive (assume UTC for backward compatibility)
                    if last_full_sync.tzinfo is None:
                        last_full_sync = last_full_sync.replace(tzinfo=datetime.timezone.utc)
                    if last_full_sync < current_time - datetime.timedelta(hours=2) and self.config.subtensor.network != "test":
                        bt.logging.warning("Metadata manager last full sync is more than 2 hours ago. Skipping scoring and setting weights.")
                        should_score_and_set_weights = False
                
                if should_score_and_set_weights:
                    # Sync our validator with the metagraph so we have the latest information
                    self.metagraph.sync()

                    all_uids = self.metagraph.uids.tolist()
                    all_hotkeys = self.metagraph.hotkeys

                    # Fetch the trading history and TAO price with retry logic
                    try:
                        # Fetch the trading history
                        trading_history = self.fetch_trading_history()

                        # Write trading history to file if enabled
                        if self.config.write_trading_history:
                            with open("trading_history.json", "w") as f:
                                json.dump(trading_history, f, indent=2)
                            bt.logging.info(f"✅ Trading history written to trading_history.json ({len(trading_history)} trades)")

                        # Fetch the $TAO price
                        tao_price_usd = self.fetch_tao_price()
                        alpha_price_usd = self.metagraph.pool.moving_price * tao_price_usd
                        bt.logging.info(f"TAO price: {tao_price_usd:.2f} USD")
                        bt.logging.info(f"Alpha price: {alpha_price_usd:.2f} USD")

                        current_epoch_budget = alpha_price_usd * TOTAL_MINER_ALPHA_PER_DAY
                        bt.logging.info(f"Current epoch (24h) budget: {current_epoch_budget:.2f} USD")
                        
                    except Exception as e:
                        bt.logging.error(f"❌ Failed to fetch required data for scoring: {str(e)}")
                        bt.logging.warning("⚠️ Skipping scoring and weight setting for this epoch due to fetch failures")
                        continue

                    # Score the miners
                    miner_history, general_pool_history, \
                    miners_scores, general_pool_scores, \
                    miner_budget, general_pool_budget = score_miners(all_uids, all_hotkeys, trading_history, current_epoch_budget)

                    # Print the pool stats
                    print("\n############################## OVERALL POOL STATS ##############################")
                    print_pool_stats(miner_history, general_pool_history)
                    print("##################################################################################\n")
                    print("\n########################## CURRENT EPOCH POOL STATS ############################")
                    print_pool_stats(miner_history, general_pool_history, include_current_epoch=True, 
                                   miner_scores=miners_scores, general_pool_scores=general_pool_scores)
                    print("##################################################################################\n")

                    # Validate the miner profiles
                    miner_profiles = {}
                    miners_to_penalize = []
                    if self.config.use_synthetic_data is False:
                        if 'miner_profiles' in miner_history:
                            miner_profiles = miner_history['miner_profiles']
                        for miner_uid in miner_profiles.keys():
                            if miner_profiles[miner_uid] is None:
                                bt.logging.error(f"❌ Miner {miner_uid} has no profile id defined. This should never happen. Setting score to 0.")
                                miners_to_penalize.append(miner_uid)
                                continue
                            if "," in miner_profiles[miner_uid]:
                                bt.logging.warning(f"❌ Miner {miner_uid} has multiple profile ids defined. {miner_profiles[miner_uid]}. This should never happen. Setting score to 0.")
                                miners_to_penalize.append(miner_uid)
                                continue
                            
                            # Get and check the miner metadata from the metadata manager
                            miner_metadata = self.get_miner_metadata(miner_uid)
                            if miner_metadata is None:
                                bt.logging.warning(f"❌ Miner {miner_uid} has no metadata defined. Setting score to 0.")
                                miners_to_penalize.append(miner_uid)
                                continue
                            
                            # Check if the miner profile id contains the metadata polymarket id as we only save partial polymarket ids to the blockchain
                            if miner_metadata not in miner_profiles[miner_uid]:
                                bt.logging.warning(f"❌ Miner {miner_uid} polymarket ids do not match. {miner_metadata} not found in profile id. Setting score to 0.")
                                miners_to_penalize.append(miner_uid)
                                continue

                            # Log success -- do not log the actual polymarket id for privacy
                            bt.logging.success(f"✅ UID {miner_uid}: Almanac polymarket id matches Bittensor chain metadata polymarket id.")
                    

                    # Calculate the weights for the miners and general pool
                    weights = calculate_weights(miners_scores, general_pool_scores, current_epoch_budget, miner_budget, general_pool_budget, miners_to_penalize, all_uids)
                    self.scores = weights
                    
                    # Update the incentive mechanism weights on the Bittensor blockchain.
                    bt.logging.info(f"Submitting weights to subnet {self.config.netuid}...")
                    result = self.subtensor.set_weights(
                        netuid=self.config.netuid,
                        wallet=self.wallet,
                        uids=self.metagraph.uids,
                        weights=weights,
                        wait_for_inclusion=True
                    )
                    
                    if result:
                        bt.logging.success(f"✅ Successfully set weights on subnet {self.config.netuid}!")
                        bt.logging.info(f"Transaction result: {result}")
                    else:
                        bt.logging.error(f"❌ Failed to set weights on subnet {self.config.netuid}")

                    if self.config.db_score_logging:
                        # Log the scores to the database
                        bt.logging.info(f"Logging scores to database...")
                        log_scores_to_database(miner_history, general_pool_history, miners_scores, general_pool_scores, miner_budget, general_pool_budget, all_hotkeys, weights)
                        bt.logging.success(f"✅ Successfully logged scores to database!")

                else:
                    # Check if we should restart the validator for auto update.
                    if self.config.auto_update and self.should_restart():
                        bt.logging.info(f"Validator is out of date, quitting to restart.")
                        raise KeyboardInterrupt

                    # Check if we should start a new wandb run.
                    if not self.config.wandb.off:
                        if (datetime.datetime.now() - self.wandb_run_start) >= datetime.timedelta(
                            days=1
                        ):
                            bt.logging.info("Current wandb run is more than 1 day old. Starting a new run.")
                            self.wandb_run.finish()
                            self.new_wandb_run()

                    # Only log an update periodically
                    if minutes % 5 == 0:
                        bt.logging.info(f"Not time to score and set weights. Waiting for next hour.")

            except RuntimeError as e:
                bt.logging.error(e)
                traceback.print_exc()

            except KeyboardInterrupt:
                bt.logging.success("Keyboard interrupt detected. Exiting validator.")
                self.metadata_manager.stop()
                exit()

            # Log metadata manager stats every 10 minutes
            if minutes % 10 == 0:
                stats = self.metadata_manager.get_stats()
                bt.logging.info(f"Metadata Manager Stats: {stats}")
            
            # sleep for 1 minute before checking again
            time.sleep(60)

# Run the validator.
if __name__ == "__main__":

    ascii_banner = r"""
    
    ________________________________________________________________________
    

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

    validator = Validator()
    validator.run()
