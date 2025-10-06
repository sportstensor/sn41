import os
import argparse
import traceback
import bittensor as bt
import datetime
import time
import wandb
from subprocess import Popen, PIPE
from typing import Optional, Dict

from substrateinterface import SubstrateInterface
from metadata_manager import MetadataManager


class Validator:
    def __init__(self):
        self.config = self.get_config()
        self.setup_logging()
        self.setup_bittensor_objects()
        self.last_update = 0
        self.current_block = 0
        self.node = SubstrateInterface(url=self.config.subtensor.chain_endpoint)
        self.tempo = self.node_query('SubtensorModule', 'Tempo', [self.config.netuid])
        self.moving_avg_scores = [1.0] * len(self.metagraph.S)
        self.alpha = 0.1

        # Set up auto update.
        self.last_update_check = datetime.datetime.now()
        self.update_check_interval = 60 * 60 * 24  # 24 hours
        
        # Set up metadata manager
        self.metadata_manager = MetadataManager(
            netuid=self.config.netuid,
            network=self.config.subtensor.network,
            state_file=f"validator_state_{self.config.netuid}.json"
        )
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

    def get_config(self):
        # Set up the configuration parser.
        parser = argparse.ArgumentParser()
        # TODO: Add your custom validator arguments to the parser.
        #parser.add_argument('--custom', default='my_custom_value', help='Adds a custom value to the parser.')
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
        bt.logging.info("Building validation weights.")
        self.scores = [1.0] * len(self.metagraph.S)
        bt.logging.info(f"Weights: {self.scores}")

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
            project="sn41-vali-logs",
            entity="sn41",
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

    def run(self):
        # The Main Validation Loop.
        bt.logging.info("=========== STARTING SN41 VALIDATOR ===========")
        while True:
            current_time = datetime.datetime.utcnow()
            minutes = current_time.minute
            hour = current_time.hour

            # Get the current block number and the last update time.
            try:
                self.current_block = self.node_query('System', 'Number', [])
                last_update_data = self.node_query('SubtensorModule', 'LastUpdate', [self.config.netuid])
                
                # Check if we have validator permit and stake
                """
                try:
                    validator_permit = self.node_query('SubtensorModule', 'ValidatorPermit', [self.config.netuid])
                    if isinstance(validator_permit, list) and len(validator_permit) > self.my_uid:
                        permit_status = validator_permit[self.my_uid]
                        bt.logging.info(f"Validator permit: {permit_status}")
                    else:
                        bt.logging.warning(f"Could not get validator permit for UID {self.my_uid}")
                except Exception as e:
                    bt.logging.warning(f"Could not query validator permit: {e}")
                """
                
                if isinstance(last_update_data, list) and len(last_update_data) > self.my_uid:
                    last_update_block = last_update_data[self.my_uid]
                    self.last_update = self.current_block - last_update_block
                    #bt.logging.debug(f"Last weight submission was {last_update_block} blocks ago (block {last_update_block})")
                elif isinstance(last_update_data, list) and len(last_update_data) == 0:
                    # New validator - no LastUpdate data yet
                    self.last_update = 0
                    bt.logging.info(f"🆕 New validator detected! No previous weight submissions found.")
                    bt.logging.info(f"Ready to submit weights when tempo ({self.tempo + 1}) blocks have passed.")
                else:
                    # Unexpected data format or UID not found
                    self.last_update = 0
                    bt.logging.warning(f"⚠️  Unexpected LastUpdate data format for UID {self.my_uid}: {last_update_data}")
                    bt.logging.warning(f"Treating as new validator. Will be ready to submit weights in {self.tempo + 1} blocks.")

                # set weights once every tempo + 1, or immediately for new validators
                if self.last_update == 0:
                    # New validator - submit weights immediately
                    should_set_weights = True
                    bt.logging.info(f"🆕 New validator: Submitting initial weights immediately!")
                else:
                    # Existing validator - wait for tempo + 1 blocks since last update
                    should_set_weights = self.last_update > self.tempo + 1
                
                if should_set_weights:
                    total = sum(self.moving_avg_scores)
                    weights = [score / total for score in self.moving_avg_scores]
                    if self.last_update == 0:
                        bt.logging.info(f"🎉 Setting initial weights: {weights}")
                    else:
                        bt.logging.info(f"Setting weights: {weights}")
                    
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
                    
                    # Sync our validator with the metagraph
                    self.metagraph.sync()

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
                    if self.last_update == 0:
                        bt.logging.info(f"🆕 New validator: Will submit initial weights next cycle.")
                    elif minutes % 2 == 0:
                            bt.logging.info(f"Last update: {self.last_update} blocks ago. ~{self.tempo + 1 - self.last_update} blocks until setting weights.")

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
    validator = Validator()
    validator.run()
