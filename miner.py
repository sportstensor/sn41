import os
import time
import argparse
import traceback
import json
import bittensor as bt
from typing import Tuple

class Miner:
    def __init__(self, interactive_mode=True, wallet_name=None, hotkey_name=None, network=None, polymarket_id=None):
        self.interactive_mode = interactive_mode
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.network = network
        self.polymarket_id = polymarket_id
        
        if not interactive_mode:
            # Only initialize config and bittensor objects if not in interactive mode
            self.config = self.get_config()
            self.setup_logging()
            self.setup_bittensor_objects()

    def get_config(self):
        # Set up the configuration parser
        parser = argparse.ArgumentParser()
        # TODO: Add your custom miner arguments to the parser.
        parser.add_argument('--custom', default='my_custom_value', help='Adds a custom value to the parser.')
        # Adds override arguments for network and netuid.
        parser.add_argument('--netuid', type=int, default=1, help="The chain subnet uid.")
        # Adds subtensor specific arguments.
        bt.Subtensor.add_args(parser)
        # Adds logging specific arguments.
        bt.logging.add_args(parser)
        # Adds wallet specific arguments.
        bt.wallet.add_args(parser)
        # Adds axon specific arguments.
        bt.axon.add_args(parser)
        
        # If wallet info provided during initialization, use it
        if self.wallet_name:
            parser.set_defaults(wallet__name=self.wallet_name)
        if self.hotkey_name:
            parser.set_defaults(wallet__hotkey=self.hotkey_name)
            
        # Parse the arguments.
        config = bt.config(parser)
        # Set up logging directory
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey_str,
                config.netuid,
                'miner',
            )
        )
        # Ensure the directory for logging exists.
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        # Activate Bittensor's logging with the set configurations.
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        # Prevent duplicate "INFO:bittensor:..." lines if a dependency triggers
        # logging.basicConfig() (e.g. substrateinterface's logging.debug()).
        import logging
        logging.getLogger("bittensor").propagate = False
        bt.logging.info(f"Running miner for subnet: {self.config.netuid} on network: {self.config.subtensor.network} with config:")
        bt.logging.info(self.config)

    def setup_bittensor_objects(self):
        # Initialize Bittensor miner objects
        bt.logging.info("Setting up Bittensor objects.")

        # Initialize wallet.
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        # Initialize subtensor.
        self.subtensor = bt.Subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        # Initialize metagraph.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(f"\nYour miner: {self.wallet} is not registered to chain connection: {self.subtensor} \nRun 'btcli register' and try again.")
            exit()
        else:
            # Each miner gets a unique identity (UID) in the network.
            self.my_subnet_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            bt.logging.info(f"Running miner on uid: {self.my_subnet_uid}")

    def setup_axon(self):
        # Build and link miner functions to the axon.
        self.axon = bt.axon(wallet=self.wallet, port=self.config.axon.port)

        # Attach functions to the axon.
        bt.logging.info(f"Attaching forward function to axon.")
        self.axon.attach(
            forward_fn=self.dummy,
            blacklist_fn=self.blacklist_fn,
        )

        # Serve the axon.
        bt.logging.info(f"Serving axon on network: {self.config.subtensor.network} with netuid: {self.config.netuid}")
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        bt.logging.info(f"Axon: {self.axon}")

        # Start the axon server.
        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()

    def run(self):
        self.setup_axon()

        # Keep the miner alive.
        bt.logging.info(f"Starting main loop")
        step = 0
        while True:
            try:
                # Periodically update our knowledge of the network graph.
                if step % 60 == 0:
                    self.metagraph.sync()
                    log = (
                        f'Block: {self.metagraph.block.item()} | '
                        f'Incentive: {self.metagraph.I[self.my_subnet_uid]} | '
                    )
                    bt.logging.info(log)
                step += 1
                time.sleep(1)

            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success('Miner killed by keyboard interrupt.')
                break
            except Exception as e:
                bt.logging.error(traceback.format_exc())
                continue

    @staticmethod
    def submit_metadata_to_chain(
        wallet_name: str,
        hotkey_name: str,
        netuid: int,
        metadata_dict: dict,
        network: str = "finney",  # or "test" for testnet
        max_retries: int = 10,
        retry_delay: int = 120,  # seconds between retries
    ):
        """
        Submit metadata to the Bittensor blockchain.
        
        Args:
            wallet_name: Name of your wallet (coldkey)
            hotkey_name: Name of your hotkey
            netuid: The subnet UID you're registered on
            metadata_dict: Dictionary containing your metadata (e.g., {"polymarket_id": "0x4Cd..."})
            network: Network to connect to ("finney" for mainnet, "test" for testnet)
            max_retries: Maximum number of retry attempts
            retry_delay: Seconds to wait between retry attempts
        
        Returns:
            bool: True if successful, False otherwise
        """
        
        # Initialize wallet and subtensor
        bt.logging.info(f"Initializing wallet: {wallet_name}/{hotkey_name}")
        wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
        
        bt.logging.info(f"Connecting to network: {network}")
        subtensor = bt.Subtensor(network=network)
        
        # Extract polymarket ID and submit only first 5 characters
        polymarket_id = metadata_dict.get('polymarket_id', '')
        
        # IMPORTANT: Only submit first 5 characters of the polymarket ID
        # This ensures we store just the beginning of the Polygon address on-chain
        metadata_to_submit = polymarket_id[:5]
        
        bt.logging.info(f"Full polymarket ID: {polymarket_id}")
        bt.logging.info(f"Submitting (first 5 chars of ID): {metadata_to_submit}")
        
        # Verify hotkey is registered
        try:
            metagraph = subtensor.metagraph(netuid)
            if wallet.hotkey.ss58_address not in metagraph.hotkeys:
                bt.logging.error(f"Hotkey {wallet.hotkey.ss58_address} is not registered on subnet {netuid}")
                return False
            bt.logging.success(f"Hotkey verified on subnet {netuid}")
        except Exception as e:
            bt.logging.error(f"Failed to verify registration: {e}")
            return False
        
        # Submit metadata with retry loop
        # The commit function is rate-limited to once every 100 blocks (~20 minutes)
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            bt.logging.info(f"Attempt {attempt}/{max_retries} to commit metadata...")
            
            try:
                # Use subtensor.commit() to submit metadata
                # This function is rate-limited and requires the hotkey to be registered
                success = subtensor.commit(
                    wallet=wallet,
                    netuid=netuid,
                    data=metadata_to_submit
                )
                
                if success:
                    bt.logging.success(f"Successfully committed metadata to chain: {metadata_to_submit}")
                    bt.logging.info(f"Hotkey: {wallet.hotkey.ss58_address}")
                    bt.logging.info(f"Subnet: {netuid}")
                    return True
                else:
                    bt.logging.warning("Commit returned False, likely rate-limited")
                    
            except Exception as e:
                bt.logging.error(f"Failed to commit metadata: {e}")
                bt.logging.debug(traceback.format_exc())
            
            # Wait before retrying
            if attempt < max_retries:
                bt.logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        bt.logging.error(f"Failed to commit metadata after {max_retries} attempts")
        return False


    @staticmethod
    def retrieve_metadata_from_chain(
        hotkey_address: str,
        netuid: int,
        network: str = "finney",
    ):
        """
        Retrieve committed metadata from the chain for a specific hotkey.
        
        Args:
            hotkey_address: The ss58 address of the hotkey
            netuid: The subnet UID
            network: Network to connect to
        
        Returns:
            str: The committed data, or None if not found
        """
        bt.logging.info(f"Retrieving metadata for hotkey: {hotkey_address}")
        subtensor = bt.Subtensor(network=network)
        
        try:
            # Get the UID for this hotkey
            metagraph = subtensor.metagraph(netuid)
            
            if hotkey_address not in metagraph.hotkeys:
                bt.logging.error(f"Hotkey not found on subnet {netuid}")
                return None
            
            # Find the UID
            uid = metagraph.hotkeys.index(hotkey_address)
            
            # Retrieve the commitment
            commitment = subtensor.get_commitment(netuid=netuid, uid=uid)
            
            bt.logging.success(f"Retrieved commitment: {commitment}")
            return commitment
            
        except Exception as e:
            bt.logging.error(f"Failed to retrieve metadata: {e}")
            bt.logging.debug(traceback.format_exc())
            return None

    def validate_polygon_address(self, address: str) -> bool:
        """
        Basic validation for Polygon address format.
        Polygon addresses are Ethereum-compatible (0x followed by 40 hex characters).
        """
        if not address.startswith('0x'):
            return False
        if len(address) != 42:  # 0x + 40 hex chars
            return False
        try:
            int(address[2:], 16)  # Check if the rest is valid hex
            return True
        except ValueError:
            return False

    def validate_wallet_exists(self, wallet_name: str, hotkey_name: str) -> bool:
        """
        Check if the wallet and hotkey exist on the local system.
        """
        try:
            wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
            # Try to access the hotkey to verify it exists
            _ = wallet.hotkey.ss58_address
            return True
        except Exception as e:
            bt.logging.error(f"Wallet validation failed: {e}")
            return False

    def validate_registration(self, wallet_name: str, hotkey_name: str, netuid: int, network: str) -> bool:
        """
        Check if the hotkey is registered on the specified subnet.
        """
        try:
            wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
            subtensor = bt.Subtensor(network=network)
            metagraph = subtensor.metagraph(netuid)
            
            if wallet.hotkey.ss58_address not in metagraph.hotkeys:
                return False
            return True
        except Exception as e:
            bt.logging.error(f"Registration validation failed: {e}")
            return False

    def get_network_netuid(self, network: str) -> int:
        """
        Get the NETUID based on the network.
        finney (mainnet) = 41, test (testnet) = 172
        """
        if network.lower() == 'finney':
            return 41
        elif network.lower() == 'test':
            return 172
        else:
            raise ValueError(f"Unknown network: {network}. Use 'finney' or 'test'")

    def interactive_setup(self):
        """
        Interactive setup process for miners.
        Prompts user for wallet name, network, and polymarket ID.
        Skips prompts if CLI arguments are already provided.
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
        print("This script will help you configure your miner for the subnet.")
        print("You'll need to provide your wallet information and Polymarket profile ID.\n")
        
        # Get wallet name (skip if provided via CLI)
        if self.wallet_name:
            wallet_name = self.wallet_name
            print(f"✅ Wallet name: {wallet_name}")
        else:
            while True:
                wallet_name = input("Enter your Bittensor wallet name (coldkey): ").strip()
                if wallet_name:
                    break
                print("❌ Wallet name cannot be empty. Please try again.")
        
        # Get hotkey name (skip if provided via CLI)
        if self.hotkey_name:
            hotkey_name = self.hotkey_name
            print(f"✅ Hotkey name: {hotkey_name}")
        else:
            hotkey_name = input("Enter your hotkey name (press Enter to use same as wallet name): ").strip()
            if not hotkey_name:
                hotkey_name = wallet_name
            print(f"✅ Hotkey name: {hotkey_name}")
        
        # Get network (skip if provided via CLI)
        if self.network:
            network = self.network.lower()
            if network == 'finney':
                netuid = 41
            elif network == 'test':
                netuid = 172
            else:
                print(f"❌ Invalid network from CLI: {network}. Use 'finney' or 'test'")
                return False
            print(f"✅ Network: {network} (NETUID: {netuid})")
        else:
            while True:
                print("\nSelect network:")
                print("1. finney (mainnet) - NETUID 41")
                print("2. test (testnet) - NETUID 172")
                network_choice = input("Enter choice (1 or 2): ").strip()
                
                if network_choice == '1':
                    network = 'finney'
                    netuid = 41
                    break
                elif network_choice == '2':
                    network = 'test'
                    netuid = 172
                    break
                else:
                    print("❌ Invalid choice. Please enter 1 or 2.")
            print(f"\n✅ Selected network: {network} (NETUID: {netuid})")
        
        # Now validate wallet and registration before asking for polymarket ID
        print(f"\n🔍 Validating wallet and registration...")
        
        # Validate wallet exists
        print("🔍 Checking wallet...")
        if not self.validate_wallet_exists(wallet_name, hotkey_name):
            print(f"❌ Wallet '{wallet_name}' with hotkey '{hotkey_name}' not found. Please make sure your wallet is properly set up.")
            print("You can create a wallet using: btcli wallet new_coldkey --wallet.name <name>")
            print("You can create a hotkey using: btcli wallet new_hotkey --wallet.name <name> --wallet.hotkey <hotkey_name>")
            return False
        
        print("✅ Wallet found!")
        
        # Validate registration
        print(f"🔍 Checking registration on subnet {netuid}...")
        if not self.validate_registration(wallet_name, hotkey_name, netuid, network):
            print(f"❌ Hotkey not registered on subnet {netuid} ({network} network).")
            print(f"Please register using: btcli subnet register --netuid {netuid} --subtensor.network {network} --wallet.name {wallet_name} --wallet.hotkey {hotkey_name}")
            return False
        
        print("✅ Hotkey is registered on the subnet!")
        
        # Check for existing metadata
        print(f"🔍 Checking for existing metadata...")
        wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
        existing_commitment = self.retrieve_metadata_from_chain(
            hotkey_address=wallet.hotkey.ss58_address,
            netuid=netuid,
            network=network,
        )
        
        if existing_commitment and existing_commitment != 'None' and existing_commitment.strip():
            print(f"⚠️  Existing metadata found: {existing_commitment}")
            print("Note: Only the first 5 characters of your Almanac/Polymarket Ethereum address (EOA) are stored on-chain for privacy.")
        else:
            print("✅ No existing metadata found.")
        
        # Now ask if user wants to proceed with metadata submission
        print("\n" + "="*60)
        print("📋 VALIDATION COMPLETE")
        print("="*60)
        print(f"Wallet Name: {wallet_name}")
        print(f"Hotkey Name: {hotkey_name}")
        print(f"Network: {network}")
        print(f"NETUID: {netuid}")
        if existing_commitment and existing_commitment != 'None' and existing_commitment.strip():
            print(f"Existing Metadata: {existing_commitment}")
        print("="*60)
        
        proceed = input("\nDo you want to submit/update your Almanac/Polymarket Ethereum address (EOA) metadata? (y/N): ").strip().lower()
        if proceed not in ['y', 'yes']:
            print("✅ Setup completed. No metadata changes made.")
            return True  # Return True since validation was successful
        
        # Get polymarket ID (skip if provided via CLI)
        if self.polymarket_id:
            polymarket_id = self.polymarket_id
            print(f"\n✅ Polymarket ID: {polymarket_id}")
        else:
            while True:
                polymarket_id = input("\nEnter your Almanac/Polymarket Ethereum address (EOA) starting with 0x: ").strip()
                if self.validate_polygon_address(polymarket_id):
                    break
                print("❌ Invalid EOA address format. Please enter a valid address (0x followed by 40 hex characters).")
            print(f"\n✅ EOA Address: {polymarket_id}")
        
        # Final confirmation for metadata submission
        print("\n" + "="*60)
        print("📋 METADATA SUBMISSION SUMMARY")
        print("="*60)
        print(f"Wallet Name: {wallet_name}")
        print(f"Hotkey Name: {hotkey_name}")
        print(f"Network: {network}")
        print(f"NETUID: {netuid}")
        print(f"Polymarket ID: {polymarket_id}")
        print("="*60)
        
        if existing_commitment and existing_commitment != 'None' and existing_commitment.strip():
            confirm = input("\nProceed with metadata update? (y/N): ").strip().lower()
        else:
            confirm = input("\nProceed with metadata submission? (y/N): ").strip().lower()
            
        if confirm != 'y' and confirm != 'yes':
            print("❌ Metadata submission cancelled.")
            return True  # Return True since validation was successful
        
        # Submit metadata
        print("\n🚀 Submitting metadata to blockchain...")
        metadata = {
            "polymarket_id": polymarket_id
        }
        
        success = self.submit_metadata_to_chain(
            wallet_name=wallet_name,
            hotkey_name=hotkey_name,
            netuid=netuid,
            metadata_dict=metadata,
            network=network,
        )
        
        if success:
            print("\n✅ Metadata successfully committed to chain!")
            
            # Retrieve and display the commitment
            retrieved = self.retrieve_metadata_from_chain(
                hotkey_address=wallet.hotkey.ss58_address,
                netuid=netuid,
                network=network,
            )
            print(f"Retrieved commitment: {retrieved}")
            print("\n📝 Note: Only the first 5 characters of your Almanac/Polymarket ID are stored on-chain for privacy.")
            print(f"Your full Almanac/Polymarket ID: {polymarket_id}")
            print(f"Stored on-chain: {polymarket_id[:5]}")
            print("\n🎉 Setup complete! Your miner is now configured.")
            return True
        else:
            print("\n❌ Failed to commit metadata. Please try again later.")
            return False

    def run_miner_setup(self):
        """
        Legacy setup function - now calls interactive setup.
        """
        return self.interactive_setup()

def main():
    """
    Main function that handles both interactive and CLI modes.
    """
    parser = argparse.ArgumentParser(description='SN41 Almanac Miner Setup')
    parser.add_argument('--wallet.name', dest='wallet_name', help='Wallet name (coldkey)')
    parser.add_argument('--wallet.hotkey', dest='hotkey_name', help='Hotkey name')
    parser.add_argument('--subtensor.network', dest='network', help='Network (finney or test)')
    parser.add_argument('--polymarket.id', dest='polymarket_id', help='Almanac/Polymarket ID (EOA address)')
    
    args = parser.parse_args()
    
    # Check if we have all required CLI arguments for full CLI mode
    has_all_cli_args = args.wallet_name and args.hotkey_name and args.polymarket_id and args.network
    
    if has_all_cli_args:
        # Full CLI mode - no interactive prompts
        print("🔧 Running in CLI mode...")
        
        # Validate network and get NETUID
        if args.network.lower() == 'finney':
            netuid = 41
        elif args.network.lower() == 'test':
            netuid = 172
        else:
            print(f"❌ Invalid network: {args.network}. Use 'finney' or 'test'")
            exit(1)
        
        # Create miner instance for CLI mode
        miner = Miner(interactive_mode=False, wallet_name=args.wallet_name, hotkey_name=args.hotkey_name)
        
        # Validate inputs
        if not miner.validate_polygon_address(args.polymarket_id):
            print("❌ Invalid Polygon address format.")
            exit(1)
        
        # Validate wallet exists
        if not miner.validate_wallet_exists(args.wallet_name, args.hotkey_name):
            print(f"❌ Wallet '{args.wallet_name}' with hotkey '{args.hotkey_name}' not found.")
            exit(1)
        
        # Validate registration
        if not miner.validate_registration(args.wallet_name, args.hotkey_name, netuid, args.network):
            print(f"❌ Hotkey not registered on subnet {netuid} ({args.network} network).")
            exit(1)
        
        # Check for existing metadata in CLI mode
        print("🔍 Checking for existing metadata...")
        wallet = bt.wallet(name=args.wallet_name, hotkey=args.hotkey_name)
        existing_commitment = miner.retrieve_metadata_from_chain(
            hotkey_address=wallet.hotkey.ss58_address,
            netuid=netuid,
            network=args.network,
        )
        
        if existing_commitment and existing_commitment != 'None' and existing_commitment.strip():
            print(f"⚠️  Existing metadata found: {existing_commitment}")
            print("Note: Only the first 5 characters of the original Polymarket ID are stored on-chain.")
            print("Use interactive mode if you want to choose whether to overwrite existing metadata.")
        else:
            print("✅ No existing metadata found.")
        
        # Submit metadata
        print("\n🚀 Submitting metadata to blockchain...")
        metadata = {"polymarket_id": args.polymarket_id}
        
        success = miner.submit_metadata_to_chain(
            wallet_name=args.wallet_name,
            hotkey_name=args.hotkey_name,
            netuid=netuid,
            metadata_dict=metadata,
            network=args.network,
        )
        
        if success:
            print("✅ Metadata successfully committed to chain!")
            retrieved = miner.retrieve_metadata_from_chain(
                hotkey_address=wallet.hotkey.ss58_address,
                netuid=netuid,
                network=args.network,
            )
            print(f"Retrieved commitment: {retrieved}")
            print("\n📝 Note: Only the first 5 characters of your Polymarket ID are stored on-chain due to blockchain constraints.")
            print(f"Your full Polymarket ID: {args.polymarket_id}")
            print(f"Stored on-chain: {args.polymarket_id[:5]}")
            print("🎉 Setup complete!")
        else:
            print("❌ Failed to commit metadata.")
            exit(1)
    else:
        # Interactive mode with optional CLI arguments
        try:
            miner = Miner(
                interactive_mode=True, 
                wallet_name=args.wallet_name, 
                hotkey_name=args.hotkey_name,
                network=args.network,
                polymarket_id=args.polymarket_id
            )
            success = miner.run_miner_setup()
            if not success:
                exit(1)
        except KeyboardInterrupt:
            print("\n❌ Setup cancelled by user.")
            exit(1)
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            bt.logging.error(traceback.format_exc())
            exit(1)

# Run the miner.
if __name__ == "__main__":
    main()
