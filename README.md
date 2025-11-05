
<div align="center">

```
                    $$$$$$\  $$\       $$\      $$\  $$$$$$\  $$\   $$\  $$$$$$\   $$$$$$\  
                    $$  __$$\ $$ |      $$$\    $$$ |$$  __$$\ $$$\  $$ |$$  __$$\ $$  __$$\ 
                    $$ /  $$ |$$ |      $$$$\  $$$$ |$$ /  $$ |$$$$\ $$ |$$ /  $$ |$$ /  \__|
                    $$$$$$$$ |$$ |      $$\$$\$$ $$ |$$$$$$$$ |$$ $$\$$ |$$$$$$$$ |$$ |      
                    $$  __$$ |$$ |      $$ \$$$  $$ |$$  __$$ |$$ \$$$$ |$$  __$$ |$$ |      
                    $$ |  $$ |$$ |      $$ |\$  /$$ |$$ |  $$ |$$ |\$$$ |$$ |  $$ |$$ |  $$\ 
                    $$ |  $$ |$$$$$$$$\ $$ | \_/ $$ |$$ |  $$ |$$ | \$$ |$$ |  $$ |\$$$$$$  |
                    \__|  \__|\________|\__|     \__|\__|  \__|\__|  \__|\__|  \__| \______/ 
```

# The future of prediction algorithms

</div>

- [Introduction](#introduction)
- [Why is this important?](#why-is-this-important)
- [Miner and Validator Functionality](#miner-and-validator-functionality)
  - [Miner](#miner)
  - [Validator](#validator)
- [Miner setup and running Validators](#miner-setup-and-running-validators)
  - [Setting up a Miner](#setting-up-a-miner)
  - [Running a Validator](#running-a-validator)
- [Community](#community)
- [License](#license)

## Introduction (@TODO: update this section)

Welcome to Sportstensor—the convergence of cutting-edge technology and sports data analytics. We are pioneering unprecedented innovation in sports prediction algorithms, powered by the Bittensor network.

The Sportstensor subnet is designed to incentivize the discovery of competitive advantages such as 'edge' over closing market odds, enabling top miners within the network to establish machine-driven dominance across the sports prediction landscape.

[DeepWiki Summary](https://deepwiki.com/sportstensor/sportstensor)

## Why is this important? (@TODO: update this section)
- Closing odds represent the pinnacle of market efficiency, determined by thousands of advanced machine learning algorithms.
- Our subnet fosters the development of true machine intelligence by outperforming competing algorithms in a highly competitive AI-versus-AI environment.
- Even with sophisticated models, bettors struggle to be profitable, as bookmakers frequently impose strict limits on consistent winners.
- There is substantial demand for high-performing predictive AI from betting operators, financial firms, syndicates, and algorithmic traders seeking more accurate models.
- We attract top AI and machine learning talent from diverse industries, encouraging them to revolutionize sports prediction markets.
- By decentralizing the creation and improvement of predictive models, we reduce reliance on any single entity or algorithm, enhancing resilience and driving innovation in the sports prediction market.

## Miner and Validator Functionality

### Miner (@TODO: update this section)

- Miners leverage their edge on a variety of curated markets via trading on Almanac.
- Collateral requirements reduce noise and the best signals rise to the top.
- Miners are open to utilize trained machine learning models or whatever methods they choose to make +EV probabilities.
- Miners who consistently win and drive substantial volume will be rewarded the most.

### Validator

- **Metadata Syncing**: The validator operates in an endless loop, syncing the subnet metadata submitted to the chain by miners. This metadata gets stored in a local state file and used during the validation and scoring steps.
- **Prediction Scoring**: At specific intervals, the validator queries all miner trading history over `ROLLING_HISTORY_IN_DAYS` days, runs various validation checks, calculates the rewards, and sets weights onchain.

### Scoring and Weights
 
- Incentives and scores are calculated every hour in a background thread.
- Validators are always calculating the previous epoch rewards based on each miner's historical performance.
  - For example, let's say the time just became 12a midnight on Monday. Validators will be calculating the rewards for Sunday and setting weights accordingly through the current epoch (Monday).
- Incentive scores are calculated through a series of complex algorithms. Please see our whitepaper for more details. Also analyze `scoring.py`.
- Validators set the miners' weights on the chain based on these scores.

## Miner setup and running Validators

### Setting up a Miner
#### Requirements
- Polymarket account
- Almanac account
- Almanac account connected to registered Bittensor coldkey
- Python 3.10+
- Pip
- CPU

#### Bittensor Wallet and Subnet Registration
You will need a Bittensor wallet and a registered UID on subnet in order to mine.
Bittensor Wallet help: [https://docs.learnbittensor.org/keys/wallets](https://docs.learnbittensor.org/keys/wallets)
Subnet registration help: [https://docs.learnbittensor.org/miners](https://docs.learnbittensor.org/miners)

#### Almanac and Polymarket Setup (add screenshots?)
1. Visit [Almanac](https://almanac.market) and click 'Create Account'
2. Complete all account creation steps
3. Complete one-time Bittensor connection to your Almanac account
- Download and install the [Bittensor wallet extension](https://www.bittensor.com/wallet)
- Create/import the coldkey that registered your miner's hotkey and UID
- Connect your wallet to Almanac in account settings
- Confirm the selection of your registered hotkey and UID

#### Miner Metadata Registration
1. To start, clone the repository and `cd` to it:
```bash
git clone https://github.com/sportstensor/sn41/
cd sn41
```
2. Next, install the requirements: `pip install -r requirements.txt`
3. Run `python miner.py` and follow prompts to submit your Polymarket wallet address to the subnet blockchain metadata.
- Run `python miner.py --subtensor.network test --netuid 172` for testnet

#### Miner Trading with Almanac dApp
Miners are more than welcome to trade directly on [Almanac](https://almanac.market). Once the steps above are completed, use the app to make trades and validators will automoatically pick up your trading history performance and score you accordingly.

#### Miner Trading with API
If you would like to programmatically trade on Almanac, you will need to:
- Complete all the setup steps defined above
- Integrate API trading logic into your workflow.

--- ADD INFO ABOUT THE API TRADING LOGIC ---


### Running a Validator
#### Requirements
- Python 3.10+
- Pip
- CPU

#### Weights & Biases
It is recommended to utilize W&B. Set environment variable with `export WANDB_API_KEY=<your API key>`. Alternatively, you can disable W&B with --wandb.off

#### Setup
1. To start, clone the repository and `cd` to it:
```bash
git clone https://github.com/sportstensor/sn41/
cd sn41
```
2. Install pm2 if you don't already have it: [pm2.io](https://pm2.io/docs/runtime/guide/installation/).
3. Next, install the requirements: `pip install -r requirements.txt`

#### Run auto-updating validator with PM2 (recommended)
```bash
pm2 start vali_auto_update.sh --name sn41-validator -- \
    --netuid 41 \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --axon.port {port} \
    --logging.debug
```
> [!NOTE]
> You might need to adjust "python" to "python3" within the `vali_auto_update.sh` depending on your preferred system python.

#### Run basic validator with PM2
```bash
pm2 start validator.py --name sn41-validator -- \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --axon.port {port} \
    --logging.debug
```

## Environments

| Network | Netuid |
| ----------- | -----: |
| Mainnet     |     41 |
| Testnet     |    172 |

## Community

Join the vibrant Bittensor community and find our channel `#פ • sporτsτensor • 41` on [Discord](https://discord.gg/bittensor).

## License

The SN41 Sportstensor subnet is released under the [MIT License](./LICENSE).

---
</div>