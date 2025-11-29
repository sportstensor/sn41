
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

Sportstensor is revolutionizing sports prediction through the world's first decentralized network for predictive intelligence. We harness the collective power of global AI talent to create prediction systems that consistently outperform traditional approaches. We've built a competitive ecosystem where independent AI developers and machine learning engineers deploy their predictive models. These models compete by submitting predictions that are scored against real-world outcomes, with rewards flowing directly to the most successful contributors.


## Why is this important? (@TODO: update this section)
Prediction markets are powerful but raw volume doesn’t always mean good information.
Sportstensor fixes this by:

- Rewarding traders for real informational value, not just profit.
- Identifying high-signal participants and routing more flow toward them.
- Improving market accuracy through continuous feedback and scoring.
- Aligning incentives between traders, liquidity, and the broader ecosystem.
- Unlocking a new meta-game, where skilled forecasters can earn daily rewards on top of Polymarket PnL.

## Miner and Validator Functionality

### Miner

- Miners generate **informative probability signals** by trading on Almanac, which routes Polymarket CLOB orders through the miner’s proxy wallet.  
- Every trade becomes a scored prediction within the **Almanac Scoring Engine**, which evaluates accuracy, ROI, timing, and informational value.  
- Miners may use **manual strategies, models, or automated systems**—the scoring is model-agnostic and purely performance-based.  
- High-signal miners earn the largest share of **daily Alpha Token emissions** based on:  
  - Truthful Flow contribution  
  - Informational Efficiency  
  - Volume-adjusted consistency  
  - Historical performance (with decaying memory)

### Validator

- **Metadata Syncing**: Validators continuously sync miner metadata from chain (wallets, proxy addresses, UIDs).  
- **Data Ingestion**: At each epoch, the validator pulls miner trading history from Almanac’s backend (rolling window).  
- **Scoring**: The validator runs the **dual-phase scoring pipeline**:  
  - **Phase 1 — Truthful Flow**: Measures how much quality volume a miner drives.  
  - **Phase 2 — Informational Efficiency**: Adjusts rewards based on ROI consistency and predictive value.  
- **Weight Setting**: After generating scores, validators set miner weights on-chain, determining TAO emissions for the next epoch.

### Scoring and Weights

- Scoring runs **hourly** in the background, always computing rewards for the **previous epoch** while updating weights for the current one.  
- Epochs follow 24-hour windows, and a **decaying memory function** favors recent performance.  
- Validators publish final weights using the scoring output, directly influencing miner rewards.  
- For implementation details, reference the scoring engine in `scoring.py` and the official docs.

---

## Miner Setup and Running Validators

### Setting up a Miner

#### Requirements
- Almanac account linked to Polymarket  
- Almanac account connected to a registered Bittensor coldkey  
- Python 3.10+  
- Pip  
- CPU  

#### Bittensor Wallet and Subnet Registration
You must have a Bittensor wallet and a registered UID on the subnet.  
- Wallet guide: https://docs.learnbittensor.org/keys/wallets  
- Miner registration: https://docs.learnbittensor.org/miners  

#### Almanac and Polymarket Setup
1. Go to **https://almanac.market**  
2. Create an account  
3. Connect your Bittensor coldkey:  
   - Install the Bittensor wallet extension  
   - Import the coldkey tied to your miner UID  
   - Link wallet in Almanac settings  
4. Connect Polymarket credentials if required (via Trading Settings)

#### Miner Metadata Registration
1. Clone the repository:
```bash
git clone https://github.com/sportstensor/sn41/
cd sn41
```
2. Install dependencies:
```bash 
pip install -r requirements.txt
```

3. Register miner metadata:
```bash 
python miner.py
```

## Miner Trading with Almanac dApp

Once linked, miners can trade directly on **https://almanac.market**.  
Validators automatically detect trades, compute scores, and distribute emissions.

---

## Miner Trading with API

For programmatic trading:

1. Complete all onboarding steps above.
2. Use the **Almanac API Trading Client** (`api_trading.py`) to:
   - Generate Polymarket API credentials  
   - Create Almanac trading sessions  
   - Search markets  
   - Place signed CLOB orders  
   - Submit proxy-signed EIP-712 orders  

---

## Running a Validator

### Requirements

- Python 3.10+  
- Pip  
- CPU  

### Weights & Biases

W&B is supported for logging.

Enable with:
```bash
export WANDB_API_KEY=<your_key>
```

### Setup

Clone and enter the repo:
```bash 
git clone https://github.com/sportstensor/sn41/
cd sn41
```
Install pm2 (if not already installed).

Install Python dependencies:
```bash 
pip install -r requirements.txt
```

### Run Auto-Updating Validator with PM2 (recommended)

```bash 
pm2 start vali_auto_update.sh --name sn41-validator -- \
    --netuid 41 \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --axon.port {port} \
    --logging.debug
```


### Run Basic Validator with PM2

```bash 
pm2 start validator.py --name sn41-validator -- \
    --netuid 41 \
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
