
<div align="center">

```
   SPORTSTENSOR PRESENTS
--------------------------------------------------------------------------------------------------
   █████████   █████       ██████   ██████   █████████   ██████   █████   █████████     █████████ 
  ███▒▒▒▒▒███ ▒▒███       ▒▒██████ ██████   ███▒▒▒▒▒███ ▒▒██████ ▒▒███   ███▒▒▒▒▒███   ███▒▒▒▒▒███
 ▒███    ▒███  ▒███        ▒███▒█████▒███  ▒███    ▒███  ▒███▒███ ▒███  ▒███    ▒███  ███     ▒▒▒ 
 ▒███████████  ▒███        ▒███▒▒███ ▒███  ▒███████████  ▒███▒▒███▒███  ▒███████████ ▒███         
 ▒███▒▒▒▒▒███  ▒███        ▒███ ▒▒▒  ▒███  ▒███▒▒▒▒▒███  ▒███ ▒▒██████  ▒███▒▒▒▒▒███ ▒███         
 ▒███    ▒███  ▒███      █ ▒███      ▒███  ▒███    ▒███  ▒███  ▒▒█████  ▒███    ▒███ ▒▒███     ███
 █████   █████ ███████████ █████     █████ █████   █████ █████  ▒▒█████ █████   █████ ▒▒█████████ 
▒▒▒▒▒   ▒▒▒▒▒ ▒▒▒▒▒▒▒▒▒▒▒ ▒▒▒▒▒     ▒▒▒▒▒ ▒▒▒▒▒   ▒▒▒▒▒ ▒▒▒▒▒    ▒▒▒▒▒ ▒▒▒▒▒   ▒▒▒▒▒   ▒▒▒▒▒▒▒▒▒  
```                                                                                                                                         

</div>

- [Introduction](#introduction)
- [How it Works](#how-it-works)
- [Miner and Validator Functionality](#miner-and-validator-functionality)
  - [Miner](#miner)
  - [Validator](#validator)
- [Miner setup and running Validators](#miner-setup-and-running-validators)
  - [Setting up a Miner](#setting-up-a-miner)
  - [Running a Validator](#running-a-validator)
- [Community](#community)
- [License](#license)

## Introduction

Sportstensor operates the world's first decentralized competition network for sports prediction. We connect global AI talent in a competitive ecosystem where independent developers deploy predictive models, compete against real-world outcomes, and earn rewards based on accuracy.

Almanac is the front end to Sportstensor, a prediction market interface that makes competing and submitting predictions by trading simpler and much more accessible.

## How It Works 
We implement a two-phase optimization system that rewards miners based on 
their historical trading performance. The mechanism distributes a fixed budget among eligible 
participants, prioritizing those who demonstrate consistent profitability and trading volume.

The system tracks trading activity over a rolling 30-day window, organizing trades into daily 
epochs. For each epoch, it:

1. Calculates Performance Metrics:
   - ROI (Return on Investment): Profit divided by trading volume
   - Qualified Volume: Volume from winning trades (after fees)
   - Trailing Performance: Historical performance across all epochs

2. Applies Eligibility Gates:
   - Minimum ROI threshold (prevents rewarding unprofitable traders)
   - Minimum volume requirement (ensures meaningful participation)
   - Build-up period: Traders must demonstrate consistent activity over multiple epochs

3. Runs Two-Phase Optimization:
   Phase 1: Maximizes the total qualified volume that can be funded within budget constraints
   Phase 2: Redistributes payouts to favor higher-ROI traders while maintaining volume targets

4. Allocates Tokens:
   - Converts optimized scores into token weights
   - Distributes rewards proportionally based on funded volume and signal strength (ROI)
   - Enforces diversity caps to prevent any single trader from dominating

### Key Features
- Dual Pool System: Separate scoring for registered miners vs. general pool traders
- Volume Decay: Recent activity weighted more heavily than older trades
- Smooth Transitions: Ramp constraints prevent sudden allocation changes
- Budget Management: Ensures total payouts never exceed available budget
- Performance Gating: Only profitable, active traders receive rewards

The system is designed to incentivize high-quality trading signals while maintaining fairness 
and preventing gaming through volume requirements and historical performance tracking.

## Miner and Validator Functionality

### Miner

- Miners generate **information signals** by trading on Almanac, which routes Polymarket CLOB orders through the miner’s proxy wallet.  
- Every trade becomes a scored prediction within the **incentive mechanism**, which evaluates accuracy, ROI, timing, and informational value.  
- Miners may use **manual strategies, models, or automated systems**—the scoring is model-agnostic and purely performance-based.  
- High-signal miners earn the largest share of **daily Alpha Token emissions**.

### Validator

- **Metadata Syncing**: Validators continuously sync miner metadata from chain (wallets, proxy addresses, UIDs).  
- **Data Ingestion**: At each epoch, the validator pulls miner trading history from Almanac’s backend (rolling window).  
- **Scoring**: The validator runs the **two-phase scoring mechanism**.
- **Weight Setting**: After generating scores, validators set miner weights on-chain, determining Alpha Token emissions for the next epoch.

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
1. Go to **https://beta.almanac.market**  
2. Create an account
    - Deploy safe wallet
    - Sign all approvals
    - Fund your safe wallet
3. Connect your Bittensor coldkey:  
   - Install the Bittensor wallet extension  
   - Import the coldkey tied to your miner UID  
   - Link wallet in Almanac settings

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

Once linked, miners can trade directly on **https://beta.almanac.market**.  
Validators automatically detect trades, compute scores, and distribute emissions.

*Note: Miners trading directly on the app don't need to connect their Bittensor wallet after initially linking account.

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
    --logging.debug
```


### Run Basic Validator with PM2

```bash 
pm2 start validator.py --name sn41-validator -- \
    --netuid 41 \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
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
