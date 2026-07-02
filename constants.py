# Constants for trade history
ROLLING_HISTORY_IN_DAYS = 30

# Constants for the scoring function
VOLUME_FEE = 0.01

# Price adjustment buffer for order placement
PRICE_BUFFER_ADJUSTMENT = 0.01

# Public Polymarket builder code (bytes32) used for order attribution.
POLY_BUILDER_CODE = "0x196258757463baebc045d1adc1c9c0a55cad7ac5d09ab7b7e1eb31803d9bfbe0"

ROI_MIN = 0.0
VOLUME_MIN = 1
VOLUME_DECAY = 0.85
RAMP = 0.1 # originally 0.1
RHO_CAP = 0.05 # originally 0.1
KAPPA_NEXT = 0.03 # originally 0.02
KAPPA_SCALING_FACTOR = 8 # originally 6; raised to 8 to fund more active high-volume contributors past the Phase 1 ROI cliff
# Minimum allocation gate (x) for eligible traders in both optimizer phases.
DUST_GATE = 0.01

# Protocol Contributor scoring: route profitable epoch flow, cap diversity on epoch volume,
# and weight Phase 2 redistribution by historical volume credibility.
ENABLE_PROTOCOL_CONTRIBUTOR = True
# Diversity cap volume basis when protocol contributor is on: "block" (epoch) or "eff" (legacy).
RHO_VOLUME_BASIS = "block"
# Phase 2: multiply ROI weights by log(1 + v_memory) to favor long-term fee contributors.
ENABLE_P2_CREDIBILITY_WEIGHT = True
# Phase 2: apply credibility only when roi > kappa (don't amplify below-kappa penalties).
ENABLE_P2_CRED_ON_POSITIVE_DELTA_ONLY = True
# Phase 2: epoch-active miners retain at least this fraction of their Phase 1 gate (0=off).
PHASE2_ACTIVE_GATE_RETENTION = 0.4
# Only apply retention when Phase 1 opened a meaningful gate (above dust).
PHASE2_ACTIVE_GATE_MIN_X1 = 0.02
# Phase 1 budget volume blend when protocol contributor is on: 0=v_eff (legacy), 1=v_block.
PHASE1_BUDGET_VOLUME_ALPHA = 1.0

# Build-up period constants for miner eligibility
MIN_EPOCHS_FOR_ELIGIBILITY = 3  # Must trade for X epochs
MIN_PREDICTIONS_FOR_ELIGIBILITY = 5  # Must have X predictions

# Weighting parameters
# If ENABLE_STATIC_WEIGHTING is True, we will use the static weighting parameters below.
ENABLE_STATIC_WEIGHTING = False
GENERAL_POOL_WEIGHT_PERCENTAGE = 0.5
MINER_WEIGHT_PERCENTAGE = 1 - GENERAL_POOL_WEIGHT_PERCENTAGE
# Max percent of the total possible epoch budget that can be allocated.
# This is used to give more weights (and in turn, more incentives) to the miners when we aren't using the full budget.
MAX_EPOCH_BUDGET_PERCENTAGE_FOR_BOOST = .25

# If ENABLE_STATIC_WEIGHTING is False, we will use the dynamic weighting.
# This is used to give more weights (and in turn, more incentives) to the miners when we aren't using the full budget by increasing the total miner pool budget.
# Set to 0 to disable.
MINER_POOL_BUDGET_BOOST_PERCENTAGE = 0

# Early stage incentive parameters for the miner pool
ENABLE_ES_MINER_INCENTIVES = False
# This is the multiplier of the trader's fees paid that is given to the miner if they have positive tokens (score). 1.2 == 120%.
ESM_MIN_MULTIPLIER = 1.2
# If ENABLE_ES_MINER_LOSS_COMPENSATION is True, we will give back the trader's fees paid to the miner if they have positive epoch profit but no score.
ENABLE_ES_MINER_LOSS_COMPENSATION = True
# This is the percentage of the trader's fees paid that is given back to the miner if they have positive epoch profit but no score. 1.0 == 100%.
ESM_LOSS_COMPENSATION_PERCENTAGE = 1.0

# Early stage incentive parameters for the general pool
ENABLE_ES_GP_INCENTIVES = False
# This is the multiplier of the trader's fees paid that is given to the gp trader if they have positive tokens (score). 1.2 == 120%.
ESGP_MIN_MULTIPLIER = 1.2
# If ENABLE_ES_GP_LOSS_COMPENSATION is True, we will give back the trader's fees paid to the gp trader if they have positive epoch profit but no score.
ENABLE_ES_GP_LOSS_COMPENSATION = False
# This is the percentage of the trader's fees paid that is given back to the gp trader if they have positive epoch profit but no score. 1.0 == 100%.
ESGP_LOSS_COMPENSATION_PERCENTAGE = 1.0

# This is used to give more weights (and in turn, more incentives) to the miners by taking the final miner pool weights and boosting them by this percentage.
# Set to 0 to disable.
MINER_POOL_WEIGHT_BOOST_PERCENTAGE = 0.5

TOTAL_MINER_ALPHA_PER_DAY = 2952 # 7200 alpha per day for entire subnet * 0.41 (41% for miners)

# Subnet owner burn UID
BURN_UID = 210
# Subnet owner excess miner weight UID
EXCESS_MINER_WEIGHT_UID = None
EXCESS_MINER_MIN_WEIGHT = 0 # 0.00001 should be low enough if used
EXCESS_MINER_TAKE_PERCENTAGE = 0 # percentage of the excess miner weight that is set to EXCESS_MINER_WEIGHT_UID. rest goes to BURN_UID.