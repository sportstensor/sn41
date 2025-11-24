# Constants for trade history
ROLLING_HISTORY_IN_DAYS = 30

# Constants for the scoring function
VOLUME_FEE = 0.01
ROI_MIN = 0.0
VOLUME_MIN = 10
VOLUME_DECAY = 0.9
RAMP = 0.1 # originally 0.1
RHO_CAP = 0.1 # originally 0.1
KAPPA_NEXT = 0.03 # originally 0.02
KAPPA_SCALING_FACTOR = 1 # originally 6

# Build-up period constants for miner eligibility
MIN_EPOCHS_FOR_ELIGIBILITY = 2  # Must trade for X epochs
MIN_PREDICTIONS_FOR_ELIGIBILITY = 2  # Must have X predictions

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

# This is used to give more weights (and in turn, more incentives) to the miners by taking the final miner pool weights and boosting them by this percentage.
# Set to 0 to disable.
MINER_POOL_WEIGHT_BOOST_PERCENTAGE = 0.5

TOTAL_MINER_ALPHA_PER_DAY = 2952 # 7200 alpha per day for entire subnet * 0.41 (41% for miners)

# Subnet owner burn UID
BURN_UID = 210
# Subnet owner excess miner weight UID
EXCESS_MINER_WEIGHT_UID = 0
EXCESS_MINER_MIN_WEIGHT = 0.0001