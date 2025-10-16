# Constants for trade history
ROLLING_WINDOW_IN_DAYS = 60

# Constants for the scoring function
VOLUME_FEE = 0.01
ROI_MIN = 0.005 # originally 0.02
VOLUME_MIN = 100
RAMP = 0.3 # originally 0.1
RHO_CAP = 0.5 # originally 0.1

# Weighting parameters
# Percentage we allocate to score well-performing miners so they don't get deregistered
ROLLING_WEIGHT_PERCENTAGE = 0.01
GENERAL_POOL_WEIGHT_PERCENTAGE = 0.20
MINER_WEIGHT_PERCENTAGE = 1 - GENERAL_POOL_WEIGHT_PERCENTAGE - ROLLING_WEIGHT_PERCENTAGE

# Subnet owner burn UID
BURN_UID = 210