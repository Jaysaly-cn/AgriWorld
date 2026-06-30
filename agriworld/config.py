import os
import torch
from agriworld.paths import MERGED_DATA_PATH, PRETRAINED_DIR, SAVE_DIR

SEED = 42
MODEL_SCHEMA = "agriworld-v3.27"

# 鈹€鈹€ 妯″瀷鐗堟湰 (娑堣瀺瀹為獙) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 'baseline': 鍘熷 CouplingHead (MLP anomaly)
# 'lstm_res': +LSTM 鏃跺簭娈嬪樊妯″潡
MODEL_VERSION = os.getenv("AGRI_MODEL_VERSION", "phys_spatial")

# 鈹€鈹€ LSTM 娈嬪樊妯″潡 (浠?lstm_res 鐗堟湰鐢熸晥) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
_default_lstm = "1" if MODEL_VERSION == "lstm_res" else "0"
USE_LSTM_RESIDUAL = bool(int(os.getenv("AGRI_USE_LSTM", _default_lstm)))
LR_LSTM = 1e-4
USE_YEAR_EMBEDDING = bool(int(os.getenv("AGRI_USE_YEAR_EMBEDDING", "0")))
USE_YIELD_YEAR_TREND = bool(int(os.getenv("AGRI_USE_YIELD_YEAR_TREND", "1")))
USE_COUPLING_ANOMALY = bool(int(os.getenv("AGRI_USE_COUPLING_ANOMALY", "1")))
USE_VPD_STRESS = bool(int(os.getenv("AGRI_USE_VPD_STRESS", "1")))
USE_NITROGEN_STRESS = bool(int(os.getenv("AGRI_USE_NITROGEN_STRESS", "1")))
USE_TEMPERATURE_STRESS = bool(int(os.getenv("AGRI_USE_TEMPERATURE_STRESS", "1")))
USE_YIELD_RESIDUAL = bool(int(os.getenv("AGRI_USE_YIELD_RESIDUAL", "0")))
USE_STATIC_INTERACTION_GATES = bool(int(os.getenv("AGRI_USE_STATIC_INTERACTION_GATES", "1")))
USE_STATIC_CROP_PARAMS = bool(int(os.getenv("AGRI_USE_STATIC_CROP_PARAMS", "1")))
USE_REPRODUCTIVE_HEAT_PENALTY = bool(int(os.getenv("AGRI_USE_REPRODUCTIVE_HEAT_PENALTY", "1")))
YIELD_RESIDUAL_MAX_LOG = float(os.getenv("AGRI_YIELD_RESIDUAL_MAX_LOG", "0.20"))
STATIC_INTERACTION_MAX = float(os.getenv("AGRI_STATIC_INTERACTION_MAX", "0.10"))
STATIC_HI_MAX_LOG = float(os.getenv("AGRI_STATIC_HI_MAX_LOG", "0.10"))
STATIC_YIELD_MAX_LOG = float(os.getenv("AGRI_STATIC_YIELD_MAX_LOG", "0.08"))
STATIC_HEAT_SENS_MAX = float(os.getenv("AGRI_STATIC_HEAT_SENS_MAX", "0.50"))
W_STATIC_ADAPT = float(os.getenv("AGRI_W_STATIC_ADAPT", "0.15"))
FACTOR_RESPONSE_EPS = float(os.getenv("AGRI_FACTOR_RESPONSE_EPS", "0.5"))
HEAT_AUDIT_HOT_DAY_C = float(os.getenv("AGRI_HEAT_AUDIT_HOT_DAY_C", "28.0"))
HEAT_AUDIT_DELTA_C = float(os.getenv("AGRI_HEAT_AUDIT_DELTA_C", "6.0"))
SAVE_TRAIN_HISTORY = bool(int(os.getenv("AGRI_SAVE_TRAIN_HISTORY", "1")))
SAVE_EVAL_TABLES = bool(int(os.getenv("AGRI_SAVE_EVAL_TABLES", "1")))
SAVE_EVAL_TRAJECTORIES = bool(int(os.getenv("AGRI_SAVE_EVAL_TRAJECTORIES", "1")))
EVAL_TRAJECTORY_SAMPLES = int(os.getenv("AGRI_EVAL_TRAJECTORY_SAMPLES", "24"))
# The raw Wang-Engel temperature response is too strong as a direct
# multiplicative stress term on this dataset. Default to an extreme-heat
# penalty: ordinary temperature still drives phenology through GDD, while only
# sustained hot days reduce growth.
TEMPERATURE_STRESS_MODE = os.getenv("AGRI_TEMPERATURE_STRESS_MODE", "heat")
TEMPERATURE_STRESS_FLOOR = float(os.getenv("AGRI_TEMPERATURE_STRESS_FLOOR", "0.95"))
TEMPERATURE_STRESS_STRENGTH = float(os.getenv("AGRI_TEMPERATURE_STRESS_STRENGTH", "0.20"))
HEAT_STRESS_THRESHOLD_C = float(os.getenv("AGRI_HEAT_STRESS_THRESHOLD_C", "33.0"))
HEAT_STRESS_WIDTH_C = float(os.getenv("AGRI_HEAT_STRESS_WIDTH_C", "2.5"))
HEAT_STRESS_MAX_REDUCTION = float(os.getenv("AGRI_HEAT_STRESS_MAX_REDUCTION", "0.16"))
HEAT_STRESS_STAGE_CENTER = float(os.getenv("AGRI_HEAT_STRESS_STAGE_CENTER", "0.45"))
HEAT_STRESS_STAGE_WIDTH = float(os.getenv("AGRI_HEAT_STRESS_STAGE_WIDTH", "0.15"))
REPRO_HEAT_THRESHOLD_C = float(os.getenv("AGRI_REPRO_HEAT_THRESHOLD_C", "30.0"))
REPRO_HEAT_WIDTH_C = float(os.getenv("AGRI_REPRO_HEAT_WIDTH_C", "2.0"))
REPRO_HEAT_MAX_HI_REDUCTION = float(os.getenv("AGRI_REPRO_HEAT_MAX_HI_REDUCTION", "0.14"))

DATA_PATH = os.getenv("AGRI_DATA_PATH", MERGED_DATA_PATH)

VAL_RATIO = 0.2
SPLIT_MODE = "auto"  # temporal when balanced; otherwise county holdout
BATCH_TRAIN = int(os.getenv("AGRI_BATCH_TRAIN", "256"))
BATCH_VAL = int(os.getenv("AGRI_BATCH_VAL", "512"))
DATA_LOADER_WORKERS = 0  # Dataset is already fully materialized in RAM
PIN_MEMORY = True

# Fast training uses one explicit state update per day instead of four RK4
# evaluations. Set AGRI_ODE_METHOD=rk4 for high-accuracy comparison runs.
ODE_METHOD = os.getenv("AGRI_ODE_METHOD", "euler")
ODE_STEP_SIZE = 1.0
# Training speed knob. The solver evaluates the ODE every N days and
# linearly fills intermediate daily states for LAI/yield losses.
TRAIN_STEP_DAYS = int(os.getenv("AGRI_TRAIN_STEP_DAYS", "3"))
ALLOW_TF32 = True
VAL_EVERY = 5

LR_EXPERT = 3e-4
LR_PHOTO = 3e-3
LR_COUPLING = 5e-4
LR_STOMATAL = 5e-4
LR_PHENOLOGY = 5e-4
LR_YIELD = 1e-2
WEIGHT_DECAY = 1e-5
FREEZE_SOIL_HYDRAULICS = True

W_LAI = 1.2
W_YIELD = 4.0
W_L1 = 0.3
W_SMOOTH = 0.1
W_STATE = 0.5
W_PRIOR = 0.03
W_CANOPY = 3.0

PHASE_LAI_ONLY = 20
PHASE_ANOM_RAMP = 30
MAX_EPOCHS = 250
PATIENCE = 12  # validation checks; with VAL_EVERY=5 this is about 60 epochs
GRAD_CLIP = 1.0

# Physical-state reference ranges used by regularization and diagnostics.
MAX_BIOMASS_T_HA = 35.0
MAX_MINERAL_N_KG_HA = 400.0

INITIAL_LAI = 0.05
INITIAL_BIOMASS_T_HA = 0.05
ESTABLISHMENT_BIOMASS_T_HA = 0.30
ESTABLISHMENT_LAI = 0.45
INITIAL_HARVEST_INDEX = 0.56
INITIAL_YIELD_SCALE = 1.15
INITIAL_YIELD_YEAR_TREND_LOG = 0.035
YIELD_TREND_CENTER_YEAR = 2021.0
SOIL_WATER_STRESS_FLOOR = 0.60

T_BASE = 10.0
T_OPT = 28.0
T_CEIL = 45.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

