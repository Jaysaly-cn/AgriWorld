"""
AgriWorld Config — v1.0 稳定基线备份
=======================================
未加入消融实验框架、LSTM 残差模块、MODEL_VERSION 之前的版本。
保留所有已验证的架构改进 (D0 独立 lr, Phenology 独立 lr, step_size=0.5)。
"""

import torch

SEED = 42

DATA_PATH = "./AgriWorld_Master/national_ode_tensors_v2.pkl"
PRETRAINED_DIR = "./pretrained"
SAVE_DIR = "./saved_models"

VAL_RATIO = 0.2
BATCH_TRAIN = 32
BATCH_VAL = 64

LR_EXPERT = 2e-4
LR_PHOTO = 2e-3
LR_COUPLING = 3e-3
LR_STOMATAL = 1e-2        # D0 单独高学习率 (梯度路径长)
LR_PHENOLOGY = 5e-3       # GDD 节点单独学习率
LR_YIELD = 1e-2
WEIGHT_DECAY = 1e-5

W_LAI = 3.0
W_YIELD = 7.0
W_L1 = 0.5
W_SMOOTH = 0.1

PHASE_LAI_ONLY = 12
PHASE_ANOM_RAMP = 20
MAX_EPOCHS = 300
PATIENCE = 60
GRAD_CLIP = 1.0

T_BASE = 10.0
T_OPT = 28.0
T_CEIL = 45.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
