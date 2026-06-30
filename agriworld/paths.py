"""Central server paths and network settings for AgriWorld."""

import os


PROJECT_ROOT = os.path.abspath(os.getenv(
    "AGRI_PROJECT_ROOT",
    "/data4/Agri/yukaijie/AgriWorld/AgriWorld/Newest_version",
))
DATA_ROOT = os.path.abspath(os.getenv(
    "AGRI_DATA_ROOT",
    "/data4/Agri/yukaijie/AgriWorld/AgriWorld/AgriWorld_Master",
))
CACHE_ROOT = os.path.abspath(os.getenv(
    "AGRI_CACHE_ROOT",
    os.path.join(DATA_ROOT, "cache_v"),
))

MERGED_DATA_PATH = os.path.join(
    DATA_ROOT, "national_ode_tensors_v2_merged.pkl"
)
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "pretrained")
SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
BENCHMARK_RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "results")

PROXY_HOST = os.getenv("AGRI_PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.getenv("AGRI_PROXY_PORT", "17897"))
PROXY_URL = os.getenv(
    "AGRI_PROXY_URL",
    f"http://{PROXY_HOST}:{PROXY_PORT}",
)

# Keep the existing credential location configurable because it is outside
# the project and data roots supplied for this deployment.
GEE_CREDENTIALS_PATH = os.path.abspath(os.getenv(
    "AGRI_GEE_CREDENTIALS",
    "/data4/Agri/yukaijie/AgriWorld/GEE_data/"
    "agri-world-model-cd58d4277876.json",
))
