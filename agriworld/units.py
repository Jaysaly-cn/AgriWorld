"""Shared physical-unit conversions used by training and evaluation."""

# USDA corn yield is reported in 56 lb bushels per acre.
CORN_BU_AC_TO_T_HA = 56.0 * 0.45359237 / 1000.0 / 0.404685642
CORN_T_HA_TO_BU_AC = 1.0 / CORN_BU_AC_TO_T_HA

# RUE × PAR produces g dry matter m-2 day-1.
G_M2_TO_T_HA = 0.01

# Effective soil layer used to estimate the initial mineral-N reserve.
TOPSOIL_DEPTH_M = 0.30
BACKGROUND_MINERAL_N_FRACTION = 0.005
FERTILIZER_AVAILABILITY = 0.70

