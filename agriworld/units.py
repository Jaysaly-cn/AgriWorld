"""Shared physical-unit conversions used by training and evaluation."""

BUSHEL_LB_BY_CROP = {
    1: 56.0,  # corn
    5: 60.0,  # soybean
}


def bu_ac_to_t_ha_factor(crop_code=1):
    bushel_lb = BUSHEL_LB_BY_CROP.get(int(crop_code), 56.0)
    return bushel_lb * 0.45359237 / 1000.0 / 0.404685642


def t_ha_to_bu_ac_factor(crop_code=1):
    return 1.0 / bu_ac_to_t_ha_factor(crop_code)


# Backward-compatible corn constants.
CORN_BU_AC_TO_T_HA = bu_ac_to_t_ha_factor(1)
CORN_T_HA_TO_BU_AC = t_ha_to_bu_ac_factor(1)

# RUE × PAR produces g dry matter m-2 day-1.
G_M2_TO_T_HA = 0.01

# Effective soil layer used to estimate the initial mineral-N reserve.
TOPSOIL_DEPTH_M = 0.30
BACKGROUND_MINERAL_N_FRACTION = 0.005
FERTILIZER_AVAILABILITY = 0.70
