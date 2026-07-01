"""Small USDA NASS QuickStats helpers for crop-specific county yields."""

import os
import re

import requests


API_KEY = os.getenv("USDA_NASS_API_KEY", "4A195BB4-78C3-36BB-A640-4D3378D21432")

CROP_YIELD_SPECS = {
    1: {
        "name": "Corn",
        "short_desc": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
        "bushel_lb": 56.0,
    },
    5: {
        "name": "Soybean",
        "short_desc": "SOYBEANS - YIELD, MEASURED IN BU / ACRE",
        "bushel_lb": 60.0,
    },
}


def normalize_county(county):
    text = str(county or "").upper().replace("&", " AND ")
    text = re.sub(r"\bCOUNTY\b", "", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return " ".join(text.split())


def make_yield_key(state, county):
    return f"{str(state).upper()}-{normalize_county(county)}"


def parse_crop_codes(crops):
    if crops is None:
        return sorted(CROP_YIELD_SPECS)
    if isinstance(crops, str):
        crops = [item.strip() for item in crops.split(",") if item.strip()]
    return [int(c) for c in crops]


def fetch_nass_yield(year, states, crop_code, api_key=API_KEY, timeout=30):
    spec = CROP_YIELD_SPECS[int(crop_code)]
    base_url = "http://quickstats.nass.usda.gov/api/api_GET/"
    out = {}
    print(f"[NASS] {year} {spec['name']}: {spec['short_desc']}")
    for state in states:
        params = {
            "key": api_key,
            "short_desc": spec["short_desc"],
            "agg_level_desc": "COUNTY",
            "state_alpha": state,
            "year": year,
            "format": "JSON",
        }
        try:
            resp = requests.get(base_url, params=params, verify=False, timeout=timeout)
            if resp.status_code != 200:
                print(f"  {state}: HTTP {resp.status_code}")
                continue
            for item in resp.json().get("data", []):
                county = item.get("county_name", "")
                try:
                    value = float(str(item.get("Value", "0")).replace(",", ""))
                except ValueError:
                    continue
                out[make_yield_key(state, county)] = value
        except Exception as exc:
            print(f"  {state}: {exc}")
    print(f"  Retrieved {len(out)} county yields")
    return out


def fetch_multi_crop_yields(years, states, crop_codes=None):
    crop_codes = parse_crop_codes(crop_codes)
    return {
        (int(year), int(crop)): fetch_nass_yield(year, states, crop)
        for year in years
        for crop in crop_codes
    }

