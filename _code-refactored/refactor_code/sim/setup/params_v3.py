"""
Scenario parameters for the current v3 engine.

This module is the live parameter source for the refactored v3 stack.
It mirrors the existing site/scenario parameter definitions while keeping
the v3 engine independent from `final/a_scenario_params.py`.
"""

from __future__ import annotations


def triangular_duration(minimum: int, mode: int, maximum: int) -> dict[str, float]:
    """Encode a bounded triangular duration from explicit min/mode/max."""
    return {
        "min": float(minimum),
        "mode": float(mode),
        "max": float(maximum),
    }


def get_interpolated_param(param_dict, year):
    """Return a parameter value at a given year with linear interpolation."""
    if not isinstance(param_dict, dict):
        return param_dict

    years = sorted(param_dict.keys())

    if year in param_dict:
        return param_dict[year]
    if year < years[0]:
        return param_dict[years[0]]
    if year > years[-1]:
        return param_dict[years[-1]]

    for i in range(len(years) - 1):
        lower_year = years[i]
        upper_year = years[i + 1]
        if lower_year <= year < upper_year:
            lower_val = param_dict[lower_year]
            upper_val = param_dict[upper_year]
            fraction = (year - lower_year) / (upper_year - lower_year)
            return lower_val + fraction * (upper_val - lower_val)

    return param_dict[years[-1]]


def generate_timesteps(base_timesteps=None, interval=None):
    """Generate assessed years, optionally adding regular sub-timesteps."""
    if base_timesteps is None:
        base_timesteps = [0, 1, 10, 30, 60, 180]

    if interval is None or interval <= 0:
        return sorted(base_timesteps)

    timesteps = set(base_timesteps)
    start = 60
    end = 180
    current = start + interval
    while current < end:
        timesteps.add(current)
        current += interval

    return sorted(timesteps)


def get_params_for_year(site, scenario, year):
    """Return interpolated parameters for a site/scenario/year."""
    params_dict = get_scenario_parameters()
    base_params = params_dict[(site, scenario)].copy()

    interpolated_params = {}
    for key, value in base_params.items():
        if isinstance(value, dict) and all(isinstance(k, (int, float)) for k in value.keys()):
            interpolated_params[key] = get_interpolated_param(value, year)
        else:
            interpolated_params[key] = value

    interpolated_params["years_passed"] = year
    return interpolated_params


def get_scenario_parameters():
    """Return raw parameter dictionaries keyed by (site, scenario)."""
    senescing_duration_years = triangular_duration(10, 90, 200)
    snag_duration_years = triangular_duration(0, 40, 100)
    fallen_duration_years = triangular_duration(10, 40, 100)
    decayed_duration_years = triangular_duration(30, 40, 75)

    params_parade_positive = {
        "growth_factor_range": [0.37, 0.51],
        "growth_model": "split",
        "growth_model_precolonial": "fischer",
        "growth_model_colonial": "ulmus",
        "mortality_model": "flat",
        "plantingDensity": 50,
        "annual_tree_death_urban": 0.06,
        "annual_tree_death_nature-reserves": 0.03,
        "minimal-tree-support-threshold": 70,
        "moderate-tree-support-threshold": 50,
        "maximum-tree-support-threshold": 10,
        "lifecycle_senescing_ramp_start": -25,
        "senescing_duration_years": senescing_duration_years.copy(),
        "snag_duration_years": snag_duration_years.copy(),
        "fallen_duration_years": fallen_duration_years.copy(),
        "decayed_duration_years": decayed_duration_years.copy(),
        "controlSteps": 20,
        "sim_TurnsThreshold": {
            0: 0,
            10: 0,
            30: 3000,
            60: 4500,
            180: 5000,
        },
        "ground_filter_mode": "node-exclusion",  # "proximity" or "node-exclusion"
    }

    params_parade_trending = {
        "growth_factor_range": [0.37, 0.51],
        "growth_model": "split",
        "growth_model_precolonial": "fischer",
        "growth_model_colonial": "ulmus",
        "mortality_model": "flat",
        "plantingDensity": 50,
        "annual_tree_death_urban": 0.06,
        "annual_tree_death_nature-reserves": 0.03,
        "minimal-tree-support-threshold": 2,
        "moderate-tree-support-threshold": 1,
        "maximum-tree-support-threshold": 0,
        "lifecycle_senescing_ramp_start": -5,
        "senescing_duration_years": senescing_duration_years.copy(),
        "snag_duration_years": snag_duration_years.copy(),
        "fallen_duration_years": fallen_duration_years.copy(),
        "decayed_duration_years": decayed_duration_years.copy(),
        "controlSteps": 20,
        "sim_TurnsThreshold": {
            0: 0,
            10: 0,
            30: 0,
            60: 0,
            180: 0,
        },
        "ground_filter_mode": "node-exclusion",
    }

    params_city_positive = {
        "growth_factor_range": [0.37, 0.51],
        "growth_model": "split",
        "growth_model_precolonial": "fischer",
        "growth_model_colonial": "ulmus",
        "mortality_model": "flat",
        "plantingDensity": 50,
        "annual_tree_death_urban": 0.06,
        "annual_tree_death_nature-reserves": 0.03,
        "minimal-tree-support-threshold": 70,
        "moderate-tree-support-threshold": 50,
        "maximum-tree-support-threshold": 10,
        "lifecycle_senescing_ramp_start": -25,
        "senescing_duration_years": senescing_duration_years.copy(),
        "snag_duration_years": snag_duration_years.copy(),
        "fallen_duration_years": fallen_duration_years.copy(),
        "decayed_duration_years": decayed_duration_years.copy(),
        "controlSteps": 20,
        "sim_TurnsThreshold": {
            0: 0,
            10: 300,
            30: 1249.75,
            60: 4000,
            180: 5000,
        },
        "sim_averageResistance": {
            0: 0,
            10: 50,
            30: 50,
            60: 67.90487670898438,
            180: 96,
        },
        "ground_filter_mode": "node-exclusion",
    }

    params_city_trending = {
        "growth_factor_range": [0.37, 0.51],
        "growth_model": "split",
        "growth_model_precolonial": "fischer",
        "growth_model_colonial": "ulmus",
        "mortality_model": "flat",
        "plantingDensity": 50,
        "annual_tree_death_urban": 0.06,
        "annual_tree_death_nature-reserves": 0.03,
        "minimal-tree-support-threshold": 2,
        "moderate-tree-support-threshold": 1,
        "maximum-tree-support-threshold": 0,
        "lifecycle_senescing_ramp_start": -5,
        "senescing_duration_years": senescing_duration_years.copy(),
        "snag_duration_years": snag_duration_years.copy(),
        "fallen_duration_years": fallen_duration_years.copy(),
        "decayed_duration_years": decayed_duration_years.copy(),
        "controlSteps": 20,
        "sim_TurnsThreshold": {
            0: 0,
            10: 20,
            30: 50,
            60: 100,
            180: 200,
        },
        "sim_averageResistance": {
            0: 0,
            10: 10,
            30: 20,
            60: 30,
            180: 50,
        },
        "ground_filter_mode": "node-exclusion",
    }

    params_uni_positive = {
        "growth_factor_range": [0.37, 0.51],
        "growth_model": "split",
        "growth_model_precolonial": "fischer",
        "growth_model_colonial": "ulmus",
        "mortality_model": "flat",
        "plantingDensity": 50,
        "annual_tree_death_urban": 0.06,
        "annual_tree_death_nature-reserves": 0.03,
        "minimal-tree-support-threshold": 70,
        "moderate-tree-support-threshold": 50,
        "maximum-tree-support-threshold": 10,
        "lifecycle_senescing_ramp_start": -25,
        "senescing_duration_years": senescing_duration_years.copy(),
        "snag_duration_years": snag_duration_years.copy(),
        "fallen_duration_years": fallen_duration_years.copy(),
        "decayed_duration_years": decayed_duration_years.copy(),
        "controlSteps": 20,
        "sim_TurnsThreshold": {
            0: 0,
            10: 300,
            30: 3000,
            60: 4500,
            180: 5000,
        },
        "sim_averageResistance": {
            0: 0,
            10: 50,
            30: 50,
            60: 67.90487670898438,
            180: 80,
        },
        "ground_filter_mode": "node-exclusion",
        "deployGroundDeadwood": {
            "enabled": True,
            "deadwood_m3_per_ha": 208.3,       # volume target per ha (matches baseline)
            "fallen_share": 0.70,              # 70% fallen, 30% decayed by volume
            "max_fraction_per_pulse": 0.10,    # cap at 10% of baseline per pulse
            "spacing_preference": 3.0,         # metres; soft — fallback places anyway
        },
    }

    params_uni_trending = {
        "growth_factor_range": [0.37, 0.51],
        "growth_model": "split",
        "growth_model_precolonial": "fischer",
        "growth_model_colonial": "ulmus",
        "mortality_model": "flat",
        "plantingDensity": 50,
        "annual_tree_death_urban": 0.06,
        "annual_tree_death_nature-reserves": 0.03,
        "minimal-tree-support-threshold": 2,
        "moderate-tree-support-threshold": 1,
        "maximum-tree-support-threshold": 0,
        "lifecycle_senescing_ramp_start": -5,
        "senescing_duration_years": senescing_duration_years.copy(),
        "snag_duration_years": snag_duration_years.copy(),
        "fallen_duration_years": fallen_duration_years.copy(),
        "decayed_duration_years": decayed_duration_years.copy(),
        "controlSteps": 20,
        "sim_TurnsThreshold": {
            0: 0,
            10: 20,
            30: 50,
            60: 100,
            180: 200,
        },
        "sim_averageResistance": {
            0: 0,
            10: 0,
            30: 0,
            60: 0,
            180: 0,
        },
        "ground_filter_mode": "node-exclusion",
    }

    return {
        ("trimmed-parade", "positive"): params_parade_positive,
        ("trimmed-parade", "trending"): params_parade_trending,
        ("city", "positive"): params_city_positive,
        ("city", "trending"): params_city_trending,
        ("uni", "positive"): params_uni_positive,
        ("uni", "trending"): params_uni_trending,
    }
