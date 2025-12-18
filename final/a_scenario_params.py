#==============================================================================
# SCENARIO PARAMETERS
#==============================================================================
"""
This module contains all scenario parameters used across the simulation.
Centralizing parameters here makes them easier to maintain and update.

Key functions:
- get_scenario_parameters(): Returns raw parameter dictionaries
- get_interpolated_param(): Interpolates time-varying parameters for sub-timesteps
- generate_timesteps(): Generates sub-timesteps with optional intervals
"""


# =============================================================================
# INTERPOLATION UTILITIES
# =============================================================================

def get_interpolated_param(param_dict, year):
    """
    Get parameter value at a given year, with linear interpolation for sub-timesteps.
    
    For parameters that vary by year (stored as dicts like {0: val0, 60: val60, 180: val180}),
    this function linearly interpolates values for years between defined timesteps.
    
    Parameters:
        param_dict: Dict mapping years to values (e.g., {0: 0, 60: 100, 180: 200})
        year: The year to get the value for
    
    Returns:
        Interpolated value at the given year
    
    Example:
        >>> param = {0: 0, 60: 100, 180: 200}
        >>> get_interpolated_param(param, 90)  # 90 is 25% through 60->180
        125.0  # 100 + 0.25 * (200 - 100)
    """
    if not isinstance(param_dict, dict):
        return param_dict  # Return as-is if not a year-keyed dict
    
    years = sorted(param_dict.keys())
    
    # Exact match
    if year in param_dict:
        return param_dict[year]
    
    # Before first defined year - use first value
    if year < years[0]:
        return param_dict[years[0]]
    
    # After last defined year - use last value
    if year > years[-1]:
        return param_dict[years[-1]]
    
    # Find bracketing years and interpolate
    for i in range(len(years) - 1):
        lower_year = years[i]
        upper_year = years[i + 1]
        
        if lower_year <= year < upper_year:
            lower_val = param_dict[lower_year]
            upper_val = param_dict[upper_year]
            
            # Linear interpolation
            fraction = (year - lower_year) / (upper_year - lower_year)
            interpolated = lower_val + fraction * (upper_val - lower_val)
            return interpolated
    
    # Fallback (shouldn't reach here)
    return param_dict[years[-1]]


def generate_timesteps(base_timesteps=None, interval=None):
    """
    Generate list of timesteps, optionally including sub-timesteps at regular intervals.
    
    Parameters:
        base_timesteps: List of main timesteps (default: [0, 10, 30, 60, 180])
        interval: Interval for sub-timesteps between 60 and 180 (None = no sub-timesteps)
    
    Returns:
        Sorted list of all timesteps
    
    Example:
        >>> generate_timesteps(interval=30)
        [0, 10, 30, 60, 90, 120, 150, 180]
    """
    if base_timesteps is None:
        base_timesteps = [0, 10, 30, 60, 180]
    
    if interval is None or interval <= 0:
        return sorted(base_timesteps)
    
    timesteps = set(base_timesteps)
    
    # Add sub-timesteps between 60 and 180
    start = 60
    end = 180
    current = start + interval
    while current < end:
        timesteps.add(current)
        current += interval
    
    return sorted(timesteps)


def get_params_for_year(site, scenario, year):
    """
    Get scenario parameters for a specific year, with interpolation for time-varying params.
    
    Parameters:
        site: Site name ('trimmed-parade', 'city', 'uni')
        scenario: Scenario type ('positive', 'trending')
        year: Year to get parameters for
    
    Returns:
        Dict of parameters with time-varying values interpolated for the given year
    """
    params_dict = get_scenario_parameters()
    base_params = params_dict[(site, scenario)].copy()
    
    # Interpolate time-varying parameters
    interpolated_params = {}
    for key, value in base_params.items():
        if isinstance(value, dict) and all(isinstance(k, (int, float)) for k in value.keys()):
            # This is a year-keyed parameter - store interpolated value
            interpolated_params[key] = get_interpolated_param(value, year)
        else:
            interpolated_params[key] = value
    
    # Add the year as years_passed
    interpolated_params['years_passed'] = year
    
    return interpolated_params


# =============================================================================
# SCENARIO PARAMETER DEFINITIONS
# =============================================================================

def get_scenario_parameters():
    """
    Returns a dictionary of parameters for each site and scenario.
    
    Returns:
    dict: Dictionary with (site, scenario) tuples as keys and parameter dictionaries as values
    """
    paramsPARADE_positive = {
    'growth_factor_range' : [0.37, 0.51], # Growth factor is a range
    'plantingDensity' : 50, # 10 per hectare
    'ageInPlaceThreshold' : 70,
    'plantThreshold' : 50,
    'rewildThreshold' : 10,
    'senescingThreshold' : -25,
    'snagThreshold' : -200,
    'collapsedThreshold' : -250,
    'controlSteps' : 20,
    'sim_TurnsThreshold' : {0: 0,
                            10: 0,
                            30: 3000,
                            60: 4000,
                            180: 4500},
    }

    paramsPARADE_trending = {
    'growth_factor_range' : [0.37, 0.51], # Growth factor is a range
    'plantingDensity' : 50, # 10 per hectare
    'ageInPlaceThreshold' : 2, # Highest threshold
    'plantThreshold' : 1, # Middle threshold
    'rewildThreshold' : 0, # Lowest threshold
    'senescingThreshold' : -5, 
    'snagThreshold' : -200,
    'collapsedThreshold' : -250,
    'controlSteps' : 20,
    'sim_TurnsThreshold' : {0: 0,
                            10: 0,
                            30: 0,
                            60: 0,
                            180: 0},
    }

    paramsCITY_positive = {
    'growth_factor_range' : [0.37, 0.51], # Growth factor is a range
    'plantingDensity' : 50, # 10 per hectare
    'ageInPlaceThreshold' : 70,
    'plantThreshold' : 50,
    'rewildThreshold' : 10,
    'senescingThreshold' : -25,
    'snagThreshold' : -200,
    'collapsedThreshold' : -250,
    'controlSteps' : 20,
    'sim_TurnsThreshold' : {0: 0,
                            10: 300,
                            30: 1249.75,
                            60: 4999,
                            180: 5000},
    'sim_averageResistance' : {0: 0,
                            10: 50,
                            30: 50,
                            60: 67.90487670898438,
                            180: 96},
    }

    paramsCITY_trending = {
    'growth_factor_range' : [0.37, 0.51], # Growth factor is a range
    'plantingDensity' : 50, # 10 per hectare
    'ageInPlaceThreshold' : 15,
    'plantThreshold' : 10,
    'rewildThreshold' : 5,
    'senescingThreshold' : -5,
    'snagThreshold' : -200,
    'collapsedThreshold' : -250,
    'controlSteps' : 20,
    'sim_TurnsThreshold' : {0: 0,
                            10: 20,
                            30: 50,
                            60: 100,
                            180: 200},
    'sim_averageResistance' : {0: 0,
                            10: 10,
                            30: 20,
                            60: 30,
                            180: 50},
    }

    paramsUNI_positive = {
    'growth_factor_range' : [0.37, 0.51], # Growth factor is a range
    'plantingDensity' : 50, # 10 per hectare
    'ageInPlaceThreshold' : 70,
    'plantThreshold' : 50,
    'rewildThreshold' : 10,
    'senescingThreshold' : -25,
    'snagThreshold' : -200,
    'collapsedThreshold' : -250,
    'controlSteps' : 20,
    'sim_TurnsThreshold' : {0: 0,
                            10: 300,
                            30: 1249.75,
                            60: 4999,
                            180: 5000},
    'sim_averageResistance' : {0: 0,
                            10: 50,
                            30: 50,
                            60: 67.90487670898438,
                            180: 80},
    }

    paramsUNI_trending = {
    'growth_factor_range' : [0.37, 0.51], # Growth factor is a range
    'plantingDensity' : 50, # 10 per hectare
    'ageInPlaceThreshold' : 15,
    'plantThreshold' : 10,
    'rewildThreshold' : 5,
    'senescingThreshold' : -5,
    'snagThreshold' : -200,
    'collapsedThreshold' : -250,
    'controlSteps' : 20,
    'sim_TurnsThreshold' : {0: 0,
                            10: 20,
                            30: 50,
                            60: 100,
                            180: 200},
    'sim_averageResistance' : {0: 0,
                            10: 0,
                            30: 0,
                            60: 0,
                            180: 0},
    }

    paramsDic = {
        ('trimmed-parade', 'positive'): paramsPARADE_positive,
        ('trimmed-parade', 'trending'): paramsPARADE_trending,
        ('city', 'positive'): paramsCITY_positive,
        ('city', 'trending'): paramsCITY_trending,
        ('uni', 'positive'): paramsUNI_positive,
        ('uni', 'trending'): paramsUNI_trending,
    }
    
    return paramsDic 

