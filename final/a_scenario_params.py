#==============================================================================
# SCENARIO PARAMETERS
#==============================================================================
"""
This module contains all scenario parameters used across the simulation.
Centralizing parameters here makes them easier to maintain and update.
"""

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
                            180: 0},
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

