import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path
import a_vis_colors
"""
STRUCTURE OF CAPABILITIES:
    # Numeric indicator layers
    # name in polydata is f'capabilities_{persona}_{capability}_{numeric indicator}'
    # possible values are 0 and 1
    # example:
        polydata.point_data['capabilities_reptile_shelter_fallen-log'] can be 0 or 1

    # Capability layers
    # name in polydata is f'capabilities_{persona}_{capability}'
    # possible values are all the nummeric indicators for that capability
    # example:
        polydata.point_data['capabilities_reptile_shelter'] can be 'none', 'fallen-log', 'fallen-tree'
    
    # Persona aggregate capabilities layer
    # name in polydata is f'capabilities_{persona}'
    # possible values are all the capability layers for that persona
    # example:
        polydata.point_data['capabilities_reptile'] can be 'none', 'traverse', 'forage', 'shelter'

"""
# Mapping for urban elements counts to capabilityID
URBAN_ELEMENT_ID_MAP = {
    # Bird
    ('bird', 'socialise', 'canopy_volume'): '1.1.2',  # Canopy volume across control levels
    ('bird', 'feed', 'artificial_bark'): '1.2.1',     # Artificial bark installed
    ('bird', 'raise-young', 'artificial_hollows'): '1.3.1',  # Artificial hollows
    
    # Reptile
    ('reptile', 'traverse', 'urban_conversion'): '2.1.1',  # Urban element conversion
    ('reptile', 'forage', 'mistletoe'): '2.2.3',      # Number of epiphytes installed
    ('reptile', 'forage', 'low_veg'): '2.2.1',        # Count of voxels converted
    ('reptile', 'forage', 'dead_branch'): '2.2.2',    # Dead branch volume
    ('reptile', 'shelter', 'near_fallen_5m'): '2.3.1', # Supporting fallen logs
    
    # Tree
    ('tree', 'grow', 'trees_planted'): '3.1.1',       # Trees planted
    ('tree', 'age', 'AGE-IN-PLACE_actions'): '3.2.1', # AGE-IN-PLACE actions
    ('tree', 'persist', 'eligible_soil'): '3.3.3'     # Eligible soil in urban elements
}

"""
CAPABILITIES INFO    
1.  Bird capabilities:

1.1. Socialise: Points where 'stat_perch branch' > 0
   - Birds need branches to perch on for social activities
   
   Numeric indicators:
   - 1.1.1 bird_socialise: Total voxels where stat_perch branch > 0 : 'perch branch'
        -label for graph: 'Perchable canopy volume'
   
   Urban element / design action: 
   - 1.1.1 Canopy volume across control levels: high, medium, low
        # search criteria: Count of 'capabilities_bird_socialise_perch-branch', broken down by ['forest_control'], where high == 'street-tree', medium == 'park-tree', low == 'reserve-tree' OR 'improved-tree'

1.2 Feed: Points where 'stat_peeling bark' > 0
   - Birds feed on insects found under peeling bark
   
   Numeric indicators:
   - 1.2.1 bird_feed: Total voxels where stat_peeling bark > 0 : 'peeling bark'
        - label for graph: 'Peeling bark volume'
   
   Urban element / design action:
   - 1.2.1 Artificial bark installed on branches, utility poles
        #search criteria: Count of 'capabilities_bird_feed_peeling-bark' where polydata['precolonial'] == False
        TO DO: could include eucs

1.3. Raise Young: Points where 'stat_hollow' > 0
   - Birds need hollows in trees to nest and raise their young
   
   Numeric indicators:
   - 1.3.1 bird_raise_young: Total voxels where stat_hollow > 0 : 'hollow'
        - label for graph: 'Hollow count'

   Urban element / design action:
   - 1.3.1 Artificial hollows installed on branches, utility poles
        #search criteria: Count of 'capabilities_bird_raise-young_hollow' where polydata['precolonial'] == False
        TO DO: could include eucs


2. Reptile capabilities:

2.1. Traverse: Points where 'search_bioavailable' != 'none'
   - Reptiles can move through any bioavailable space
   
   Numeric indicators:
   - 2.1.1 reptile_traverse: Total voxels where search_bioavailable != 'none' : traversable
        - label for graph: 'Non-paved surface area'

   Urban element / design action: 
   - 2.1.1 Count of site voxels converted from: car parks, roads, green roofs, brown roofs, facades
        # search criteria: Total voxels where polydata['capabilities_reptile_traverse'] == 'traversable', 
        # broken down by the defined urban element catagories in polydata['search_urban_elements']


2.2 Foraige: Points where any of the following conditions are met:
   - 'search_bioavailable' == 'low-vegetation' (areas reptiles can move through)
   - 'stat_dead branch' > 0 (dead branches in canopy generate coarse woody debris)
   - 'stat_epiphyte' > 0 (epiphytes in canopy generate fallen leaves)
   
   Numeric indicators:
   - 2.2.1 reptile_forage_low_veg: Voxels where search_bioavailable == 'low-vegetation' : 'ground-cover'
         - label for graph: 'Low vegetation surface area'
   - 2.2.2 reptile_forage_dead_branch: Voxels where stat_dead branch > 0 : 'dead-branch'
         - label for graph: 'Canopy dead branch volume'
   - 2.2.3 reptile_forage_epiphyte: Voxels where stat_epiphyte > 0 : 'epiphyte'
        - label for graph: 'Epiphyte count'

   Urban element / design action:
   - 2.2.1 Count of voxels converted from  : car parks, roads, green roofs, brown roofs, facades
        # search criteria: Count of 'reptile_forage_low_veg', broken down by the defined urban element catagories in polydata['search_urban_elements']
   - 2.2.2 Dead branch volume across control levels: high, medium, low
        # search criteria: Count of 'reptile_forage_dead_branch', broken down by their ['forest_control'], where high == 'street-tree', medium == 'park-tree', low == 'reserve-tree' OR 'improved-tree'
   - 2.2.3 Number of epiphytes installed in elms
        # search criteria: Count of 'reptile_forage_epiphyte' where 'forest_precolonial' == False

2.3. Shelter: Points where any of the following conditions are met:
   - 'stat_fallen log' > 0 (fallen logs provide shelter)
   - 'forest_size' == 'fallen' (fallen trees provide shelter)
   
   Numeric indicators:
   - 2.3.1 reptile_shelter_fallen_log: Voxels where stat_fallen log > 0 : 'fallen-log'
        - label for graph: 'Nurse log volume'
   - 2.3.2 reptile_shelter_fallen_tree: Voxels where forest_size == 'fallen' : 'fallen-tree'
        - label for graph: 'Fallen tree volume'

   Urban element / design action:
   - 2.3.1 Count of ground elements supporting fallen logs  : car parks, roads, green roofs, brown roofs, facades within 5m of fallen trees and logs
        #search criteria. Use a ckdTree to find points within 5m where 'reptile_shelter_fallen_log' == True 
        #break these down by the defined urban element catagories in polydata['search_urban_elements']
   - 2.3.2 Count of ground elements supporting fallen trees  : car parks, roads, green roofs, brown roofs, facades within 5m of fallen trees and logs
        #search criteria. Use a ckdTree to find points within 5m where 'reptile_shelter_fallen_tree' == True 
        #break these down by the defined urban element catagories in polydata['search_urban_elements']

3. Tree capabilities:

3.1. Grow: Points where 'stat_other' > 0
   - Areas where trees can grow and establish
   
   Numeric indicators:
   - 3.1.1 tree_grow (Canopy biovolume): Total voxels where stat_other > 0 : 'volume'
        - label for graph: 'Forest biovolume'
   #Urban element / design action: 
   - 3.1.1 count of number of trees planted this timestep
        # search criteria: sum of df['number_of_trees_to_plant']      

3.2. Age: Points where 'forest_control' == 'improved-tree' OR 'forest_control' == 'reserve-tree'
   - Areas where trees are protected and can mature
   
   Numeric indicators:
   #TODO: change numeric indicators to 3.2.1: forest_size == 'senesent', 3.2.2: forest_size == 'snag' or 'fallen'
   - 3.2.1 tree_age: Total voxels where forest_control == 'improved-tree' : 'improved-tree'
        - label for graph: 'Canopy volume supported by humans'
   - 3.2.2 tree_age: Total voxels where forest_control == 'reserve-tree' : 'reserve-tree'
        - label for graph: 'Canopy volume autonomous'

   #Urban element / design action: 
   -3.2.1 count of AGE-IN-PLACE actions: exoskeletons, habitat islands, depaved areas
        # search criteria:  counts of df['rewilded'] == 'footprint-depaved','exoskeleton','node-rewilded'
     -3.2.1 count of AGE-IN-PLACE actions: exoskeletons, habitat islands, depaved areas
        # search criteria:  counts of df['rewilded'] == 'footprint-depaved','exoskeleton','node-rewilded'

3.3. Persist: Terrain points eligible for new tree plantings (ie. depaved and unmanaged land away from trees)

    Numeric indicator:
    - 3.3.1 scenario_rewildingPlantings >= 1 : 'eligible-soil'
        - label for graph: 'Ground area for tree recruitment'
    - 3.3.2 volume of saplings
        - #search criteria: count where 'forest_size' == 'small'
        - label for graph: 'Sapling volume'

    #Urban element / design action:
    -3.3.1/ Count of site voxels converted from: car parks, roads, etc
        #count of subset of polydata['scenario_rewildingPlantings'] >= 1, broken down by the defined urban element catagories in polydata['search_urban_elements']

"""

"""
#search criteria for many of the Urban element / design action count 'urban elements'. Use these:
##polydata['search_urban_elements'] catagories:
'open space'
'green roof'
'brown roof'
'facade'
'roadway'
'busy roadway'
'existing conversion'
'other street potential'
'parking'
"""

def create_capabilities_info():
    """Create a dataframe with structured capability information from the docstring."""
    lifecycle_colors = a_vis_colors.get_lifecycle_colors()
    resource_colors = a_vis_colors.get_resource_colours()
    envelope_colors = a_vis_colors.get_bioenvelope_colors()
    # Darken the 'other' color to make it less grey and more visible
    other_colour = (
        max(0, resource_colors['other'][0] - 30),
        max(0, resource_colors['other'][1] - 30),
        max(0, resource_colors['other'][2] - 30)
    )
    
    # Define the capabilities directly from docstring (numeric indicator level only)
    data = [
        # Bird capabilities
        {'persona': 'bird', 'capability': 'socialise', 'numeric_indicator': 'perch-branch', 'capability_id': '1.1.1', 'indicator_no': 1, 'color': resource_colors['perch branch']},
        {'persona': 'bird', 'capability': 'feed', 'numeric_indicator': 'peeling-bark', 'capability_id': '1.2.1', 'indicator_no': 2, 'color': resource_colors['peeling bark']},
        {'persona': 'bird', 'capability': 'raise-young', 'numeric_indicator': 'hollow', 'capability_id': '1.3.1', 'indicator_no': 3, 'color': resource_colors['hollow']},
        
        # Reptile capabilities
        {'persona': 'reptile', 'capability': 'traverse', 'numeric_indicator': 'traversable', 'capability_id': '2.1.1', 'indicator_no': 1, 'color': envelope_colors['brownRoof']},
        {'persona': 'reptile', 'capability': 'forage', 'numeric_indicator': 'ground-cover', 'capability_id': '2.2.1', 'indicator_no': 2, 'color': envelope_colors['rewilded']},
        {'persona': 'reptile', 'capability': 'forage', 'numeric_indicator': 'dead-branch', 'capability_id': '2.2.2', 'indicator_no': 3, 'color': resource_colors['dead branch']},
        {'persona': 'reptile', 'capability': 'forage', 'numeric_indicator': 'epiphyte', 'capability_id': '2.2.3', 'indicator_no': 4, 'color': resource_colors['epiphyte']},
        {'persona': 'reptile', 'capability': 'shelter', 'numeric_indicator': 'fallen-log', 'capability_id': '2.3.1', 'indicator_no': 5, 'color': resource_colors['fallen log']},
        {'persona': 'reptile', 'capability': 'shelter', 'numeric_indicator': 'fallen-tree', 'capability_id': '2.3.2', 'indicator_no': 6, 'color': lifecycle_colors['fallen']},
        
        # Tree capabilities
        {'persona': 'tree', 'capability': 'grow', 'numeric_indicator': 'volume', 'capability_id': '3.1.1', 'indicator_no': 1, 'color': other_colour},
        {'persona': 'tree', 'capability': 'age', 'numeric_indicator': 'senescing-tree', 'capability_id': '3.2.1', 'indicator_no': 2, 'color': lifecycle_colors['senescing']},
        {'persona': 'tree', 'capability': 'age', 'numeric_indicator': 'rotting-tree', 'capability_id': '3.2.2', 'indicator_no': 3, 'color': lifecycle_colors['fallen']},
        {'persona': 'tree', 'capability': 'persist', 'numeric_indicator': 'eligible-soil', 'capability_id': '3.3.3', 'indicator_no': 4, 'color': envelope_colors['rewilded']},
        {'persona': 'tree', 'capability': 'persist', 'numeric_indicator': 'sapling-volume', 'capability_id': '3.3.3', 'indicator_no': 5, 'color': lifecycle_colors['small']}
    ]

    # Create color_hex key-value pairs in each dictionary
    for item in data:
        rgb = item['color']
        item['color_hex'] = f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Add RGB string representation (e.g., 'rgb(255, 152, 0)')
    df['color_rgb_string'] = df['color'].apply(lambda x: f'rgb({x[0]}, {x[1]}, {x[2]})')
    
    # Assign hpos - first digit of capability id (ie. bird = 1, reptile = 2, tree = 3)
    df['hpos'] = df['capability_id'].apply(lambda x: int(x.split('.')[0]) - 1)
    
    # Assign capability_no - second digit of capability id
    df['capability_no'] = df['capability_id'].apply(lambda x: int(x.split('.')[1]) - 1)
    
    # Create layer names for the polydata
    df['layer_name'] = df.apply(
        lambda row: f"capabilities_{row['persona']}_{row['capability']}_{row['numeric_indicator']}", 
        axis=1
    )
    
    return df

def create_bird_capabilities(polydata):
    print("  Creating bird capability layers...")
    
    # Initialize capability_numeric_indicator layers (boolean arrays)
    bird_socialise_perch_branch = np.zeros(polydata.n_points, dtype=bool)
    bird_feed_peeling_bark = np.zeros(polydata.n_points, dtype=bool)
    bird_raise_young_hollow = np.zeros(polydata.n_points, dtype=bool)
    
    # Initialize individual_capability layers (string arrays)
    capabilities_bird_socialise = np.full(polydata.n_points, 'none', dtype='<U20')
    capabilities_bird_feed = np.full(polydata.n_points, 'none', dtype='<U20')
    capabilities_bird_raise_young = np.full(polydata.n_points, 'none', dtype='<U20')
    
    # Initialize persona_aggregate_capabilities layer
    capabilities_bird = np.full(polydata.n_points, 'none', dtype='<U20')
    
    # 1.1. Bird Socialise: points in stat_perch_branch > 0
    # - Birds need branches to perch on for social activities
    perch_data = polydata.point_data['stat_perch branch']
    if np.issubdtype(perch_data.dtype, np.number):
        perch_branch_mask = perch_data > 0
    else:
        perch_branch_mask = (perch_data != 'none') & (perch_data != '') & (perch_data != 'nan')
    
    # 1.1.1 capability_numeric_indicator: perch branch
    bird_socialise_perch_branch |= perch_branch_mask
    capabilities_bird_socialise[perch_branch_mask] = 'perch-branch'
    capabilities_bird[perch_branch_mask] = 'socialise'
    print(f"    Bird socialise points: {np.sum(perch_branch_mask):,}")
    
    # 1.2. Bird Feed: points in stat_peeling bark > 0
    # - Birds feed on insects found under peeling bark
    bark_data = polydata.point_data['stat_peeling bark']
    if np.issubdtype(bark_data.dtype, np.number):
        peeling_bark_mask = bark_data > 0
    else:
        peeling_bark_mask = (bark_data != 'none') & (bark_data != '') & (bark_data != 'nan')
    
    # 1.2.1 capability_numeric_indicator: peeling bark
    bird_feed_peeling_bark |= peeling_bark_mask
    capabilities_bird_feed[peeling_bark_mask] = 'peeling-bark'
    # Override any existing values (later capabilities take precedence)
    capabilities_bird[peeling_bark_mask] = 'feed'
    print(f"    Bird feed points: {np.sum(peeling_bark_mask):,}")
    
    # 1.3. Bird Raise Young: points in stat_hollow > 0
    # - Birds need hollows in trees to nest and raise their young
    hollow_data = polydata.point_data['stat_hollow']
    if np.issubdtype(hollow_data.dtype, np.number):
        hollow_mask = hollow_data > 0
    else:
        hollow_mask = (hollow_data != 'none') & (hollow_data != '') & (hollow_data != 'nan')
    
    # 1.3.1 capability_numeric_indicator: hollow
    bird_raise_young_hollow |= hollow_mask
    capabilities_bird_raise_young[hollow_mask] = 'hollow'
    # Override any existing values (later capabilities take precedence)
    capabilities_bird[hollow_mask] = 'raise-young'
    print(f"    Bird raise-young points: {np.sum(hollow_mask):,}")
    
    # Add capability_numeric_indicator layers to polydata
    polydata.point_data['capabilities_bird_socialise_perch-branch'] = bird_socialise_perch_branch
    polydata.point_data['capabilities_bird_feed_peeling-bark'] = bird_feed_peeling_bark
    polydata.point_data['capabilities_bird_raise-young_hollow'] = bird_raise_young_hollow
    
    # Add individual_capability layers to polydata
    polydata.point_data['capabilities_bird_socialise'] = capabilities_bird_socialise
    polydata.point_data['capabilities_bird_feed'] = capabilities_bird_feed
    polydata.point_data['capabilities_bird_raise-young'] = capabilities_bird_raise_young
    
    # Add persona_aggregate_capabilities layer to polydata
    polydata.point_data['capabilities_bird'] = capabilities_bird
    
    return polydata

def create_reptile_capabilities(polydata):
    print("  Creating reptile capability layers...")
    
    # Initialize capability_numeric_indicator layers (boolean arrays)
    reptile_traverse_traversable = np.zeros(polydata.n_points, dtype=bool)
    reptile_forage_ground_cover = np.zeros(polydata.n_points, dtype=bool)
    reptile_forage_dead_branch = np.zeros(polydata.n_points, dtype=bool)
    reptile_forage_epiphyte = np.zeros(polydata.n_points, dtype=bool)
    reptile_shelter_fallen_log = np.zeros(polydata.n_points, dtype=bool)
    reptile_shelter_fallen_tree = np.zeros(polydata.n_points, dtype=bool)
    
    # Initialize individual_capability layers (string arrays)
    capabilities_reptile_traverse = np.full(polydata.n_points, 'none', dtype='<U20')
    capabilities_reptile_forage = np.full(polydata.n_points, 'none', dtype='<U20')
    capabilities_reptile_shelter = np.full(polydata.n_points, 'none', dtype='<U20')
    
    # Initialize persona_aggregate_capabilities layer
    capabilities_reptile = np.full(polydata.n_points, 'none', dtype='<U20')
    
    # 2.1. Reptile Traverse: points in 'search_bioavailable' != 'none'
    # - Reptiles can move through any bioavailable space
    bioavailable_data = polydata.point_data['search_bioavailable']
    traversable_mask = bioavailable_data != 'none'
    
    # 2.1.1 capability_numeric_indicator: traversable
    reptile_traverse_traversable |= traversable_mask
    capabilities_reptile_traverse[traversable_mask] = 'traversable'
    capabilities_reptile[traversable_mask] = 'traverse'
    print(f"    Reptile traverse points: {np.sum(traversable_mask):,}")
    
    # 2.2. Reptile Forage: points with multiple conditions
    
    # 2.2.1 Low vegetation points - ground cover
    ground_cover_mask = polydata.point_data['search_bioavailable'] == 'low-vegetation'
    reptile_forage_ground_cover |= ground_cover_mask
    capabilities_reptile_forage[ground_cover_mask] = 'ground-cover'
    # Override any existing values in capabilities_reptile if needed
    capabilities_reptile[ground_cover_mask] = 'forage'
    print(f"    Reptile forage ground cover points: {np.sum(ground_cover_mask):,}")
    
    # 2.2.2 Dead branch points
    dead_branch_data = polydata.point_data['stat_dead branch']
    if np.issubdtype(dead_branch_data.dtype, np.number):
        dead_branch_mask = dead_branch_data > 0
    else:
        dead_branch_mask = (dead_branch_data != 'none') & (dead_branch_data != '') & (dead_branch_data != 'nan')
    
    reptile_forage_dead_branch |= dead_branch_mask
    capabilities_reptile_forage[dead_branch_mask] = 'dead-branch'
    # Override any existing values in capabilities_reptile if needed
    capabilities_reptile[dead_branch_mask] = 'forage'
    print(f"    Reptile forage dead branch points: {np.sum(dead_branch_mask):,}")
    
    # 2.2.3 Epiphyte points
    epiphyte_data = polydata.point_data['stat_epiphyte']
    if np.issubdtype(epiphyte_data.dtype, np.number):
        epiphyte_mask = epiphyte_data > 0
    else:
        epiphyte_mask = (epiphyte_data != 'none') & (epiphyte_data != '') & (epiphyte_data != 'nan')
    
    reptile_forage_epiphyte |= epiphyte_mask
    capabilities_reptile_forage[epiphyte_mask] = 'epiphyte'
    # Override any existing values in capabilities_reptile if needed
    capabilities_reptile[epiphyte_mask] = 'forage'
    print(f"    Reptile forage epiphyte points: {np.sum(epiphyte_mask):,}")
    
    # 2.3.1 Fallen log points
    fallen_log_data = polydata.point_data['stat_fallen log']
    if np.issubdtype(fallen_log_data.dtype, np.number):
        fallen_log_mask = fallen_log_data > 0
    else:
        fallen_log_mask = (fallen_log_data != 'none') & (fallen_log_data != '') & (fallen_log_data != 'nan')
    
    reptile_shelter_fallen_log |= fallen_log_mask
    capabilities_reptile_shelter[fallen_log_mask] = 'fallen-log'
    # Override any existing values in capabilities_reptile if needed
    capabilities_reptile[fallen_log_mask] = 'shelter'
    print(f"    Reptile shelter fallen log points: {np.sum(fallen_log_mask):,}")
    
    # 2.3.2 Fallen tree points
    forest_size = polydata.point_data['forest_size']
    if forest_size.dtype.kind == 'U' or forest_size.dtype.kind == 'S':  # String types
        fallen_tree_mask = (forest_size == 'fallen')
    else:
        fallen_tree_mask = np.zeros(polydata.n_points, dtype=bool)  # No fallen trees if numeric
    
    reptile_shelter_fallen_tree |= fallen_tree_mask
    capabilities_reptile_shelter[fallen_tree_mask] = 'fallen-tree'
    # Override any existing values in capabilities_reptile if needed
    capabilities_reptile[fallen_tree_mask] = 'shelter'
    print(f"    Reptile shelter fallen tree points: {np.sum(fallen_tree_mask):,}")
    
    # Add capability_numeric_indicator layers to polydata
    polydata.point_data['capabilities_reptile_traverse_traversable'] = reptile_traverse_traversable
    polydata.point_data['capabilities_reptile_forage_ground-cover'] = reptile_forage_ground_cover
    polydata.point_data['capabilities_reptile_forage_dead-branch'] = reptile_forage_dead_branch
    polydata.point_data['capabilities_reptile_forage_epiphyte'] = reptile_forage_epiphyte
    polydata.point_data['capabilities_reptile_shelter_fallen-log'] = reptile_shelter_fallen_log
    polydata.point_data['capabilities_reptile_shelter_fallen-tree'] = reptile_shelter_fallen_tree
    
    # Add individual_capability layers to polydata
    polydata.point_data['capabilities_reptile_traverse'] = capabilities_reptile_traverse
    polydata.point_data['capabilities_reptile_forage'] = capabilities_reptile_forage
    polydata.point_data['capabilities_reptile_shelter'] = capabilities_reptile_shelter
    
    # Add persona_aggregate_capabilities layer to polydata
    polydata.point_data['capabilities_reptile'] = capabilities_reptile
    
    return polydata

def create_tree_capabilities(polydata):
    print("  Creating tree capability layers...")
    
    # Initialize boolean arrays for all capabilities
    n_points = polydata.n_points
    tree_grow_volume = np.zeros(n_points, dtype=bool)
    tree_age_senescent_tree = np.zeros(n_points, dtype=bool)
    tree_age_rotting_tree = np.zeros(n_points, dtype=bool)
    tree_persist_eligible_soil = np.zeros(n_points, dtype=bool)
    tree_persist_sapling_volume = np.zeros(n_points, dtype=bool)
    
    # Initialize string arrays for capability categories
    capabilities_tree_grow = np.full(n_points, '', dtype='U20')
    capabilities_tree_age = np.full(n_points, '', dtype='U20')
    capabilities_tree_persist = np.full(n_points, '', dtype='U20')
    capabilities_tree = np.full(n_points, '', dtype='U20')
    
    # 3.1. Tree Grow: Biovolume
    # 3.1.1 capability_numeric_indicator: biovolume
    #print all point data keys
    print(polydata.point_data.keys())
    volume_mask = polydata.point_data['resource_other'] >= 0
    tree_grow_volume |= volume_mask
    capabilities_tree_grow[volume_mask] = 'biovolume'
    capabilities_tree[volume_mask] = 'grow'
    print(f"    Tree grow points: {np.sum(tree_grow_volume):,}")

    # 3.2. Tree Age: 
    # 3.2.1 capability_numeric_indicator: senescent tree
    #forest_size == 'senescing'
    senescent_tree_mask = polydata.point_data['forest_size'] == 'senescing'
    tree_age_senescent_tree |= senescent_tree_mask
    capabilities_tree_age[senescent_tree_mask] = 'senescing-tree'
    # Override any existing values (later capabilities take precedence)
    capabilities_tree[senescent_tree_mask] = 'age'
    print(f"    Tree age points from senescing-tree: {np.sum(senescent_tree_mask):,}")
    
    # 3.2.2 capability_numeric_indicator: 'dead'
    #polydata.point_data['forest_size'] == 'snag' or 'fallen'
    dead_tree_mask = (polydata.point_data['forest_size'] == 'snag') | (polydata.point_data['forest_size'] == 'fallen')
    tree_age_rotting_tree |= dead_tree_mask
    capabilities_tree_age[dead_tree_mask] = 'rotting-tree'
    # Override any existing values (later capabilities take precedence)
    capabilities_tree[dead_tree_mask] = 'age'
    print(f"    Tree age points from decay-tree: {np.sum(dead_tree_mask):,}")
    
    # 3.3. Tree Persist
    # 3.3.1 Tree Persist: eligible soil
    #  capability_numeric_indicator: eligible soil
    # Use rewilding plantings
    rewilding_data = polydata.point_data['scenario_rewildingPlantings']
    if np.issubdtype(rewilding_data.dtype, np.number):
        eligible_soil_mask = rewilding_data >= 1
    else:
        eligible_soil_mask = np.zeros(polydata.n_points, dtype=bool)
        for i, val in enumerate(rewilding_data):
            if val not in ['none', '', 'nan']:
                try:
                    eligible_soil_mask[i] = float(val) >= 1
                except (ValueError, TypeError):
                    pass
    
    tree_persist_eligible_soil |= eligible_soil_mask
    capabilities_tree_persist[eligible_soil_mask] = 'eligible-soil'
    # Override any existing values (later capabilities take precedence)
    capabilities_tree[eligible_soil_mask] = 'persist'
    print(f"    Tree persist points from eligible soil (scenario): {np.sum(eligible_soil_mask):,}")

    #3.3.2 Tree Persist: sapling volume
    #  capability_numeric_indicator: sapling volume
    # Use small trees
    small_tree_mask = polydata.point_data['forest_size'] == 'small'
    tree_persist_sapling_volume |= small_tree_mask
    capabilities_tree_persist[small_tree_mask] = 'sapling-volume'
    # Override any existing values (later capabilities take precedence)
    capabilities_tree[small_tree_mask] = 'persist'
    
    # Add capability_numeric_indicator layers to polydata
    polydata.point_data['capabilities_tree_grow_volume'] = tree_grow_volume
    polydata.point_data['capabilities_tree_age_senescing-tree'] = tree_age_senescent_tree
    polydata.point_data['capabilities_tree_age_rotting-tree'] = tree_age_rotting_tree
    polydata.point_data['capabilities_tree_persist_eligible-soil'] = tree_persist_eligible_soil
    polydata.point_data['capabilities_tree_persist_sapling-volume'] = tree_persist_sapling_volume
    # Add individual_capability layers to polydata
    polydata.point_data['capabilities_tree_grow'] = capabilities_tree_grow
    polydata.point_data['capabilities_tree_age'] = capabilities_tree_age
    polydata.point_data['capabilities_tree_persist'] = capabilities_tree_persist
    
    # Add persona_aggregate_capabilities layer to polydata
    polydata.point_data['capabilities_tree'] = capabilities_tree
    
    return polydata

def main():
    """Main function to generate capability-enhanced pyvista polydata VTKs and save capabilities info"""
    # Create capabilities info dataframe
    capabilities_info = create_capabilities_info()
    
    # Save capabilities info to CSV
    output_dir = Path('data/revised/final/stats/arboreal-future-stats/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    capabilities_info_path = output_dir / 'capabilities_info.csv'
    capabilities_info.to_csv(capabilities_info_path, index=False)
    print(f"Saved capabilities info to {capabilities_info_path}")
    
    #--------------------------------------------------------------------------
    # GATHER USER INPUTS
    #--------------------------------------------------------------------------
    # Default values
    default_sites = ['trimmed-parade']
    default_scenarios = ['baseline', 'positive', 'trending']
    default_years = [0, 10, 30, 60, 180]
    default_voxel_size = 1
    
    # Ask for sites
    sites_input = input(f"Enter site(s) to process (comma-separated) or press Enter for default {default_sites}: ")
    sites = sites_input.split(',') if sites_input else default_sites
    sites = [site.strip() for site in sites]
    
    # Ask for scenarios
    print("\nAvailable scenarios: baseline, positive, trending")
    scenarios_input = input(f"Enter scenario(s) to process (comma-separated) or press Enter for default {default_scenarios}: ")
    scenarios = scenarios_input.split(',') if scenarios_input else default_scenarios
    scenarios = [scenario.strip() for scenario in scenarios]
    
    # Check if baseline is included
    include_baseline = 'baseline' in scenarios
    if include_baseline:
        scenarios.remove('baseline')
    
    # Ask for years/trimesters
    years_input = input(f"Enter years to process (comma-separated) or press Enter for default {default_years}: ")
    years = [int(year.strip()) for year in years_input.split(',')] if years_input else default_years
    
    # Ask for voxel size
    voxel_size_input = input(f"Enter voxel size (default {default_voxel_size}): ")
    voxel_size = int(voxel_size_input) if voxel_size_input else default_voxel_size
    
    # Print summary of selected options
    print("\n===== Processing with the following parameters =====")
    print(f"Sites: {sites}")
    print(f"Scenarios: {scenarios}")
    print(f"Process baseline: {include_baseline}")
    print(f"Years/Trimesters: {years}")
    print(f"Voxel Size: {voxel_size}")
    
    # Confirm proceeding
    confirm = input("\nProceed with these settings? (yes/no, default yes): ")
    if confirm.lower() in ['no', 'n']:
        print("Operation cancelled.")
        return
    
    #--------------------------------------------------------------------------
    # PROCESS CAPABILITIES AND SAVE RESULTS
    #--------------------------------------------------------------------------
    print("\n===== PROCESSING CAPABILITIES =====")
    
    # Process each site
    for site in sites:
        print(f"\n=== Processing site: {site} ===")
        
        # Process baseline if requested
        if include_baseline:
            print(f"Processing baseline for site: {site}")
            baseline_path = f'data/revised/final/baselines/{site}_baseline_combined_{voxel_size}_urban_features.vtk'
            
            baseline_polydata = pv.read(baseline_path)
            print(f"  Creating capability layers for baseline...")

            # Make a scenario_rewildingPlantings in the baseline
            bioavailable = baseline_polydata.point_data['search_bioavailable']
            eligible_soil_mask = bioavailable == 'low-vegetation'
            baseline_polydata.point_data['scenario_rewildingPlantings'] = eligible_soil_mask

            # Create capabilities for baseline
            baseline_polydata = create_bird_capabilities(baseline_polydata)
            baseline_polydata = create_reptile_capabilities(baseline_polydata)
            baseline_polydata = create_tree_capabilities(baseline_polydata)
            
            # Save updated baseline with capabilities
            baseline_output_path = f'data/revised/final/{site}/{site}_baseline_resources_{voxel_size}_with_capabilities.vtk'
            baseline_polydata.save(baseline_output_path)
            print(f"  Saved baseline with capabilities to {baseline_output_path}")
        
        # Process each scenario
        for scenario in scenarios:
            print(f"\n=== Processing scenario: {scenario} for site: {site} ===")
            
            # Process each year
            for year in years:
                print(f"Processing year {year} for site: {site}, scenario: {scenario}")
                
                # Load VTK file
                vtk_path = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_urban_features.vtk'
                polydata = pv.read(vtk_path)
                
                # Create capabilities
                print(f"  Creating capability layers...")
                polydata = create_bird_capabilities(polydata)
                polydata = create_reptile_capabilities(polydata)
                polydata = create_tree_capabilities(polydata)
                
                # Save updated VTK file with capabilities
                output_path = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_with_capabilities.vtk'
                polydata.save(output_path)
                print(f"  Saved polydata with capabilities to {output_path}")
    
    print("\n===== All processing completed =====")

if __name__ == "__main__":
    main()