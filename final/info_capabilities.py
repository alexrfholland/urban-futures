import pyvista as pv
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os

def load_vtk_file(site, scenario, voxel_size, year):
    """Load VTK file with features for given parameters"""
    filepath = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_with_features.vtk'
    try:
        return pv.read(filepath)
    except Exception as e:
        print(f"Error loading VTK file for year {year}: {e}")
        return None

def create_bird_capabilities(vtk_data):
    """Create capability layers for birds
    
    Bird capabilities:
    1. Socialise: Points where 'resource_perch branch' > 0
       - Birds need branches to perch on for social activities
       
       Numeric indicators:
       - bird_socialise: Total voxels where resource_perch branch > 0 : 'perch branch'
    
    2. Feed: Points where 'resource_peeling bark' > 0
       - Birds feed on insects found under peeling bark
       
       Numeric indicators:
       - bird_feed: Total voxels where resource_peeling bark > 0 : 'peeling bark'
    
    3. Raise Young: Points where 'resource_hollow' > 0
       - Birds need hollows in trees to nest and raise their young
       
       Numeric indicators:
       - bird_raise_young: Total voxels where resource_hollow > 0 : 'hollow'
    """
    print("  Creating bird capability layers...")
    
    # Initialize individual boolean capability layers
    bird_socialise = np.zeros(vtk_data.n_points, dtype=bool)
    bird_feed = np.zeros(vtk_data.n_points, dtype=bool)
    bird_raise_young = np.zeros(vtk_data.n_points, dtype=bool)
    
    # Initialize aggregate categorical layer
    capabilities_bird = np.full(vtk_data.n_points, 'none', dtype='<U20')
    
    # Bird Socialise: points in resource_perch branch > 0
    if 'resource_perch branch' in vtk_data.point_data:
        perch_branch_data = vtk_data.point_data['resource_perch branch']
        print(f"    Found 'resource_perch branch' data with type {perch_branch_data.dtype}")
        
        if np.issubdtype(perch_branch_data.dtype, np.number):
            bird_socialise_mask = perch_branch_data > 0
            print(f"    Numeric data: {np.sum(bird_socialise_mask):,} points with value > 0")
        else:
            bird_socialise_mask = (perch_branch_data != 'none') & (perch_branch_data != '') & (perch_branch_data != 'nan')
            print(f"    String data: {np.sum(bird_socialise_mask):,} points with non-empty values")
        
        bird_socialise |= bird_socialise_mask
        capabilities_bird[bird_socialise_mask] = 'socialise'
        print(f"    Bird socialise points: {np.sum(bird_socialise_mask):,}")
    else:
        print("    WARNING: 'resource_perch branch' not found in point data")
        # List all resource keys to help debug
        resource_keys = [key for key in vtk_data.point_data.keys() if key.startswith('resource_')]
        print(f"    Available resource keys: {resource_keys}")
    
    # Bird Feed: points in resource_peeling bark > 0
    if 'resource_peeling bark' in vtk_data.point_data:
        peeling_bark_data = vtk_data.point_data['resource_peeling bark']
        if np.issubdtype(peeling_bark_data.dtype, np.number):
            bird_feed_mask = peeling_bark_data > 0
        else:
            bird_feed_mask = (peeling_bark_data != 'none') & (peeling_bark_data != '') & (peeling_bark_data != 'nan')
        
        bird_feed |= bird_feed_mask
        # Override any existing values (later capabilities take precedence)
        capabilities_bird[bird_feed_mask] = 'feed'
        print(f"    Bird feed points: {np.sum(bird_feed_mask):,}")
    else:
        print("    'resource_peeling bark' not found in point data")
    
    # Bird Raise Young: points in resource_hollow > 0
    if 'resource_hollow' in vtk_data.point_data:
        hollow_data = vtk_data.point_data['resource_hollow']
        if np.issubdtype(hollow_data.dtype, np.number):
            bird_raise_young_mask = hollow_data > 0
        else:
            bird_raise_young_mask = (hollow_data != 'none') & (hollow_data != '') & (hollow_data != 'nan')
        
        bird_raise_young |= bird_raise_young_mask
        # Override any existing values (later capabilities take precedence)
        capabilities_bird[bird_raise_young_mask] = 'raise-young'
        print(f"    Bird raise-young points: {np.sum(bird_raise_young_mask):,}")
    else:
        print("    'resource_hollow' not found in point data")
    
    # Add bird capability layers to vtk_data
    vtk_data.point_data['capabilities-bird-socialise'] = bird_socialise
    vtk_data.point_data['capabilities-bird-feed'] = bird_feed
    vtk_data.point_data['capabilities-bird-raise-young'] = bird_raise_young
    vtk_data.point_data['capabilities-bird'] = capabilities_bird
    
    return vtk_data

def create_reptile_capabilities(vtk_data):
    """Create capability layers for reptiles
    
    Reptile capabilities:
    1. Traverse: Points where 'search_bioavailable' != 'none'
       - Reptiles can move through any bioavailable space
       
       Numeric indicators:
       - reptile_traverse: Total voxels where search_bioavailable != 'none' : traversable
    
    2. Foraige: Points where any of the following conditions are met:
       - 'search_bioavailable' == 'low-vegetation' (areas reptiles can move through)
       - 'resource_dead branch' > 0 (dead branches provide foraging opportunities)
       - 'resource_epiphyte' > 0 (epiphytes provide habitat for prey)
       
       Numeric indicators:
       - reptile_forage_low_veg: Voxels where search_bioavailable == 'low-vegetation' : 'ground cover'
       - reptile_forage_dead_branch: Voxels where resource_dead branch > 0 : 'dead branch'
       - reptile_forage_epiphyte: Voxels where resource_epiphyte > 0 : 'epiphyte'
    
    3. Shelter: Points where any of the following conditions are met:
       - 'resource_fallen log' > 0 (fallen logs provide shelter)
       - 'forest_size' == 'fallen' (fallen trees provide shelter)
       
       Numeric indicators:
       - reptile_shelter_fallen_log: Voxels where resource_fallen log > 0 : 'fallen log'
       - reptile_shelter_fallen_tree: Voxels where forest_size == 'fallen' : 'fallen tree'
    """
    print("  Creating reptile capability layers...")
    
    # Initialize detailed component layers
    reptile_forage_low_veg = np.zeros(vtk_data.n_points, dtype=bool)
    reptile_forage_dead_branch = np.zeros(vtk_data.n_points, dtype=bool)
    reptile_forage_epiphyte = np.zeros(vtk_data.n_points, dtype=bool)
    
    reptile_shelter_fallen_log = np.zeros(vtk_data.n_points, dtype=bool)
    reptile_shelter_fallen_tree = np.zeros(vtk_data.n_points, dtype=bool)
    
    # Initialize aggregate capability layers
    capabilities_reptile_traverse = np.full(vtk_data.n_points, 'none', dtype='<U20')
    capabilities_reptile_forage = np.full(vtk_data.n_points, 'none', dtype='<U20')
    capabilities_reptile_shelter = np.full(vtk_data.n_points, 'none', dtype='<U20')
    
    # Initialize overall aggregate categorical layer
    capabilities_reptile = np.full(vtk_data.n_points, 'none', dtype='<U20')
    
    # Reptile Traverse: points in 'search_bioavailable' != 'none'
    if 'search_bioavailable' in vtk_data.point_data:
        bioavailable_data = vtk_data.point_data['search_bioavailable']
        reptile_traverse_mask = bioavailable_data != 'none'
        
        capabilities_reptile_traverse[reptile_traverse_mask] = 'traversable'
        capabilities_reptile[reptile_traverse_mask] = 'traverse'
        print(f"    Reptile traverse points: {np.sum(reptile_traverse_mask):,}")
    else:
        print("    'search_bioavailable' not found in point data")
    
    # Reptile Foraige: points in 'search_bioavailable' == 'low-vegetation' OR points in 'resource_dead branch' > 0 OR resource_epiphyte > 0
    reptile_forage_mask = np.zeros(vtk_data.n_points, dtype=bool)
    
    # Check low-vegetation points
    if 'search_bioavailable' in vtk_data.point_data:
        low_veg_mask = vtk_data.point_data['search_bioavailable'] == 'low-vegetation'
        reptile_forage_low_veg |= low_veg_mask
        reptile_forage_mask |= low_veg_mask
        capabilities_reptile_forage[low_veg_mask] = 'ground cover'
        print(f"    Reptile forage low-vegetation points: {np.sum(low_veg_mask):,}")
    
    # Check dead branch points
    if 'resource_dead branch' in vtk_data.point_data:
        dead_branch_data = vtk_data.point_data['resource_dead branch']
        if np.issubdtype(dead_branch_data.dtype, np.number):
            dead_branch_mask = dead_branch_data > 0
        else:
            dead_branch_mask = (dead_branch_data != 'none') & (dead_branch_data != '') & (dead_branch_data != 'nan')
        
        reptile_forage_dead_branch |= dead_branch_mask
        reptile_forage_mask |= dead_branch_mask
        capabilities_reptile_forage[dead_branch_mask] = 'dead branch'
        print(f"    Reptile forage dead branch points: {np.sum(dead_branch_mask):,}")
    else:
        print("    'resource_dead branch' not found in point data")
    
    # Check epiphyte points
    if 'resource_epiphyte' in vtk_data.point_data:
        epiphyte_data = vtk_data.point_data['resource_epiphyte']
        if np.issubdtype(epiphyte_data.dtype, np.number):
            epiphyte_mask = epiphyte_data > 0
        else:
            epiphyte_mask = (epiphyte_data != 'none') & (epiphyte_data != '') & (epiphyte_data != 'nan')
        
        reptile_forage_epiphyte |= epiphyte_mask
        reptile_forage_mask |= epiphyte_mask
        capabilities_reptile_forage[epiphyte_mask] = 'epiphyte'
        print(f"    Reptile forage epiphyte points: {np.sum(epiphyte_mask):,}")
    else:
        print("    'resource_epiphyte' not found in point data")
    
    # Override any existing values (later capabilities take precedence)
    capabilities_reptile[reptile_forage_mask] = 'forage'
    print(f"    Reptile forage total points: {np.sum(reptile_forage_mask):,}")
    
    # Reptile Shelter: points in 'resource_fallen log' > 0 OR points in 'forest_size' == 'fallen'
    reptile_shelter_mask = np.zeros(vtk_data.n_points, dtype=bool)
    
    # Check fallen log points
    if 'resource_fallen log' in vtk_data.point_data:
        fallen_log_data = vtk_data.point_data['resource_fallen log']
        if np.issubdtype(fallen_log_data.dtype, np.number):
            fallen_log_mask = fallen_log_data > 0
        else:
            fallen_log_mask = (fallen_log_data != 'none') & (fallen_log_data != '') & (fallen_log_data != 'nan')
        
        reptile_shelter_fallen_log |= fallen_log_mask
        reptile_shelter_mask |= fallen_log_mask
        capabilities_reptile_shelter[fallen_log_mask] = 'fallen log'
        print(f"    Reptile shelter fallen log points: {np.sum(fallen_log_mask):,}")
    else:
        print("    'resource_fallen log' not found in point data")
    
    # Check fallen forest points
    if 'forest_size' in vtk_data.point_data:
        forest_size = vtk_data.point_data['forest_size']
        if forest_size.dtype.kind == 'U' or forest_size.dtype.kind == 'S':  # String types
            fallen_forest_mask = forest_size == 'fallen'
            reptile_shelter_fallen_tree |= fallen_forest_mask
            reptile_shelter_mask |= fallen_forest_mask
            capabilities_reptile_shelter[fallen_forest_mask] = 'fallen tree'
            print(f"    Reptile shelter fallen tree points: {np.sum(fallen_forest_mask):,}")
    else:
        print("    'forest_size' not found in point data")
    
    # Override any existing values (later capabilities take precedence)
    capabilities_reptile[reptile_shelter_mask] = 'shelter'
    print(f"    Reptile shelter total points: {np.sum(reptile_shelter_mask):,}")
    
    # Add detailed component layers
    vtk_data.point_data['capabilities-reptile-forage-low-veg'] = reptile_forage_low_veg
    vtk_data.point_data['capabilities-reptile-forage-dead-branch'] = reptile_forage_dead_branch
    vtk_data.point_data['capabilities-reptile-forage-epiphyte'] = reptile_forage_epiphyte
    
    vtk_data.point_data['capabilities-reptile-shelter-fallen-log'] = reptile_shelter_fallen_log
    vtk_data.point_data['capabilities-reptile-shelter-fallen-tree'] = reptile_shelter_fallen_tree
    
    # Add capability layers
    vtk_data.point_data['capabilities-reptile-traverse'] = capabilities_reptile_traverse
    vtk_data.point_data['capabilities-reptile-forage'] = capabilities_reptile_forage
    vtk_data.point_data['capabilities-reptile-shelter'] = capabilities_reptile_shelter
    
    # Add overall aggregate layer
    vtk_data.point_data['capabilities-reptile'] = capabilities_reptile
    
    return vtk_data

def create_tree_capabilities(vtk_data):
    """Create capability layers for trees
    
    Tree capabilities:
    1. Grow: Points where 'resource_other' > 0
       - Areas where trees can grow and establish
       
       Numeric indicators:
       - tree_grow: Total voxels where resource_other > 0 : 'volume'
    
    2. Age: Points where 'search_design_action' == 'improved-tree' OR 'forset_control' == 'reserve-tree'
       - Areas where trees are protected and can mature
       
       Numeric indicators:
       - tree_age: Total voxels where search_design_action == 'improved-tree' : 'improved tree'
       - tree_age: Total voxels where forest_control == 'reserve-tree' : 'reserve tree'
    
    3. Persist: Points where both conditions are met:
       - 'search_bioavailable' == 'traversable' (suitable growing conditions)
       - Within 1m of points where 'forest_size' == 'medium' OR 'forest_size' == 'large'
         (proximity to mature trees that can reproduce)
       
       Numeric indicators:
       - tree_persist_near_medium: Traversable voxels within 1m of medium trees : 'medium tree'
       - tree_persist_near_large: Traversable voxels within 1m of large trees : 'large tree'
    """
    print("  Creating tree capability layers...")
    
    # Initialize individual boolean capability layers
    tree_grow = np.zeros(vtk_data.n_points, dtype=bool)
    tree_age = np.zeros(vtk_data.n_points, dtype=bool)
    tree_persist = np.zeros(vtk_data.n_points, dtype=bool)
    
    # Initialize detailed component layers
    tree_persist_near_medium = np.zeros(vtk_data.n_points, dtype=bool)
    tree_persist_near_large = np.zeros(vtk_data.n_points, dtype=bool)
    
    # Initialize string attribute layers
    tree_grow_attr = np.full(vtk_data.n_points, 'none', dtype='<U20')
    tree_age_attr = np.full(vtk_data.n_points, 'none', dtype='<U20')
    tree_persist_attr = np.full(vtk_data.n_points, 'none', dtype='<U20')
    
    # Initialize aggregate categorical layer
    capabilities_tree = np.full(vtk_data.n_points, 'none', dtype='<U20')
    
    # Tree Grow: points in 'resource_other' > 0
    if 'resource_other' in vtk_data.point_data:
        other_data = vtk_data.point_data['resource_other']
        if np.issubdtype(other_data.dtype, np.number):
            tree_grow_mask = other_data > 0
        else:
            tree_grow_mask = (other_data != 'none') & (other_data != '') & (other_data != 'nan')
        
        tree_grow |= tree_grow_mask
        tree_grow_attr[tree_grow_mask] = 'volume'
        capabilities_tree[tree_grow_mask] = 'grow'
        print(f"    Tree grow points: {np.sum(tree_grow_mask):,}")
    else:
        print("    'resource_other' not found in point data")
    
    # Tree Age: points in 'improved-tree' OR 'reserve-tree'
    if 'search_design_action' in vtk_data.point_data:
        design_action = vtk_data.point_data['search_design_action']
        tree_age_mask = design_action == 'improved-tree'
        
        tree_age |= tree_age_mask
        tree_age_attr[tree_age_mask] = 'improved tree'
        # Override any existing values (later capabilities take precedence)
        capabilities_tree[tree_age_mask] = 'age'
        print(f"    Tree age points from improved-tree: {np.sum(tree_age_mask):,}")
    else:
        print("    'search_design_action' not found in point data")
    
    # Add the second condition: forest_control == 'reserve-tree'
    if 'forest_control' in vtk_data.point_data:
        forest_control = vtk_data.point_data['forest_control']
        reserve_tree_mask = forest_control == 'reserve-tree'
        
        tree_age |= reserve_tree_mask
        tree_age_attr[reserve_tree_mask] = 'reserve tree'
        # Override any existing values (later capabilities take precedence)
        capabilities_tree[reserve_tree_mask] = 'age'
        print(f"    Tree age points from reserve-tree: {np.sum(reserve_tree_mask):,}")
    else:
        print("    'forest_control' not found in point data")
    
    # Tree Persist: points in 'search_bioavailable' == 'traversable' WITHIN 1m of points where forest_size == 'medium' OR forest_size == 'large'
    # Use KDTree for spatial search
    tree_persist_mask = np.zeros(vtk_data.n_points, dtype=bool)
    
    if 'search_bioavailable' in vtk_data.point_data and 'forest_size' in vtk_data.point_data:
        bioavailable = vtk_data.point_data['search_bioavailable']
        forest_size = vtk_data.point_data['forest_size']
        
        # Find traversable points
        traversable_mask = bioavailable == 'traversable'
        traversable_points = np.where(traversable_mask)[0]
        
        # Find tree reproductive points (medium or large trees)
        if forest_size.dtype.kind == 'U' or forest_size.dtype.kind == 'S':  # String types
            # Process medium trees
            medium_tree_mask = forest_size == 'medium'
            medium_tree_points = np.where(medium_tree_mask)[0]
            
            if len(medium_tree_points) > 0 and len(traversable_points) > 0:
                print(f"    Found {len(medium_tree_points):,} medium tree points and {len(traversable_points):,} traversable points")
                
                # Get coordinates
                medium_tree_coords = vtk_data.points[medium_tree_points]
                
                # Convert to 2D by dropping Z coordinate
                medium_tree_coords_2d = medium_tree_coords[:, 0:2]
                
                # Remove duplicates
                unique_medium_tree_coords_2d = np.unique(medium_tree_coords_2d, axis=0)
                print(f"    After removing duplicates: {len(unique_medium_tree_coords_2d):,} unique medium tree locations")
                
                # Get traversable coordinates
                traversable_coords = vtk_data.points[traversable_points]
                traversable_coords_2d = traversable_coords[:, 0:2]
                
                # Build KDTree for unique medium tree points (2D)
                medium_tree = cKDTree(unique_medium_tree_coords_2d)
                
                # Query the KDTree for points within 1m
                distances, _ = medium_tree.query(traversable_coords_2d, k=1, distance_upper_bound=1.0)
                
                # Points with finite distance are within 1m of a medium tree
                within_1m_medium = np.isfinite(distances)
                
                # Mark these points in the tree_persist_near_medium mask
                tree_persist_near_medium[traversable_points[within_1m_medium]] = True
                tree_persist_mask[traversable_points[within_1m_medium]] = True
                # Set attribute for these points
                tree_persist_attr[traversable_points[within_1m_medium]] = 'medium tree'
                
                print(f"    Tree persist points near medium trees: {np.sum(tree_persist_near_medium):,}")
            
            # Process large trees
            large_tree_mask = forest_size == 'large'
            large_tree_points = np.where(large_tree_mask)[0]
            
            if len(large_tree_points) > 0 and len(traversable_points) > 0:
                print(f"    Found {len(large_tree_points):,} large tree points and {len(traversable_points):,} traversable points")
                
                # Get coordinates
                large_tree_coords = vtk_data.points[large_tree_points]
                
                # Convert to 2D by dropping Z coordinate
                large_tree_coords_2d = large_tree_coords[:, 0:2]
                
                # Remove duplicates
                unique_large_tree_coords_2d = np.unique(large_tree_coords_2d, axis=0)
                print(f"    After removing duplicates: {len(unique_large_tree_coords_2d):,} unique large tree locations")
                
                # Get traversable coordinates
                traversable_coords = vtk_data.points[traversable_points]
                traversable_coords_2d = traversable_coords[:, 0:2]
                
                # Build KDTree for unique large tree points (2D)
                large_tree = cKDTree(unique_large_tree_coords_2d)
                
                # Query the KDTree for points within 1m
                distances, _ = large_tree.query(traversable_coords_2d, k=1, distance_upper_bound=1.0)
                
                # Points with finite distance are within 1m of a large tree
                within_1m_large = np.isfinite(distances)
                
                # Mark these points in the tree_persist_near_large mask
                tree_persist_near_large[traversable_points[within_1m_large]] = True
                tree_persist_mask[traversable_points[within_1m_large]] = True
                # Set attribute for these points (may override medium tree)
                tree_persist_attr[traversable_points[within_1m_large]] = 'large tree'
                
                print(f"    Tree persist points near large trees: {np.sum(tree_persist_near_large):,}")
            
            print(f"    Tree persist total points: {np.sum(tree_persist_mask):,}")
        else:
            print("    'forest_size' is not a string type, cannot identify medium/large trees")
    else:
        if 'search_bioavailable' not in vtk_data.point_data:
            print("    'search_bioavailable' not found in point data")
        if 'forest_size' not in vtk_data.point_data:
            print("    'forest_size' not found in point data")
    
    tree_persist |= tree_persist_mask
    # Override any existing values (later capabilities take precedence)
    capabilities_tree[tree_persist_mask] = 'persist'
    
    # Add tree capability layers to vtk_data
    vtk_data.point_data['capabilities-tree-grow'] = tree_grow
    vtk_data.point_data['capabilities-tree-age'] = tree_age
    vtk_data.point_data['capabilities-tree-persist'] = tree_persist
    
    # Add detailed component layers
    vtk_data.point_data['capabilities-tree-persist-near-medium'] = tree_persist_near_medium
    vtk_data.point_data['capabilities-tree-persist-near-large'] = tree_persist_near_large
    
    # Add string attribute layers
    vtk_data.point_data['capabilities-tree-grow-attr'] = tree_grow_attr
    vtk_data.point_data['capabilities-tree-age-attr'] = tree_age_attr
    vtk_data.point_data['capabilities-tree-persist-attr'] = tree_persist_attr
    
    vtk_data.point_data['capabilities-tree'] = capabilities_tree
    
    return vtk_data

def create_capability_layers(vtk_data):
    """Create all capability layers for birds, reptiles, and trees"""
    print("Creating capability layers...")
    
    # Create bird capabilities
    vtk_data = create_bird_capabilities(vtk_data)
    
    # Create reptile capabilities
    vtk_data = create_reptile_capabilities(vtk_data)
    
    # Create tree capabilities
    vtk_data = create_tree_capabilities(vtk_data)
    
    # Print summary of all capability layers
    print("\nSummary of all capability layers:")
    capability_keys = [key for key in vtk_data.point_data.keys() if key.startswith('capabilities-')]
    
    # Check if we have any capability layers
    if not capability_keys:
        print("  WARNING: No capability layers were created!")
        
        # Debug: Check if the required resource layers exist
        resource_keys = [key for key in vtk_data.point_data.keys() if key.startswith('resource_')]
        print("\nAvailable resource layers:")
        for key in sorted(resource_keys):
            data = vtk_data.point_data[key]
            if np.issubdtype(data.dtype, np.number):
                non_zero_count = np.sum(data > 0)
                print(f"  {key}: {non_zero_count:,} non-zero values")
            else:
                non_none_count = np.sum(data != 'none')
                print(f"  {key}: {non_none_count:,} non-'none' values")
        
        # Debug: Check if search variables exist
        search_keys = [key for key in vtk_data.point_data.keys() if key.startswith('search_')]
        print("\nAvailable search variables:")
        for key in sorted(search_keys):
            data = vtk_data.point_data[key]
            if np.issubdtype(data.dtype, np.number):
                non_zero_count = np.sum(data > 0)
                print(f"  {key}: {non_zero_count:,} non-zero values")
            else:
                non_none_count = np.sum(data != 'none')
                print(f"  {key}: {non_none_count:,} non-'none' values")
    
    # Print summary of each capability layer
    for key in sorted(capability_keys):
        data = vtk_data.point_data[key]
        if np.issubdtype(data.dtype, np.bool_):
            # Boolean array
            true_count = np.sum(data)
            print(f"  {key}: {true_count:,} True values")
        else:
            # String array
            non_none_count = np.sum(data != 'none')
            print(f"  {key}: {non_none_count:,} non-'none' values")
            
            # Print counts for each unique value
            values, counts = np.unique(data[data != 'none'], return_counts=True)
            for value, count in zip(values, counts):
                print(f"    {value}: {count:,}")
    
    return vtk_data

def count_capabilities(vtk_data):
    """Count capabilities for each persona and capability type"""
    print("  Counting capabilities...")
    
    # Dictionary to store counts
    counts = {}
    
    # Get all capability keys
    capability_keys = [key for key in vtk_data.point_data.keys() if key.startswith('capabilities-')]
    
    # Process each capability key
    for key in capability_keys:
        # Skip aggregate layers
        if key.count('-') == 1:  # e.g., 'capabilities-bird'
            continue
            
        # Extract persona and capability from key
        parts = key.split('-')
        if len(parts) >= 3:
            persona = parts[1]  # e.g., 'bird'
            capability = parts[2]  # e.g., 'socialise'
            
            # Check if this is a detailed component layer
            if len(parts) > 3:
                # This is a detailed component (e.g., 'capabilities-reptile-forage-traversable')
                component = '-'.join(parts[3:])  # e.g., 'traversable'
                
                # Count boolean component layers
                if np.issubdtype(vtk_data.point_data[key].dtype, np.bool_):
                    count = np.sum(vtk_data.point_data[key])
                    # Store with the exact key name for easier lookup
                    counts[key] = count
                    print(f"    {persona.title()} {capability} {component}: {count:,}")
                    
                    # Warning for zero counts
                    if count == 0:
                        print(f"    WARNING: Zero count for {persona.title()} {capability} {component}")
            else:
                # This is a capability layer (e.g., 'capabilities-bird-socialise')
                # Count boolean capability layers
                if np.issubdtype(vtk_data.point_data[key].dtype, np.bool_):
                    count = np.sum(vtk_data.point_data[key])
                    # Store with the exact key name for easier lookup
                    counts[key] = count
                    print(f"    {persona.title()} {capability}: {count:,}")
                    
                    # Warning for zero counts
                    if count == 0:
                        print(f"    WARNING: Zero count for {persona.title()} {capability}")
                
                # Count categorical capability layers
                elif vtk_data.point_data[key].dtype.kind in ('U', 'S'):  # String types
                    values, value_counts = np.unique(vtk_data.point_data[key], return_counts=True)
                    for value, value_count in zip(values, value_counts):
                        if value != 'none':
                            # Store with the exact key name and value for easier lookup
                            counts[f"{key}_{value}"] = value_count
                            print(f"    {persona.title()} {capability} {value}: {value_count:,}")
                            
                            # Warning for zero counts
                            if value_count == 0:
                                print(f"    WARNING: Zero count for {persona.title()} {capability} {value}")
    
    return counts

def create_capability_plots(capability_df, site, scenario):
    """Save capability statistics as CSV files (no plots)"""
    print("\nSaving capability CSV files...")
    
    # Create stats directory for R project
    stats_dir = Path('data/revised/final/stats/arboreal-future-stats')
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for ggplot
    plot_df = prepare_for_ggplot(capability_df)
    
    # Save the prepared dataframe for ggplot to the R project directory
    stats_ggplot_csv_path = stats_dir / f'{site}_{scenario}_capabilities_for_ggplot.csv'
    plot_df.to_csv(stats_ggplot_csv_path, index=False)
    print(f"  Prepared data for ggplot saved to: {stats_ggplot_csv_path}")
    
    print("  CSV files saved successfully.")
    
    return capability_df

def process_all_capabilities(site='trimmed-parade', years=None, scenario='positive', voxel_size=1, include_baseline=True):
    """Process capabilities for all years in a scenario"""
    if years is None:
        years = [0, 10, 30, 60, 180]
    
    # Initialize dictionary to store statistics
    stats = {}
    
    # Process baseline if requested
    if include_baseline:
        print("\nProcessing baseline capabilities...")
        baseline_path = f'data/revised/final/{site}/{site}_baseline_resources_{voxel_size}.vtk'
        try:
            baseline_vtk = pv.read(baseline_path)
            print(f"Loaded baseline VTK from {baseline_path}")
            
            # Create capabilities for baseline
            baseline_vtk = create_bird_capabilities(baseline_vtk)
            baseline_vtk = create_reptile_capabilities(baseline_vtk)
            baseline_vtk = create_tree_capabilities(baseline_vtk)
            
            # Count capabilities
            baseline_counts = count_capabilities(baseline_vtk)
            
            # Store in stats with 'baseline' key
            stats['baseline'] = baseline_counts
            
            # Save updated baseline with capabilities
            baseline_output_path = f'data/revised/final/{site}/{site}_baseline_resources_{voxel_size}_with_capabilities.vtk'
            baseline_vtk.save(baseline_output_path)
            print(f"Saved baseline with capabilities to {baseline_output_path}")
        except Exception as e:
            print(f"Error processing baseline: {e}")
    
    # Process each year
    for year in years:
        print(f"\nProcessing year {year}...")
        
        # Load VTK file
        vtk_data = load_vtk_file(site, scenario, voxel_size, year)
        if vtk_data is None:
            continue
        
        # Create capabilities
        vtk_data = create_bird_capabilities(vtk_data)
        vtk_data = create_reptile_capabilities(vtk_data)
        vtk_data = create_tree_capabilities(vtk_data)
        
        # Count capabilities
        counts = count_capabilities(vtk_data)
        
        # Store in stats
        stats[year] = counts
        
        # Save updated VTK file
        output_path = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_with_capabilities.vtk'
        vtk_data.save(output_path)
        print(f"Saved updated VTK file to {output_path}")
    
    # Convert stats to DataFrame
    if stats:
        stats_df = pd.DataFrame(stats)
        print("\nCapability statistics:")
        print(stats_df)
        
        # Save statistics to CSV
        stats_dir = Path('data/revised/final/stats')
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_path = stats_dir / f'{site}_{scenario}_capabilities.csv'
        stats_df.to_csv(stats_path)
        print(f"Saved capability statistics to {stats_path}")
        
        return stats_df
    else:
        print("No statistics were collected.")
        return None

def generate_capability_dataframe(stats_df, scenario):
    """Generate a dataframe tracking capabilities across timesteps"""
    print("\nGenerating capability dataframe...")
    
    # Print column names to help debug
    print("  Available columns in stats dataframe:")
    for col in sorted(stats_df.columns):
        print(f"    {col}")
    
    # Define the expected capabilities and their column names in the stats dataframe
    expected_indicators = {
        'Bird': {
            'Socialise': [('perch branch', 'capabilities-bird-socialise')],
            'Feed': [('peeling bark', 'capabilities-bird-feed')],
            'Raise Young': [('hollow', 'capabilities-bird-raise-young')]
        },
        'Reptile': {
            'Traverse': [('traversable', 'capabilities-reptile-traverse_traversable')],
            'Forage': [
                ('ground cover', 'capabilities-reptile-forage-low-veg'),
                ('dead branch', 'capabilities-reptile-forage-dead-branch'),
                ('epiphyte', 'capabilities-reptile-forage-epiphyte')
            ],
            'Shelter': [
                ('fallen log', 'capabilities-reptile-shelter-fallen-log'),
                ('fallen tree', 'capabilities-reptile-shelter-fallen-tree')
            ]
        },
        'Tree': {
            'Grow': [('volume', 'capabilities-tree-grow')],
            'Age': [
                ('improved tree', 'capabilities-tree-age-improved-tree'),
                ('reserve tree', 'capabilities-tree-age-reserve-tree')
            ],
            'Persist': [
                ('medium tree', 'capabilities-tree-persist-near-medium'),
                ('large tree', 'capabilities-tree-persist-near-large')
            ]
        }
    }
    
    # Create a list to store rows for the dataframe
    rows = []
    
    # Get all years (timesteps) from the stats dataframe, including baseline if present
    years = sorted([col for col in stats_df.columns if col != 'baseline'])
    if 'baseline' in stats_df.columns:
        years = ['baseline'] + years
    
    # For each persona, capability, and numeric indicator, extract counts for all years
    for persona, capabilities in expected_indicators.items():
        # Assign a capability number (0, 1, 2) to each capability for this persona
        capability_numbers = {capability: i for i, capability in enumerate(capabilities.keys())}
        
        for capability, indicators in capabilities.items():
            for indicator_name, column_name in indicators:
                # Create a row with persona, capability, capability number, numeric indicator, and scenario
                row = {
                    'Persona': persona,
                    'Capability': capability,
                    'CapabilityNo': capability_numbers[capability],
                    'NumericIndicator': indicator_name,
                    'Scenario': scenario,
                    'hpos': 0,  # Default value
                    'is_dummy': False  # Default value
                }
                
                # Add counts for each year
                for year in years:
                    if year in stats_df.columns:
                        if column_name in stats_df.index:
                            row[str(year)] = stats_df.at[column_name, year]
                        else:
                            # Try alternative column names
                            alt_names = [
                                column_name,
                                column_name.replace('-', '_'),
                                f"{column_name}_bool"
                            ]
                            
                            found = False
                            for alt_name in alt_names:
                                if alt_name in stats_df.index:
                                    row[str(year)] = stats_df.at[alt_name, year]
                                    found = True
                                    break
                            
                            if not found:
                                print(f"  Warning: {column_name} not found in stats dataframe")
                                row[str(year)] = 0
                    else:
                        row[str(year)] = 0
                
                # Add to rows
                rows.append(row)
    
    # Create dataframe
    capability_df = pd.DataFrame(rows)
    
    # Print the dataframe
    print("\nCapability dataframe:")
    print(capability_df.head())
    print(f"Total rows: {len(capability_df)}")
    
    return capability_df

def check_zero_capabilities(site, scenario, voxel_size, years):
    """Check for zero capability counts in VTK files"""
    print("\nChecking for zero capability counts in VTK files...")
    
    # Dictionary to track zero counts
    zero_counts = {}
    
    for year in years:
        print(f"\nChecking year {year}...")
        
        # Load VTK file
        vtk_path = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_with_capabilities.vtk'
        try:
            vtk_data = pv.read(vtk_path)
        except Exception as e:
            print(f"  Error loading VTK file for year {year}: {e}")
            continue
        
        # Get all capability keys
        capability_keys = [key for key in vtk_data.point_data.keys() if key.startswith('capabilities-')]
        
        # Check each capability key
        for key in sorted(capability_keys):
            data = vtk_data.point_data[key]
            
            if np.issubdtype(data.dtype, np.bool_):
                # Boolean array
                count = np.sum(data)
                if count == 0:
                    print(f"  WARNING: Zero count for {key} in year {year}")
                    zero_counts.setdefault(year, []).append(key)
            elif isinstance(data[0], (str, np.str_)):
                # String array
                non_none_count = np.sum(data != 'none')
                if non_none_count == 0:
                    print(f"  WARNING: Zero non-'none' values for {key} in year {year}")
                    zero_counts.setdefault(year, []).append(key)
                
                # Check each unique value
                values, counts = np.unique(data[data != 'none'], return_counts=True)
                for value, count in zip(values, counts):
                    if count == 0:
                        print(f"  WARNING: Zero count for {key} value '{value}' in year {year}")
                        zero_counts.setdefault(year, []).append(f"{key}_{value}")
    
    # Print summary of zero counts
    if zero_counts:
        print("\nSummary of zero capability counts:")
        for year, keys in sorted(zero_counts.items()):
            print(f"  Year {year}: {len(keys)} zero counts")
            for key in sorted(keys):
                print(f"    {key}")
    else:
        print("\nNo zero capability counts found in any year.")
    
    return zero_counts

def prepare_for_ggplot(capability_df):
    """Prepare capability dataframe for ggplot by adding hpos and dummy data"""
    print("\nPreparing capability dataframe for ggplot...")
    
    # Create a copy of the dataframe
    plot_df = capability_df.copy()
    
    # Group by persona and count numeric indicators
    persona_counts = plot_df.groupby('Persona')['NumericIndicator'].nunique()
    print(f"  Numeric indicators per persona: {persona_counts.to_dict()}")
    
    # Find the persona with the most numeric indicators
    max_persona = persona_counts.idxmax()
    max_count = persona_counts.max()
    print(f"  Persona with most indicators: {max_persona} ({max_count})")
    
    # Create a mapping of NumericIndicator to hpos for each persona
    hpos_mapping = {}
    for persona in plot_df['Persona'].unique():
        indicators = plot_df[plot_df['Persona'] == persona]['NumericIndicator'].unique()
        hpos_mapping[persona] = {indicator: i for i, indicator in enumerate(sorted(indicators))}
    
    # Add hpos column based on the mapping
    plot_df['hpos'] = plot_df.apply(
        lambda row: hpos_mapping[row['Persona']][row['NumericIndicator']], 
        axis=1
    )
    
    # Create dummy data for personas with fewer indicators
    dummy_rows = []
    
    # Dummy data values
    dummy_values = [0, 5, 10, 20, 30, 50]
    
    # Get all scenarios
    scenarios = plot_df['Scenario'].unique()
    
    # Get all years (columns that are numeric)
    year_columns = [col for col in plot_df.columns if col.isdigit()]
    
    # For each persona, add dummy data to reach max_count
    for persona in plot_df['Persona'].unique():
        current_count = persona_counts[persona]
        if current_count < max_count:
            # Need to add dummy data
            for hpos in range(current_count, max_count):
                for scenario in scenarios:
                    # Create a dummy row
                    dummy_row = {
                        'Persona': persona,
                        'Capability': f'Dummy{hpos}',
                        'CapabilityNo': 99,  # High number to distinguish dummy data
                        'NumericIndicator': f'default-{hpos}',
                        'Scenario': scenario,
                        'hpos': hpos
                    }
                    
                    # Add dummy values for each year
                    for i, year in enumerate(year_columns):
                        dummy_row[year] = dummy_values[i % len(dummy_values)]
                    
                    dummy_rows.append(dummy_row)
    
    # Add dummy rows to the dataframe
    if dummy_rows:
        dummy_df = pd.DataFrame(dummy_rows)
        plot_df = pd.concat([plot_df, dummy_df], ignore_index=True)
        print(f"  Added {len(dummy_rows)} dummy rows for balanced facet grid")
    
    # Add a column to identify dummy data
    plot_df['is_dummy'] = plot_df['CapabilityNo'] == 99
    
    # Print summary of the prepared dataframe
    print(f"  Final dataframe shape: {plot_df.shape}")
    print(f"  Personas: {plot_df['Persona'].nunique()}")
    print(f"  Scenarios: {plot_df['Scenario'].nunique()}")
    print(f"  Numeric indicators: {plot_df['NumericIndicator'].nunique()}")
    print(f"  hpos values: {sorted(plot_df['hpos'].unique())}")
    
    return plot_df

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process capabilities for scenarios')
    parser.add_argument('--sites', nargs='+', default=['trimmed-parade'], 
                        help='Sites to process (default: [trimmed-parade])')
    parser.add_argument('--years', nargs='+', type=int, help='Years to process (default: [0, 10, 30, 60, 180])')
    parser.add_argument('--scenarios', nargs='+', default=['positive', 'trending'], 
                        help='Scenarios to process (default: [positive, trending])')
    parser.add_argument('--voxel-size', type=int, default=1, help='Voxel size (default: 1)')
    
    args = parser.parse_args()
    
    # Process all sites and scenarios
    for site in args.sites:
        print(f"\n=== Processing site: {site} ===")
        
        all_capability_dfs = []
        
        for scenario in args.scenarios:
            print(f"\nProcessing scenario: {scenario}")
            
            # Process capabilities for all years in this scenario
            stats_df = process_all_capabilities(
                site=site,
                years=args.years,
                scenario=scenario,
                voxel_size=args.voxel_size
            )
            
            if stats_df is not None:
                # Check for zero capability counts in VTK files
                years = args.years if args.years else [0, 10, 30, 60, 180]
                check_zero_capabilities(site, scenario, args.voxel_size, years)
                
                # Generate capability dataframe with scenario column
                capability_df = generate_capability_dataframe(stats_df, scenario)
                
                # Create capability plots and save data for this scenario
                create_capability_plots(capability_df, site, scenario)
                
                # Add to list of all dataframes
                all_capability_dfs.append(capability_df)
            else:
                print(f"\nNo capability statistics were processed for site {site}, scenario {scenario}.")
        
        # Combine all scenario dataframes if we have more than one
        if len(all_capability_dfs) > 1:
            combined_df = pd.concat(all_capability_dfs, ignore_index=True)
            
            # Create stats directory for R project
            stats_dir = Path('data/revised/final/stats/arboreal-future-stats')
            stats_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare combined data for ggplot
            combined_plot_df = prepare_for_ggplot(combined_df)
            
            # Save ggplot-ready combined dataframe to the R project directory
            stats_combined_ggplot_path = stats_dir / f'{site}_all_scenarios_capabilities_for_ggplot.csv'
            combined_plot_df.to_csv(stats_combined_ggplot_path, index=False)
            print(f"Combined ggplot data saved to: {stats_combined_ggplot_path}")

if __name__ == "__main__":
    main() 