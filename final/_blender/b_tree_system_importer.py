import bpy
import pandas as pd
import numpy as np
import os
import glob

SITE = 'trimmed-parade'
YEAR = 60
#POSITION_OBJECT = "Camera"  # Object to measure distance from
POSITION_OBJECT = "WorldCam"
DISTANCE_UNITS = 100.0     # Distance for culling

# Derived paths
BASE_PATH = r"D:\Data 2023 Volumetric Scenarios\ply"
CSV_FILENAME = f'{SITE}_1_treeDF_{YEAR}.csv'
CSV_FILEPATH = os.path.join(BASE_PATH, SITE, 'scenarios', CSV_FILENAME)
PLY_FOLDER = os.path.join(BASE_PATH, 'ply')  # Removed extra 'ply' folder

#MAC
PLY_FOLDER = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/treeMeshesPly'
BASE_PATH = f'/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/{SITE}'
CSV_FILEPATH = os.path.join(BASE_PATH, CSV_FILENAME)


def convert_control(value):
    control_map = {
        'street-tree': 0,
        'park-tree': 1,
        'reserve-tree': 2,
        'improved-tree': 3
    }
    # Vectorized mapping using pandas
    return pd.Series(value).str.lower().map(control_map).fillna(-1)

def convert_size(value):
    size_map = {
        'small': 0,
        'medium': 1,
        'large': 2,
        'senescing': 3,
        'snag': 4,
        'fallen': 5
    }
    # Vectorized mapping using pandas
    return pd.Series(value).str.lower().map(size_map).fillna(-1)

def parse_ply_filenames(filenames):
    """Vectorized parsing of multiple PLY filenames"""
    # Split filenames into components using pandas string operations
    df = pd.DataFrame({'filename': filenames})
    
    # Remove .ply and split into components
    parts = df['filename'].str[:-4].str.split('_', expand=True)
    
    # Extract components using vectorized operations
    try:
        result = pd.DataFrame({
            'precolonial': parts[0].str.split('.').str[1].str.lower(),
            'size': parts[1].str.split('.').str[1].str.lower(),
            'control': parts[2].str.split('.').str[1].str.lower(),
            'id': parts[3].str.split('.').str[1].astype(int),
            'filename': df['filename']
        })
        return result
    except Exception as e:
        print(f"Warning: Error parsing filenames: {str(e)}")
        return pd.DataFrame()

def create_instance_system(distance_units=DISTANCE_UNITS, camera_object_name=POSITION_OBJECT):
    print("\nStarting enhanced tree instance system creation...")
    
    # Setup collections hierarchy
    year_collection_name = f"Year_{YEAR}"
    year_collection = bpy.data.collections.get(year_collection_name)
    
    # Create or get year collection
    if not year_collection:
        year_collection = bpy.data.collections.new(year_collection_name)
        bpy.context.scene.collection.children.link(year_collection)
    
    # Clean up existing Trees collection within year collection
    existing_tree_collection = year_collection.children.get("Trees")
    if existing_tree_collection:
        print("Cleaning up existing Trees collection...")
        for obj in existing_tree_collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(existing_tree_collection)
    
    # Remove existing TreePositions object if it exists
    existing_positions = year_collection.objects.get("TreePositions")
    if existing_positions:
        bpy.data.objects.remove(existing_positions, do_unlink=True)
        
    print("Cleanup complete")

    # Get camera position
    camera_obj = bpy.data.objects.get(camera_object_name)
    if not camera_obj:
        raise ValueError(f"Camera object '{camera_object_name}' not found in scene")
    
    camera_x = camera_obj.location.x
    camera_y = camera_obj.location.y
    
    print(f"Camera position: x={camera_x:.2f}, y={camera_y:.2f}")
    
    # 1. Read CSV and filter by simple square distance
    print("\n1. Reading CSV and filtering by distance...")
    df = pd.read_csv(CSV_FILEPATH)
    total_trees = len(df)
    
    # Vectorized distance calculation
    df['distance_to_camera'] = np.sqrt(
        np.square(df['x'] - camera_x) + 
        np.square(df['y'] - camera_y)
    )
    
    # Vectorized filtering
    mask = (
        (df['x'].between(camera_x - distance_units, camera_x + distance_units)) & 
        (df['y'].between(camera_y - distance_units, camera_y + distance_units))
    )
    df_filtered = df[mask]
    
    trees_culled = total_trees - len(df_filtered)
    print(f"Total trees in CSV: {total_trees}")
    print(f"Trees within ±{distance_units} units of camera: {len(df_filtered)}")
    print(f"Trees culled: {trees_culled} ({(trees_culled/total_trees*100):.1f}%)")
    
    if len(df_filtered) == 0:
        raise ValueError("No trees within distance threshold!")

    # 2. Create filenames using vectorized string operations
    print("\n2. Creating filenames...")
    df_filtered['filename'] = (
        'precolonial.' + df_filtered['precolonial'].astype(str).str.lower() + 
        '_size.' + df_filtered['size'].astype(str).str.lower() + 
        '_control.' + df_filtered['control'].astype(str).str.lower() + 
        '_id.' + df_filtered['tree_id'].astype(str) + 
        '.ply'
    )
    
    # 3. Find unique filenames from filtered data
    unique_filenames = df_filtered['filename'].unique()
    print(f"\n3. Found {len(unique_filenames)} unique tree type combinations needed:")
    for i, filename in enumerate(sorted(unique_filenames), 1):
        print(f"{i}. {filename}")

    # 4. Scan and parse PLY files using vectorized operations
    print("\n4. Scanning and parsing PLY files...")
    available_plys = pd.Series([f for f in os.listdir(PLY_FOLDER) if f.endswith('.ply')])
    available_templates = parse_ply_filenames(available_plys)
    
    print(f"Successfully parsed {len(available_templates)} PLY files")
    
    # 5. Create fallback mapping using vectorized operations
    print("\n5. Creating fallback mapping...")
    needed_templates = parse_ply_filenames(unique_filenames)
    fallback_map = {}
    
    # Vectorized matching operations
    for idx, needed in needed_templates.iterrows():
        # Create masks for different matching criteria
        exact_match_mask = (
            (available_templates['precolonial'] == needed['precolonial']) &
            (available_templates['size'] == needed['size']) &
            (available_templates['control'] == needed['control']) &
            (available_templates['id'] == needed['id'])
        )
        
        best_match_mask = (
            (available_templates['precolonial'] == needed['precolonial']) &
            (available_templates['size'] == needed['size']) &
            (available_templates['control'] == needed['control'])
        )
        
        size_match_mask = (available_templates['size'] == needed['size'])
        
        # Find matches using masks
        if exact_match_mask.any():
            match = available_templates[exact_match_mask].iloc[0]
            fallback_map[needed['filename']] = match['filename']
            print(f"Exact match found for {needed['filename']}")
        elif best_match_mask.any():
            match = available_templates[best_match_mask].iloc[0]
            fallback_map[needed['filename']] = match['filename']
            print(f"Size/control match found for {needed['filename']} -> using {match['filename']}")
        elif size_match_mask.any():
            match = available_templates[size_match_mask].iloc[0]
            fallback_map[needed['filename']] = match['filename']
            print(f"Size match found for {needed['filename']} -> using {match['filename']}")
        else:
            fallback_map[needed['filename']] = available_templates.iloc[0]['filename']
            print(f"No good match for {needed['filename']} -> using {available_templates.iloc[0]['filename']}")
    
    print("\n6. Final fallback mapping:")
    for orig, fallback in fallback_map.items():
        print(f"{orig} -> {fallback}")

    # Create a new collection for the trees as child of year collection
    print("\n7. Creating new Trees collection...")
    tree_collection = bpy.data.collections.new("Trees")
    year_collection.children.link(tree_collection)
    
    # 8. Import PLY files using fallback mapping
    print("\n8. Importing PLY files with fallbacks...")
    tree_objects = {}
    
    # Create instance mapping using pandas
    instance_map = pd.Series(range(len(unique_filenames)), index=sorted(unique_filenames))
    df_filtered['instanceID'] = df_filtered['filename'].map(instance_map)
    
    # This section cannot be vectorized due to Blender's API requirements
    for orig_filename, instance_id in instance_map.items():
        actual_filename = fallback_map.get(orig_filename, orig_filename)
        filepath = os.path.join(PLY_FOLDER, actual_filename)
        
        if os.path.exists(filepath):
            bpy.ops.wm.ply_import(filepath=filepath)
            tree = bpy.context.active_object
            tree.name = orig_filename[:-4]
            tree_objects[instance_id] = tree
            
            if tree.users_collection:
                for collection in tree.users_collection:
                    collection.objects.unlink(tree)
            tree_collection.objects.link(tree)
            
            print(f"Imported {actual_filename} as instance {instance_id} for {orig_filename}")
        else:
            print(f"Warning: PLY file not found: {filepath}")
            
    print("\nChecking Trees collection order:")
    for i, obj in enumerate(tree_collection.objects):
        print(f"Index {i}: {obj.name}")
    
    # 9. Create point cloud using numpy
    print("\n9. Creating point cloud...")
    points = df_filtered[['x', 'y', 'z']].to_numpy()
    
    mesh = bpy.data.meshes.new("TreePositions")
    mesh.from_pydata(points.tolist(), [], [])
    mesh.update()
    
    point_cloud = bpy.data.objects.new("TreePositions", mesh)
    year_collection.objects.link(point_cloud)  # Link to year collection instead of scene
    
    # 10. Add attributes using vectorized operations
    print("10. Adding attributes...")
    attr_types = {
        'rotation': ('FLOAT', 'value', df_filtered['rotateZ'].to_numpy()),
        'tree_type': ('INT', 'value', df_filtered['tree_id'].to_numpy(dtype=np.int32)),
        'structure_id': ('INT', 'value', df_filtered['structureID'].to_numpy(dtype=np.int32)),
        'size': ('INT', 'value', convert_size(df_filtered['size']).to_numpy(dtype=np.int32)),
        'control': ('INT', 'value', convert_control(df_filtered['control']).to_numpy(dtype=np.int32)),
        #'precolonial': ('BOOLEAN', 'value', df_filtered['precolonial'].astype(str).str.lower().eq('true').to_numpy()),
        'life_expectancy': ('INT', 'value', df_filtered['useful_life_expectancy'].to_numpy(dtype=np.int32)),
        'instanceID': ('INT', 'value', df_filtered['instanceID'].to_numpy(dtype=np.int32))
    }
    
    for attr_name, (attr_type, value_type, data) in attr_types.items():
        attr = mesh.attributes.new(name=attr_name, type=attr_type, domain='POINT')
        attr.data.foreach_set(value_type, data)
        
        if hasattr(attr, 'is_runtime_only'):
            attr.is_runtime_only = False
    
    print("\nChecking instance mapping:")
    print("Sample of points and their instanceIDs:")
    print(df_filtered[['filename', 'instanceID']].head(10))
    print("\nUnique instanceIDs:", sorted(df_filtered['instanceID'].unique()))
    print("Number of unique instanceIDs:", len(df_filtered['instanceID'].unique()))
    print("Number of trees in collection:", len(tree_collection.objects))
        
    # Select point cloud
    bpy.ops.object.select_all(action='DESELECT')
    point_cloud.select_set(True)
    bpy.context.view_layer.objects.active = point_cloud
    
    print(f"\nSuccessfully created system with {len(tree_collection.objects)} tree instances!")
    return point_cloud, tree_objects

# Run setup
try:
    point_cloud, trees = create_instance_system()
    print(f"\nSuccessfully created system with {len(trees)} tree instances!")
except Exception as e:
    print(f"\nError: {str(e)}")