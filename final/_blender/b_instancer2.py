import bpy
import pandas as pd
import numpy as np
import os
import glob

# Constants
#SITE = 'uni'
SITE = 'trimmed-parade'
SCENARIO = 'positive'
YEAR = 10
PASS_INDEX = 11
DISTANCE_UNITS = 200
POLE_BYPASS = True
POLE_FALLBACK_PLY = 'artificial_precolonial.False_size.snag_control.improved-tree_id.10.ply'
# Paths
PLY_FOLDER = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/treeMeshesPly'
LOG_FOLDER = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/logMeshesPly'
BASE_PATH = f'/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/{SITE}'

if SITE == 'trimmed-parade':
    BASE_PATH = f'/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/{SITE}/initial'


CSV_FILENAME = f'{SITE}_{SCENARIO}_1_nodeDF_{YEAR}.csv'
CSV_FILEPATH = os.path.join(BASE_PATH, CSV_FILENAME)

print(f'CSV_FILEPATH IS: {CSV_FILEPATH}')

#Helper Functions
def cleanup_scene():
    """Clean up any existing data to ensure fresh start"""
    # Clear all collections first
    for collection in bpy.data.collections:
        if collection.name.startswith(f"Year_{YEAR}"):
            for obj in collection.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.collections.remove(collection)
    
    # Clear any orphaned data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

def convert_control(value):
    control_map = {
        'street-tree': 0,
        'park-tree': 1,
        'reserve-tree': 2,
        'improved-tree': 3
    }
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
    return pd.Series(value).str.lower().map(size_map).fillna(-1)

def find_camera_by_pass_index(pass_index):
    """Find the first object with the specified pass index"""
    for obj in bpy.data.objects:
        if obj.pass_index == pass_index:
            return obj
    raise ValueError(f"No object found with pass index {pass_index}")

def parse_ply_filenames(filenames, node_type='tree'):
    """Vectorized parsing of multiple PLY filenames based on node type"""
    df = pd.DataFrame({'filename': filenames})
    
    try:
        if node_type in ['tree', 'pole']:
            # Split by underscore first
            parts = df['filename'].str[:-4].str.split('_', expand=True)
            
            # Extract components maintaining original case
            result = pd.DataFrame({
                'precolonial': parts[0].str.split('.').str[1],
                'size': parts[1].str.split('.').str[1],
                'control': parts[2].str.split('.').str[1],
                'id': parts[3].str.split('.').str[1].astype(int),  # id.7 -> 7
                'filename': df['filename']
            })
            print("\nDEBUG - ID parts:")
            print(parts[3])  # Let's see what we're working with
            
        else:  # log
            parts = df['filename'].str[:-4].str.split('.', expand=True)
            result = pd.DataFrame({
                'size': parts[1],
                'id': parts[3].astype(int),
                'filename': df['filename']
            })
        return result
    except Exception as e:
        print(f"Warning: Error parsing filenames: {str(e)}")
        print("Parts shape:", parts.shape)
        print("Column 3 (should be IDs):", parts[3].head())
        return pd.DataFrame()
    
    
#Process Collection Function
def process_collection(df, ply_folder, node_type, year_collection):
    """Main function to process a collection of nodes"""
    print(f"\nStarting process_collection for {node_type}")
    print(f"Number of {node_type}s to process: {len(df)}")
    print(f"Using PLY folder: {ply_folder}")
    
    if len(df) == 0:
        print(f"No {node_type}s to process in dataframe")
        return None, {}
        
    print(f"\nProcessing {node_type} collection...")

################ SECTION A - INITIAL SETUP AND COLLECTION MANAGEMENT ################
    # Setup collections with new naming convention
    positions_name = f"{node_type}_{YEAR}_{SCENARIO}_positions"
    models_name = f"{node_type}_{YEAR}_{SCENARIO}_plyModels"
    
    # Clean up existing collections
    for coll_name in [positions_name, models_name]:
        existing_collection = year_collection.children.get(coll_name)
        if existing_collection:
            print(f"Cleaning up existing {coll_name} collection...")
            for obj in existing_collection.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.collections.remove(existing_collection)
    
    # Create new collections
    positions_collection = bpy.data.collections.new(positions_name)
    models_collection = bpy.data.collections.new(models_name)
    year_collection.children.link(positions_collection)
    year_collection.children.link(models_collection)
    
    print(f"Created collections: {positions_name} and {models_name}")
    
################ SECTION B - FILENAME CREATION AND MAPPING ################
    # Create filenames
    print("Creating filenames...")
    if node_type in ['tree', 'pole']:
        df['filename'] = (
            'precolonial.' + df['precolonial'].astype(str).str.capitalize() + 
            '_size.' + df['size'].astype(str) + 
            '_control.' + df['control'].astype(str) + 
            '_id.' + df['tree_id'].astype(str) + 
            '.ply'
        )
    else:  # log
        df['filename'] = (
            'size.' + df['size'].astype(str) + 
            '.log.' + df['tree_id'].astype(str) + 
            '.ply'
        )

    # Find unique filenames
    unique_filenames = df['filename'].unique()
    print(f"\nFound {len(unique_filenames)} unique {node_type} type combinations")



################ SECTION C - FALLBACK MAPPING ################
    # Scan PLY files
    print("Scanning PLY files...")
    available_plys = pd.Series([
        f for f in os.listdir(ply_folder) 
        if f.endswith('.ply') and not f.startswith('artificial_')
    ])
    print(f"Found {len(available_plys)} usable PLY files")
    
    # Early fallback map for poles if bypass is enabled
    fallback_map = {}
    if node_type == 'pole' and POLE_BYPASS:
        print(f"\n⚡ POLE BYPASS ENABLED")
        print(f"   Using {POLE_FALLBACK_PLY} for all poles")
        print(f"   Number of unique filenames: {len(unique_filenames)}")
        print(f"   Unique filenames: {unique_filenames}")
        fallback_map = {filename: POLE_FALLBACK_PLY for filename in unique_filenames}
        print(f"   Created fallback map with {len(fallback_map)} entries")
        
        # Create empty template DataFrames to continue processing
        available_templates = pd.DataFrame(columns=['precolonial', 'size', 'control', 'id', 'filename'])
        needed_templates = pd.DataFrame(columns=['precolonial', 'size', 'control', 'id', 'filename'])
    else:
        available_templates = parse_ply_filenames(available_plys, node_type)
        needed_templates = parse_ply_filenames(unique_filenames, node_type)
        
        print("\nPreprocessing size mappings...")
        # Convert 'artificial' to 'snag' in needed templates
        if 'size' in needed_templates.columns:
            artificial_mask = needed_templates['size'] == 'artificial'
            if artificial_mask.any():
                print(f"Converting {artificial_mask.sum()} 'artificial' sizes to 'snag' for template search")
                needed_templates.loc[artificial_mask, 'size'] = 'snag'
        
        # Setup random selection
        def get_random_index(row, max_val):
            """Create a reproducible random index from node properties"""
            seed_str = f"{row.get('precolonial', '')}_{row.get('size', '')}_{row.get('control', '')}_{row.get('id', 0)}"
            return hash(seed_str) % max_val  # Use modulo to get index within range

        print("\nCreating fallback mapping...")
        fallback_map = {}
        for idx, needed in needed_templates.iterrows():
            print(f"\nProcessing template {idx + 1}/{len(needed_templates)}:")
            print(f"Searching for: precolonial={needed.get('precolonial', 'N/A')}, "
                  f"size={needed.get('size', 'N/A')}, "
                  f"control={needed.get('control', 'N/A')}, "
                  f"id={needed.get('id', 'N/A')}")
            
            if node_type in ['tree', 'pole']:
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
            else:  # log
                exact_match_mask = (
                    (available_templates['size'] == needed['size']) &
                    (available_templates['id'] == needed['id'])
                )
                best_match_mask = (available_templates['size'] == needed['size'])
                
            size_match_mask = (available_templates['size'] == needed['size'])
            
            if exact_match_mask.any():
                matches = available_templates[exact_match_mask]
                match = matches.iloc[0]  # Keep exact match if found
                fallback_map[needed['filename']] = match['filename']
                print("✓ Exact match found!")
                print(f"   Using: {match['filename']}")
                
            elif best_match_mask.any():
                matches = available_templates[best_match_mask]
                random_idx = get_random_index(needed, len(matches))
                match = matches.iloc[random_idx]
                fallback_map[needed['filename']] = match['filename']
                print("↳ No exact match, falling back to best match")
                print(f"   Found {len(matches)} matching templates")
                print(f"   Selected index {random_idx}: {match['filename']}")
                
            elif size_match_mask.any():
                matches = available_templates[size_match_mask]
                random_idx = get_random_index(needed, len(matches))
                match = matches.iloc[random_idx]
                fallback_map[needed['filename']] = match['filename']
                print("↳ No control match, falling back to size match")
                print(f"   Found {len(matches)} size-matching templates")
                print(f"   Selected index {random_idx}: {match['filename']}")
                
            else:
                random_idx = get_random_index(needed, len(available_templates))
                match = available_templates.iloc[random_idx]
                fallback_map[needed['filename']] = match['filename']
                print("⚠ No good matches found, using random template")
                print(f"   Selecting from {len(available_templates)} available templates")
                print(f"   Selected index {random_idx}: {match['filename']}")


################ SECTION D - PLY IMPORT ################
    # Create instance mapping
    instance_map = pd.Series(range(len(unique_filenames)), index=sorted(unique_filenames))
    df['instanceID'] = df['filename'].map(instance_map)
    
    # Import PLY files
    print(f"\nImporting {len(instance_map)} PLY files...")
    
    # Initialize objects dictionary
    node_objects = {}
        
    # Import all unique models
    for i, orig_filename in enumerate(sorted(unique_filenames)):
        actual_filename = fallback_map.get(orig_filename, orig_filename)
        filepath = os.path.join(ply_folder, actual_filename)
        
        if os.path.exists(filepath):
            print(f"Importing {actual_filename} as instance {i}")
            
            # First unlink any existing objects with this name
            target_name = f"instanceID.{i}_{actual_filename[:-4]}"
            for obj in bpy.data.objects:
                if obj.name.startswith(target_name):
                    if obj.users_collection:
                        for collection in obj.users_collection:
                            collection.objects.unlink(obj)
                    bpy.data.objects.remove(obj)
            
            # Import the PLY
            bpy.ops.wm.ply_import(filepath=filepath)
            
            # Find and set up the newly imported object
            imported = False
            for obj in bpy.data.objects:
                if obj.name.startswith(actual_filename[:-4]) or obj.name.startswith('artificial'):  # Added check for artificial
                    node = obj
                    node.name = f"instanceID.{i}_{actual_filename[:-4]}"
                    
                    # Ensure it's only in our models collection
                    if node.users_collection:
                        for collection in node.users_collection:
                            collection.objects.unlink(obj)
                    models_collection.objects.link(node)
                    
                    # Set pass index
                    node.pass_index = 3
                    
                    # Store reference
                    node_objects[i] = node
                    imported = True
                    break
                    
            if not imported:
                print(f"Warning: Could not find imported object for {actual_filename}")
        else:
            print(f"Warning: File not found: {filepath}")
    
    print(f"Imported {len(node_objects)} unique models")
    
    # Map instanceIDs to model indices
    df['model_index'] = df['filename'].map(lambda x: list(sorted(unique_filenames)).index(x))




################ SECTION E - POINT CLOUD AND ATTRIBUTES ################
    # Create point cloud
    print("\nCreating point cloud...")
    points = df[['x', 'y', 'z']].to_numpy()
    mesh = bpy.data.meshes.new(f"{node_type.capitalize()}Positions")
    mesh.from_pydata(points.tolist(), [], [])
    mesh.update()
    
    point_cloud = bpy.data.objects.new(f"{node_type.capitalize()}Positions", mesh)
    point_cloud.pass_index = 3
    positions_collection.objects.link(point_cloud)
    
    print(f"Created point cloud with {len(points)} locations")
    
    # Add attributes
    print("\nAdding attributes...")
    base_attr_types = {
        'rotation': ('FLOAT', 'value', df['rotateZ'].to_numpy()),
        'tree_type': ('INT', 'value', df['tree_id'].to_numpy(dtype=np.int32)),
        'structure_id': ('INT', 'value', df['structureID'].to_numpy(dtype=np.int32)),
        'size': ('INT', 'value', convert_size(df['size']).to_numpy(dtype=np.int32)),
        'instanceID': ('INT', 'value', df['model_index'].to_numpy(dtype=np.int32))
    }
    
    # Add type-specific attributes
    if node_type in ['tree', 'pole']:
        base_attr_types['control'] = ('INT', 'value', convert_control(df['control']).to_numpy(dtype=np.int32))
        base_attr_types['life_expectancy'] = ('INT', 'value', df['useful_life_expectancy'].to_numpy(dtype=np.int32))
    else:  # log
        base_attr_types['logMass'] = ('FLOAT', 'value', df['logMass'].to_numpy())
    
    for attr_name, (attr_type, value_type, data) in base_attr_types.items():
        print(f"Adding {attr_name} attribute...")
        attr = mesh.attributes.new(name=attr_name, type=attr_type, domain='POINT')
        attr.data.foreach_set(value_type, data)
        if hasattr(attr, 'is_runtime_only'):
            attr.is_runtime_only = False
    
    print(f"\nSuccessfully created {node_type} instance system!")
    print(f"- {len(node_objects)} unique meshes")
    print(f"- {len(df)} instance locations")
    print(f"- {len(base_attr_types)} attributes")

################ SECTION F - GEOMETRY NODES SETUP ################
    # Setup geometry nodes for point cloud
    print("\nSetting up geometry nodes...")
    
    template_name = "instance_template"
    if template_name in bpy.data.node_groups:
        print(f"Found template node group: {template_name}")
        
        try:
            # Create copy of template
            new_node_group = bpy.data.node_groups[template_name].copy()
            new_node_group.name = f"{node_type}_{SCENARIO}_{YEAR}"
            print(f"Created new node group: {new_node_group.name}")
            
            # Create geometry nodes modifier
            geo_nodes = point_cloud.modifiers.new(
                name=f"{node_type}_{SCENARIO}_{YEAR}", 
                type='NODES'
            )
            geo_nodes.node_group = new_node_group
            print("Applied geometry nodes modifier to point cloud")
            
            # Update Collection Info node using the exact models_collection reference
            for node in new_node_group.nodes:
                if node.type == 'COLLECTION_INFO':
                    if hasattr(node, "inputs") and "Collection" in node.inputs:
                        node.inputs["Collection"].default_value = models_collection  # Use the exact reference
                        print(f"Updated Collection Info node to use {models_collection.name}")
                        break
            
            print("Geometry nodes setup complete")
            return point_cloud, node_objects
            
        except Exception as e:
            print(f"Error in geometry nodes setup: {str(e)}")
            return None, None
            
    else:
        print(f"Warning: Could not find template node group '{template_name}'")
        return None, None  # Return None values if template not found



def main():
    cleanup_scene()
    
    print("\nStarting instance system creation...")
    
    # Add CSV debugging
    print(f"CSV_FILEPATH IS: {CSV_FILEPATH}")
    print(f"File exists: {os.path.exists(CSV_FILEPATH)}")
    
    # Read and verify CSV contents
    print("\nReading CSV file...")
    df = pd.read_csv(CSV_FILEPATH)
    
        
    if SITE == 'trimmed-parade':
        df['nodeType'] = 'tree'
        if YEAR == 10:
            df.loc[df['structureID'] == 381, 'size'] = 'senescing'
            df.loc[df['structureID'] == 381, 'control'] = 'improved-tree'
            df.loc[df['structureID'] == 381, 'rotateZ'] += 70
            df.loc[df['structureID'] == 381, 'tree_id'] = 12
            
    
    
    
    print("\nCSV Contents Summary:")
    print(f"Total rows: {len(df)}")
    print("Unique nodeTypes:", df['nodeType'].unique())
    print("Counts by nodeType:")
    print(df['nodeType'].value_counts())
    
    # Setup collections hierarchy
    scene = bpy.context.scene
    if not scene:
        raise ValueError("No active scene found")
        
    year_collection_name = f"Year_{YEAR}"
    
    # Clean up any existing year collection
    existing_year_coll = bpy.data.collections.get(year_collection_name)
    if existing_year_coll:
        # If it exists in any scene, unlink it
        for s in bpy.data.scenes:
            if existing_year_coll.name in [c.name for c in s.collection.children]:
                s.collection.children.unlink(existing_year_coll)
        bpy.data.collections.remove(existing_year_coll)
    
    # Create new year collection
    year_collection = bpy.data.collections.new(year_collection_name)
    scene.collection.children.link(year_collection)
    
    # Get camera position using pass index
    camera_obj = find_camera_by_pass_index(PASS_INDEX)
    if not camera_obj:
        raise ValueError(f"No object found with pass index {PASS_INDEX}")
    
    camera_x = camera_obj.location.x
    camera_y = camera_obj.location.y
    
    print(f"Found camera at position: x={camera_x:.2f}, y={camera_y:.2f}")
    
    # Calculate distances
    df['distance_to_camera'] = np.sqrt(
        np.square(df['x'] - camera_x) + 
        np.square(df['y'] - camera_y)
    )
    
    # Filter by distance
    mask = (
        (df['x'].between(camera_x - DISTANCE_UNITS, camera_x + DISTANCE_UNITS)) & 
        (df['y'].between(camera_y - DISTANCE_UNITS, camera_y + DISTANCE_UNITS))
    )
    df_filtered = df[mask]
    print(f"Filtered {len(df)} objects to {len(df_filtered)} within distance threshold")
    
    # Split into tree, pole, and log dataframes
    treeDF = df_filtered[df_filtered['nodeType'] == 'tree'].copy()
    poleDF = df_filtered[df_filtered['nodeType'] == 'pole'].copy()
    logDF = df_filtered[df_filtered['nodeType'] == 'log'].copy()
    
    print(f"\nFound {len(treeDF)} trees, {len(poleDF)} poles, and {len(logDF)} logs to process")
    
    # Process trees if any exist
    tree_results = None
    pole_results = None
    log_results = None
    
    if len(treeDF) > 0:
        print("\nProcessing trees...")
        tree_results = process_collection(treeDF, PLY_FOLDER, 'tree', year_collection)
        if tree_results[0]:  # point cloud exists
            print(f"Successfully processed {len(tree_results[1])} tree types")
    else:
        print("No trees to process")
    
    # Process poles if any exist
    if len(poleDF) > 0:
        print("\nProcessing poles...")
        pole_results = process_collection(poleDF, PLY_FOLDER, 'pole', year_collection)
        if pole_results[0]:
            print(f"Successfully processed {len(pole_results[1])} pole types")
    else:
        print("No poles to process")
    
    # Process logs if any exist
    if len(logDF) > 0:
        print("\nProcessing logs...")
        log_results = process_collection(logDF, LOG_FOLDER, 'log', year_collection)
        if log_results[0]:
            print(f"Successfully processed {len(log_results[1])} log types")
    else:
        print("No logs to process")
    
    print("\nAll processing complete!")
    return tree_results, pole_results, log_results

if __name__ == "__main__":
    try:
        tree_results, pole_results, log_results = main()
    except Exception as e:
        print(f"\nError: {str(e)}")