# Assume tree_templates is already loaded in RAM as a dictionary
# processedDF is the dataframe containing tree instances (e.g., x, y, z values to translate)

import dask
from dask import delayed
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from tqdm import tqdm
import pickle
import pyvista as pv
import pandas as pd
import random
import numpy as np
import hashlib
from pathlib import Path
import os
import sys

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_code-refactored")
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from refactor_code.paths import scenario_log_df_path, scenario_tree_df_path


# Set a random seed for reproducibility
random.seed(42)

_TEMPLATE_CACHE: dict[tuple[str, float], tuple[pd.DataFrame, Path]] = {}


def _resource_debug_enabled() -> bool:
    return os.environ.get("RESOURCE_DISTRIBUTOR_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _resource_debug(*args, **kwargs):
    if _resource_debug_enabled():
        print(*args, **kwargs)


def _stable_fallback_seed(row, fallback_stage):
    parts = [fallback_stage]
    for field in ("x", "y", "z", "structureID", "tree_number", "NodeID", "logNo", "precolonial", "size", "logSize", "control", "tree_id"):
        value = row.get(field, "missing")
        if pd.isna(value):
            parts.append("nan")
        elif field in {"x", "y", "z"}:
            parts.append(f"{float(value):.6f}")
        else:
            parts.append(str(value))
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _choose_deterministic_template(result, row, fallback_stage):
    ordered = result.sort_values(["precolonial", "size", "control", "tree_id"]).reset_index(drop=True)
    choice_index = _stable_fallback_seed(row, fallback_stage) % len(ordered)
    return ordered.iloc[choice_index]


# Assume tree_templates is already loaded in RAM as a dictionary
# processedDF is the dataframe containing tree instances (e.g., x, y, z values to translate)

def create_pyvista_object(combined_df):
    """
    Converts the processed DataFrame into a PyVista PolyData object.
    """
    print('##########################')
    print(f'Checking if correct combined_df variable: {combined_df.head()}')
    
    # Ensure combined_df is a pandas DataFrame and not a tuple
    print(f"Type of combined_df: {type(combined_df)}")
    
    if not isinstance(combined_df, pd.DataFrame):
        raise ValueError("combined_df is not a pandas DataFrame")

    # Extract the X, Y, Z columns as points
    points = combined_df[['x', 'y', 'z']].to_numpy()

    # Create the PolyData object with points
    polydata = pv.PolyData(points) 

    # Add other columns as point data
    for col in combined_df.columns:
        if col not in ['x', 'y', 'z']:
            polydata.point_data[col] = combined_df[col].to_numpy()

    return polydata

def convert_key_to_tuple(row):
    """
    Converts a string representation of a tuple into an actual tuple.
    This assumes the new key format (isPreColonial, size, control, treeID).
    """
    if isinstance(row['tree_template_key'], str):
        # Assume the string is formatted as "(isPreColonial, size, control, treeID)"
        return eval(row['tree_template_key'])  # Convert the string back to a tuple
    return row['tree_template_key']


###LOG ASSIGNMENT FUNCTIONS###
def preprocess_logLibrary():
    """
    Preprocesses the log library to match the tree resource format.
    """
    #Step 0: get log library
    logLibrary = pd.read_pickle('data/treeOutputs/logLibrary.pkl')

    print(f'first bit of log library is {logLibrary.head()}')
    # Group by logNo and count rows
    log_counts = logLibrary.groupby('logNo').size()
    
    # Filter to keep only logs with >= 200 points
    valid_logs = log_counts[log_counts >= 200].index
    
    # Filter logLibrary to only include valid logs
    logLibrary = logLibrary[logLibrary['logNo'].isin(valid_logs)]

    #add the following columns:
    logLibrary['resource_fallen log'] = 1
    logLibrary['resource_perch branch'] = 0
    logLibrary['resource_hollow'] = 0
    logLibrary['resource_epiphyte'] = 0
    logLibrary['resource_peeling branch'] = 0
    logLibrary['resource_dead branch'] = 0
    logLibrary['resource_other'] = 0
    
    print(f'After filtering, {len(valid_logs)} logs remain')

    return logLibrary

def voxelise_log_library(logLibrary, voxel_size):
    """
    Voxelises the log library by grouping points into voxels for each log.
    
    Parameters:
        logLibrary (pd.DataFrame): The log library DataFrame
        voxel_size (float): The voxel size to use for voxelisation
        
    Returns:
        pd.DataFrame: Voxelised log library
    """
    print(f'Voxelising log library with voxel size {voxel_size}')
    
    # Print column names to debug
    print(f"Log library columns: {logLibrary.columns.tolist()}")
    
    # Create a list to store voxelised logs
    voxelised_logs = []
    
    # Define coordinate columns mapping based on actual columns in logLibrary
    # Assuming the coordinates are in 'X', 'Y', 'Z' columns
    coord_columns = {
        'X': 'voxel_X',
        'Y': 'voxel_Y',
        'Z': 'voxel_Z'
    }
    
    # Process each log group separately
    for log_no, log_group in logLibrary.groupby('logNo'):
        print(f'Voxelising log {log_no} with {len(log_group)} points')
        
        # Create a copy to avoid modifying the original
        voxel_group = log_group.copy()
        
        # Assign voxel coordinates using the correct column names
        for orig_col, voxel_col in coord_columns.items():
            if orig_col in voxel_group.columns:
                voxel_group[voxel_col] = np.floor(voxel_group[orig_col] / voxel_size) * voxel_size
            else:
                print(f"Warning: Column {orig_col} not found in log data")
        
        # Get resource columns
        resource_cols = [col for col in voxel_group.columns if col.startswith('resource_')]
        
        # Group by voxel coordinates and sum all resource columns
        group_cols = list(coord_columns.values())
        try:
            voxelised_log = voxel_group.groupby(group_cols)[resource_cols].sum().reset_index()
            
            # Rename coordinate columns to match expected format for later processing
            voxelised_log = voxelised_log.rename(columns={
                'voxel_X': 'X',
                'voxel_Y': 'Y',
                'voxel_Z': 'Z'
            })
            
            # Add back the logNo column
            voxelised_log['logNo'] = log_no
            
            # Add to the list
            voxelised_logs.append(voxelised_log)
        except Exception as e:
            print(f"Error voxelising log {log_no}: {e}")
            print(f"Columns available: {voxel_group.columns.tolist()}")
            print(f"Group columns: {group_cols}")
            print(f"Resource columns: {resource_cols}")
    
    if not voxelised_logs:
        print("Warning: No logs were successfully voxelised")
        return logLibrary  # Return original if voxelisation failed
        
    # Combine all voxelised logs
    voxelised_logLibrary = pd.concat(voxelised_logs, ignore_index=True)
    
    print(f'Voxelisation complete. Original points: {len(logLibrary)}, Voxelised points: {len(voxelised_logLibrary)}')
    
    return voxelised_logLibrary

def preprocess_logLocationsDF(logLocationsDF, logLibraryDF, seed=42, voxel_size=None):
    """
    Preprocesses the log locations DataFrame by assigning appropriate log models using Dask.
    If voxel_size is provided, voxelises the log library first.
    """
    # Voxelise the log library if voxel_size is provided
    if voxel_size is not None:
        logLibraryDF = voxelise_log_library(logLibraryDF, voxel_size)
    
    # Remove log groups 1, 2, 3, 4
    logLibraryDF = logLibraryDF[~logLibraryDF['logNo'].isin([1, 2, 3, 4])]

    # Create logInfo DataFrame - one row per unique log
    logInfo = logLibraryDF.groupby('logNo').first().reset_index()
    
    # Create a mapping of logSize to available logNos
    size_to_logs = logInfo.groupby('logSize')['logNo'].agg(list).to_dict()
    
    # Debug print
    print("\nDEBUG: Log size to available log numbers mapping:")
    for size, logs in size_to_logs.items():
        print(f"{size}: {logs}")
    
    all_logs = sorted(logInfo['logNo'].astype(int).tolist())

    # Function to select a log model deterministically from stable row fields
    def select_log_model(row):
        row_size = row['logSize']
        matching_logs = sorted(int(log_no) for log_no in size_to_logs.get(row_size, []))
        if matching_logs:
            selected = matching_logs[_stable_fallback_seed(row, f"log-size:{row_size}:{seed}") % len(matching_logs)]
            print(f"Assigned log size {row_size} to log number {selected}")
            return selected
        print(f"Warning: No logs found matching size {row_size}, selecting deterministic fallback log")
        selected = all_logs[_stable_fallback_seed(row, f"log-fallback:{seed}") % len(all_logs)]
        print(f"Deterministically selected log number {selected}")
        return selected

    result_df = logLocationsDF.copy()
    result_df['tree_id'] = result_df.apply(select_log_model, axis=1)

    # Rename columns to match tree resource DF
    result_df = result_df.rename(columns={
        'logNo': 'tree_number',
        'logSize': 'size'
    })
    result_df['log_template_size'] = result_df['size']
    result_df['size'] = 'fallen'
    
    # Initialize new columns with default values
    result_df['precolonial'] = False
    result_df['control'] = 'unassigned'
    result_df['diameter_breast_height'] = -1
    result_df['useful_life_expectancy'] = -1
    result_df['isNewTree'] = False
    result_df['nodeType'] = 'log'  # Mark as log for identification
    
    # Debug print
    print("\nDEBUG: Result DataFrame tree_id distribution:")
    print(result_df['tree_id'].value_counts())
    print(f"Any NaN values in tree_id: {result_df['tree_id'].isnull().any()}")
    
    return result_df

def process_single_log(row, log_templates):
    """
    Processes a single log location using the template.
    
    Parameters:
        row: DataFrame row containing log location and metadata
        log_templates: Dictionary of log templates indexed by tree_id
        
    Returns:
        DataFrame with translated log template and metadata
    """
    template = log_templates[row['tree_id']].copy()  # Changed from logModel to tree_id
    
    # Debug print
    print(f"\nProcessing log model {row['tree_id']}")
    print(f"Template coordinates dtypes:\n{template[['x', 'y', 'z']].dtypes}")
    print(f"Row coordinates dtypes:\n{row[['x', 'y', 'z']].dtypes}")
    
    # Ensure coordinates are numeric before addition
    template[['x', 'y', 'z']] = template[['x', 'y', 'z']].astype(float)
    row_coords = row[['x', 'y', 'z']].astype(float)
    
    # Translate coordinates
    template[['x', 'y', 'z']] += row_coords.values
    
    # Create a new DataFrame with only the necessary columns
    result_df = pd.DataFrame()
    
    # 1. Add coordinates
    result_df[['x', 'y', 'z']] = template[['x', 'y', 'z']]
    
    # 2. Add all resource columns from template
    resource_cols = [col for col in template.columns if col.startswith('resource_')]
    for col in resource_cols:
        result_df[col] = template[col]
    
    # Add all stat columns from template (if they exist)
    stat_cols = [col for col in template.columns if col.startswith('stat_')]
    for col in stat_cols:
        result_df[col] = template[col]
    
    # 3. Add metadata columns from row
    metadata_columns = [
        'tree_number', 'size', 'tree_id', 'precolonial', 'control',
        'diameter_breast_height', 'structureID', 'useful_life_expectancy',
        'isNewTree', 'nodeType', 'log_template_size'
    ]
    
    for col in metadata_columns:
        if col in row:
            result_df[col] = row[col]

    # Broadcast deploy-structure proposal from the log's df row onto template voxels
    _deploy_dec = row.get('proposal-deploy-structure_decision', 'not-assessed')
    _deploy_int = row.get('proposal-deploy-structure_intervention', 'none')
    if _deploy_dec is None or str(_deploy_dec) in ('None', 'nan', ''):
        _deploy_dec = 'not-assessed'
    if _deploy_int is None or str(_deploy_int) in ('None', 'nan', ''):
        _deploy_int = 'none'
    result_df['proposal_deploy_structureV4'] = _deploy_dec
    result_df['proposal_deploy_structureV4_intervention'] = _deploy_int

    return result_df

def create_log_resource_df(logLocationsDF, logLibraryDF, voxel_size=None):
    """
    Creates the final log resource DataFrame using Dask for parallel processing.
    Optionally voxelizes the output if voxel_size is provided.
    """
    # First rename the columns in logLibraryDF
    logLibraryDF = logLibraryDF.rename(columns={
        'X': 'x', 
        'Y': 'y', 
        'Z': 'z',
        'logNo': 'tree_id'  # This is the key rename we need
    })
    
    print(f"Columns after renaming: {logLibraryDF.columns}")  # Debug print
    
    # Create templates dictionary using tree_id as key
    try:
        log_templates = dict(tuple(logLibraryDF.groupby('tree_id')))
    except KeyError as e:
        print(f"Error: Column not found. Available columns: {logLibraryDF.columns}")
        raise e
    
    # Convert to Dask DataFrame
    dask_df = dd.from_pandas(logLocationsDF, npartitions=10)
    
    # Create delayed objects for each row
    delayed_results = []
    for _, row in tqdm(dask_df.iterrows(), total=len(logLocationsDF)):
        delayed_results.append(delayed(process_single_log)(row, log_templates))
    
    # Compute results in parallel
    with dask.diagnostics.ProgressBar():
        results = dask.compute(*delayed_results)
    
    # Combine results
    logResourceDF = pd.concat(results, ignore_index=True)

    # Debug prints
    print("\nDEBUG INFO:")
    print(f"logResourceDF columns: {logResourceDF.columns}")
    print(f"logResourceDF dtypes:\n{logResourceDF.dtypes}")
    print("\nFirst few rows of coordinates:")
    print(logResourceDF[['x', 'y', 'z']].head())
    
    # Ensure coordinates are numeric
    logResourceDF[['x', 'y', 'z']] = logResourceDF[['x', 'y', 'z']].astype(float)
    
    # Voxelize if needed
    if voxel_size is not None:
        print(f"Voxelizing log resource DF with voxel size {voxel_size}")
        logResourceDF = voxelize_log_resource_df(logResourceDF, voxel_size)
    else:
        # If not voxelizing, still add stat_ columns based on resource_ columns
        resource_cols = [col for col in logResourceDF.columns if col.startswith('resource_')]
        for col in resource_cols:
            stat_col = col.replace('resource_', 'stat_')
            print(f"Creating stat column: {stat_col} from {col}")
            logResourceDF[stat_col] = logResourceDF[col]
    
    return logResourceDF

def voxelize_log_resource_df(df, voxel_size):
    """
    Voxelizes the log resource DataFrame.
    
    Parameters:
        df (pd.DataFrame): Log resource DataFrame with x, y, z coordinates
        voxel_size (float): Voxel size for discretization
        
    Returns:
        pd.DataFrame: Voxelized log resource DataFrame
    """
    print(f"Original log resource DF size: {len(df)}")
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Add voxel coordinate columns
    df_copy['voxel_x'] = np.floor(df_copy['x'] / voxel_size) * voxel_size
    df_copy['voxel_y'] = np.floor(df_copy['y'] / voxel_size) * voxel_size
    df_copy['voxel_z'] = np.floor(df_copy['z'] / voxel_size) * voxel_size
    
    # Identify resource columns (those starting with 'resource_')
    resource_cols = [col for col in df_copy.columns if col.startswith('resource_')]
    
    # Identify other columns (non-coordinate, non-resource)
    other_cols = [col for col in df_copy.columns if col not in resource_cols + 
                 ['x', 'y', 'z', 'voxel_x', 'voxel_y', 'voxel_z']]
    
    print(f"Resource columns: {resource_cols}")
    print(f"Other columns: {other_cols}")
    
    # Group by voxel coordinates
    grouped = df_copy.groupby(['voxel_x', 'voxel_y', 'voxel_z'])
    
    # Aggregate resource columns by sum
    agg_resources = grouped[resource_cols].sum()
    
    # For other columns, take the first row in each group
    agg_others = grouped[other_cols].first()
    
    # Combine the aggregated DataFrames
    voxelized_df = pd.concat([agg_resources, agg_others], axis=1).reset_index()
    
    # Rename voxel coordinates back to x, y, z
    voxelized_df = voxelized_df.rename(columns={
        'voxel_x': 'x',
        'voxel_y': 'y',
        'voxel_z': 'z'
    })
    
    # Add stat_ columns based on resource_ columns
    for col in resource_cols:
        stat_col = col.replace('resource_', 'stat_')
        print(f"Creating stat column: {stat_col} from {col}")
        voxelized_df[stat_col] = voxelized_df[col]
    
    print(f"Voxelized log resource DF size: {len(voxelized_df)}")
    
    return voxelized_df


def initialise_and_translate_tree(tree_template, row):
    """
    Translates the tree template by the given x, y, z offsets and initializes additional columns.
    """
    tree_template_copy = tree_template.copy()

    # Translate the X, Y, Z coordinates by offsets from the row
    tree_template_copy['x'] += row['x']
    tree_template_copy['y'] += row['y']
    tree_template_copy['z'] += row['z']

    # Initialize additional columns: precolonial, size, control, tree_id, useful_life_expectency
    tree_template_copy['precolonial'] = row['precolonial']
    tree_template_copy['size'] = row['size']
    tree_template_copy['control'] = row['control']
    tree_template_copy['tree_id'] = row['tree_id']

    tree_template_copy['diameter_breast_height'] = row['diameter_breast_height']
    tree_template_copy['tree_number'] = row['tree_number']
    tree_template_copy['NodeID'] = row['NodeID']
    tree_template_copy['structureID'] = row['structureID']
    tree_template_copy['useful_life_expectancy'] = row['useful_life_expectancy']
    
    # check if nodeType is in df, if not set to 'tree'
    if 'nodeType' not in tree_template_copy.columns:
        tree_template_copy['nodeType'] = 'tree'
    else:
        tree_template_copy['nodeType'] = row['nodeType']

    if 'isNewTree' in row:
        tree_template_copy['isNewTree'] = row['isNewTree']

    # Broadcast node-level proposal/intervention values directly into V4 arrays.
    # These are excluded from the forest_* rename step so they land on ds with
    # their V4 names, ready for the proposal logic in a_info_gather_capabilities.
    _proposal_map = {
        'proposal_decayV4': 'proposal-decay_decision',
        'proposal_decayV4_intervention': 'proposal-decay_intervention',
        'proposal_release_controlV4': 'proposal-release-control_decision',
        'proposal_release_controlV4_intervention': 'proposal-release-control_intervention',
        'proposal_deploy_structureV4': 'proposal-deploy-structure_decision',
        'proposal_deploy_structureV4_intervention': 'proposal-deploy-structure_intervention',
    }
    _proposal_defaults = {
        'proposal_decayV4': 'not-assessed',
        'proposal_decayV4_intervention': 'none',
        'proposal_release_controlV4': 'not-assessed',
        'proposal_release_controlV4_intervention': 'none',
        'proposal_deploy_structureV4': 'not-assessed',
        'proposal_deploy_structureV4_intervention': 'none',
    }
    for v4_col, node_col in _proposal_map.items():
        val = row[node_col] if node_col in row.index else None
        if val is None or (isinstance(val, float) and pd.isna(val)) or str(val) in ('None', 'nan', ''):
            val = _proposal_defaults[v4_col]
        tree_template_copy[v4_col] = val

    return tree_template_copy

def update_old_template(tree_templateDF):
    # Find the unique values for tree_template['resource']
    unique_resources = tree_templateDF['resource'].unique()
    print(f'Found {len(unique_resources)} unique resources: {unique_resources}')

    # For each resource in unique resources
    for resource in unique_resources:
        # Create column name
        col_name = f'treeResource_{resource}'
        print(f'Creating column {col_name}')
        
        # Initialize column as False
        tree_templateDF[col_name] = False
        
        # Get mask where resource matches
        mask = tree_templateDF['resource'] == resource
        
        # Update column to True where mask matches
        tree_templateDF.loc[mask, col_name] = True
        print(f'Updated {mask.sum()} rows for {resource}')
    
    print('Finished updating template with resource columns')
    return tree_templateDF

def query_tree_template(df, row, precolonial, size, control, tree_id):
    """
    Attempts to query the tree template using fallbacks if necessary.
    Fallbacks are tied to stable row position/identity fields so reruns resolve
    to the same donor template independent of parallel execution order.
    """
    # 0. If size is 'artificial', search for 'snag' instead
    if size == 'artificial':
        print(f"Converting 'artificial' size to 'snag' for template search")
        size = 'snag'

    # 1. Try to find an exact match
    result = df.loc[(df['precolonial'] == precolonial) &
                    (df['size'] == size) &
                    (df['control'] == control) &
                    (df['tree_id'] == tree_id)]

    if not result.empty:
        print(f"Exact match found: {(precolonial, size, control, tree_id)}")
        return result.iloc[0]['template']

    # 2. Try to find a match with the first three columns, pick any tree_id if no match found
    result = df.loc[(df['precolonial'] == precolonial) &
                    (df['size'] == size) &
                    (df['control'] == control)]

    if not result.empty:
        print(f"Falling back to control and deterministically picking tree_id for: {(precolonial, size, control)}")
        chosen_row = _choose_deterministic_template(result, row, "control-fallback")
        print(f"Found template with key: {chosen_row[['precolonial', 'size', 'control', 'tree_id']]}")
        return chosen_row['template']

    # 3. Try to match by just `precolonial` and `size`, and randomly pick control and tree_id
    result = df.loc[(df['precolonial'] == precolonial) &
                    (df['size'] == size)]

    if not result.empty:
        print(f"Falling back to size and deterministically picking control and tree_id for: {(precolonial, size)}")
        chosen_row = _choose_deterministic_template(result, row, "size-fallback")
        print(f"Found template with key: {chosen_row[['precolonial', 'size', 'control', 'tree_id']]}")
        return chosen_row['template']

    # 4. If no match found at all
    print(f"######################ERROR######################")
    print(f"No match found for: {(precolonial, size, control, tree_id)}")
    return None

def process_single_tree(row, tree_templates_df):
    """
    Processes a single tree using fallbacks.
    """
    precolonial = bool(row['precolonial'])
    size = str(row['size']).strip()
    control = str(row['control']).strip()
    tree_id = int(row['tree_id'])

    # Query the DataFrame using the fallback logic
    template = query_tree_template(tree_templates_df, row, precolonial, size, control, tree_id)

    if template is None:
        print(f"No match found for: {(precolonial, size, control, tree_id)}")
        return None

    # Use the retrieved template and apply it in your logic
    return initialise_and_translate_tree(template, row)


def _template_root() -> Path:
    override = os.environ.get("TREE_TEMPLATE_ROOT")
    if override:
        return Path(override)
    return _canonical_base_template_root()


def _canonical_base_template_root() -> Path:
    override = os.environ.get("TREE_TEMPLATE_BASE_ROOT") or os.environ.get("BASE_TREE_TEMPLATES_ROOT")
    if override:
        return Path(override)
    return Path("_data-refactored/model-inputs/tree_libraries/base/trees")


def _apply_template_edits(
    canonical_templates: pd.DataFrame,
    template_edits: pd.DataFrame,
) -> pd.DataFrame:
    key_cols = ["precolonial", "size", "control", "tree_id"]
    canonical = canonical_templates.copy()
    edits = template_edits.copy()

    for col in ("precolonial",):
        if col in canonical.columns:
            canonical[col] = canonical[col].astype(bool)
        if col in edits.columns:
            edits[col] = edits[col].astype(bool)
    for col in ("size", "control"):
        if col in canonical.columns:
            canonical[col] = canonical[col].astype(str).str.strip()
        if col in edits.columns:
            edits[col] = edits[col].astype(str).str.strip()
    if "tree_id" in canonical.columns:
        canonical["tree_id"] = canonical["tree_id"].astype(int)
    if "tree_id" in edits.columns:
        edits["tree_id"] = edits["tree_id"].astype(int)

    edit_keys = set(tuple(row[col] for col in key_cols) for _, row in edits[key_cols].iterrows())
    keep_mask = canonical[key_cols].apply(lambda row: tuple(row[col] for col in key_cols) not in edit_keys, axis=1)
    merged = pd.concat([canonical[keep_mask], edits], ignore_index=True)
    merged = merged.sort_values(key_cols).reset_index(drop=True)
    return merged


def _load_full_template_table(template_dir: Path) -> tuple[pd.DataFrame, Path]:
    combined_candidates = [
        template_dir / "template-library.overrides-applied.pkl",
        template_dir / "template-library.base.pkl",
    ]
    for path in combined_candidates:
        if path.exists():
            return pd.read_pickle(path), path

    template_edits_path = template_dir / "template-library.selected-overrides.pkl"
    if template_edits_path.exists():
        canonical_path = _canonical_base_template_root() / "template-library.base.pkl"
        canonical_templates = pd.read_pickle(canonical_path)
        template_edits = pd.read_pickle(template_edits_path)
        return _apply_template_edits(canonical_templates, template_edits), template_edits_path

    raise FileNotFoundError(
        f"Could not find combined templates in {template_dir}. "
        "Expected template-library.overrides-applied.pkl, template-library.base.pkl, "
        "or template-library.selected-overrides.pkl."
    )


def _load_tree_templates(voxel_size: float | int) -> tuple[pd.DataFrame, Path]:
    template_dir = _template_root()
    cache_key = (str(template_dir.resolve()), float(voxel_size))
    if cache_key in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[cache_key]

    voxel_prefix = f"{voxel_size:g}"
    if float(voxel_size) == 0:
        loaded = _load_full_template_table(template_dir)
        _TEMPLATE_CACHE[cache_key] = loaded
        return loaded

    voxel_candidates = [
        template_dir / f"{voxel_prefix}_combined_voxel_templateDF.pkl",
        template_dir / f"combined_voxelSize_{voxel_prefix}_templateDF.pkl",
    ]
    for path in voxel_candidates:
        if path.exists():
            loaded = (pd.read_pickle(path), path)
            _TEMPLATE_CACHE[cache_key] = loaded
            return loaded

    raise FileNotFoundError(
        f"Could not find voxel template table for voxel_size={voxel_size} in {template_dir}. "
        f"Tried: {[str(path.name) for path in voxel_candidates]}"
    )

def process_all_trees(locationDF, voxel_size=0.5):
    """
    Process all trees with the fallback logic in parallel using Dask.
    """
    print(f'Loading tree templates of voxel size {voxel_size}')

    tree_templates_df, input_path = _load_tree_templates(voxel_size)
    print(f"Loaded tree templates from {input_path}")
    _resource_debug("DataFrame columns:", tree_templates_df.columns)
    if _resource_debug_enabled():
        _resource_debug("DataFrame info:", tree_templates_df.info())
        _resource_debug("First row:", tree_templates_df.iloc[0])
        first_template = tree_templates_df.iloc[0]['template']
        _resource_debug(f'Template df is {first_template.head()}')

    # Enforce Python native types and strip potential hidden characters
    locationDF['precolonial'] = locationDF['precolonial'].astype(bool)  # Convert to Python bool
    locationDF['size'] = locationDF['size'].astype(str).str.strip()  # Ensure string and strip
    locationDF['control'] = locationDF['control'].astype(str).str.strip()  # Ensure string and strip
    locationDF['tree_id'] = locationDF['tree_id'].astype(int)  # Ensure integer type

    # Convert processedDF to a Dask DataFrame
    dask_df = dd.from_pandas(locationDF, npartitions=10)

    # Use TQDM to show progress during map_partitions
    delayed_results = []
    for _, row in tqdm(dask_df.iterrows(), total=len(locationDF)):
        delayed_results.append(delayed(process_single_tree)(row, tree_templates_df))

    # Use Dask's ProgressBar
    with dask.diagnostics.ProgressBar():
        # Compute the results in parallel and show progress
        results = dask.compute(*delayed_results)

    print(f'Resources distributed')
    print('Combining results into a single DataFrame...')

    # Combine the results into a single DataFrame, excluding None results
    resourceDF = pd.concat([res for res in results if res is not None])
    return locationDF, resourceDF

def convertToPoly(resourceDF):
    
    # Extracting coordinates from the DataFrame
    points = resourceDF[['x', 'y', 'z']].values
    print("Extracted coordinates from DataFrame for PolyData creation.")

    #define columns to use as point_data attributes
    #all columns starting with resource_
    resource_columns = [col for col in resourceDF.columns if col.startswith('resource_')]
    
    #all columns starting with stat_
    stat_columns = [col for col in resourceDF.columns if col.startswith('stat_')]

    #extend with 'precolonial', 'size', 'control', 'tree_id', 'nodeID', 'structureID'
    attribute_columns = ['precolonial', 'size', 'control', 'tree_id', 'useful_life_expectancy']

    # Check which attribute columns exist in resourceDF
    existing_attribute_columns = [col for col in attribute_columns if col in resourceDF.columns]

    if len(existing_attribute_columns) < len(attribute_columns):
        print("Warning: Some attribute columns are missing from resourceDF:")
        print(set(attribute_columns) - set(existing_attribute_columns))

    # Creating a PolyData object from the extracted points
    poly = pv.PolyData(points)
    print("Created PolyData object from extracted points.")
    
    # Add resource columns
    for resource in resource_columns:
        print(f'adding resource column {resource}')
        print(f'counts: {resourceDF[resource].value_counts()}')
        print(f'types: {resourceDF[resource].dtype}')

        poly.point_data[resource] = resourceDF[resource].values
        print(f"Added column '{resource}' as point data attribute to PolyData.")
    
    # Add stat columns
    for stat in stat_columns:
        print(f'adding stat column {stat}')
        print(f'counts: {resourceDF[stat].value_counts()}')
        print(f'types: {resourceDF[stat].dtype}')

        poly.point_data[stat] = resourceDF[stat].values
        print(f"Added column '{stat}' as point data attribute to PolyData.")

    # Add existing attribute columns only
    for col in existing_attribute_columns:
        poly.point_data[col] = resourceDF[col].values
        print(f"Added column '{col}' as point data attribute to PolyData.")
    
    print(f'PolyData object created with {len(poly.points)} points and {len(existing_attribute_columns) + len(resource_columns) + len(stat_columns)} point data attributes')
    return poly
    #poly.plot(scalars=treeResource_peeling bark', render_points_as_spheres=True)

def rotate_resource_structures(locationDF, resourceDF):
    def rotate_points(points, angle_deg, pivot):
        angle_rad = np.deg2rad(angle_deg)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        rotation_matrix = np.array([[cos_angle, -sin_angle],
                                    [sin_angle, cos_angle]])
        shifted_points = points - pivot
        rotated_shifted = shifted_points @ rotation_matrix.T
        rotated = rotated_shifted + pivot
        return rotated

    def apply_rotation(df, angle, pivot):
        points = df[['x', 'y']].values
        rotated = rotate_points(points, angle, pivot)
        df = df.copy()
        df['x'] = rotated[:, 0]
        df['y'] = rotated[:, 1]
        return df

    def rotate_group(group):
        # Get rotation angle and pivot point from respective dictionaries
        angle = rotation_angles.get(group.name, 0.0)
        pivot = tree_positions.get(group.name, np.array([0.0, 0.0]))
        if angle == 0.0:
            return group
        return apply_rotation(group, angle, pivot)

    # Create two separate dictionaries
    temp_dict = locationDF.set_index('structureID')[['x', 'y', 'rotateZ']].to_dict('index')
    
    # Dictionary for pivot points (as numpy arrays)
    tree_positions = {k: np.array([v['x'], v['y']]) for k, v in temp_dict.items()}
    
    # Dictionary for rotation angles
    rotation_angles = {k: v['rotateZ'] for k, v in temp_dict.items()}

    # Group by 'structureID' and apply rotation
    updated_resourceDF = resourceDF.groupby('structureID').apply(rotate_group).reset_index(drop=True)
    return updated_resourceDF

# Main execution
if __name__ == "__main__":
    site = 'city'
    scenarioVoxelSize = 1
    outputVoxelSize = 1
    year = 30
    
    # Load tree data
    scenario = 'positive'
    filepath = scenario_tree_df_path(site, scenario, year, scenarioVoxelSize)
    log_filepath = scenario_log_df_path(site, scenario, year, scenarioVoxelSize)

    print(f'loading treeDF from {filepath}')
    print(f'loading logDF from {log_filepath}')

    treeDF = pd.read_csv(filepath)
    logLocationsDF = pd.read_csv(log_filepath)

    # Process logs
    logLibrary = preprocess_logLibrary()
    logLocationsDF = preprocess_logLocationsDF(logLocationsDF, logLibrary)
    # Pass voxel size to create_log_resource_df for voxelization at the end
    logResourceDF = create_log_resource_df(logLocationsDF, logLibrary, voxel_size=outputVoxelSize)
        
    # Process trees
    locationDF, resourceDF = process_all_trees(treeDF, voxel_size=outputVoxelSize)
    resourceDF = rotate_resource_structures(locationDF, resourceDF)
    
    print(f'logDF is {logResourceDF.head()}')
    
    # Combine resources
    combined_resourceDF = pd.concat([resourceDF, logResourceDF], ignore_index=True)
    
    # Convert to PolyData and save
    poly = convertToPoly(combined_resourceDF)
    polyfilePath = filepath.parent / f'{site}_{scenario}_{outputVoxelSize}_treeDF_{year}_poly.vtk'

    # Create a plotter
    plotter = pv.Plotter()

    # Option 1: Simple point cloud visualization
    # plotter.add_mesh(poly, scalars='resource_fallen log', render_points_as_spheres=True, 
    #                 point_size=5.0, cmap='viridis')

    # Option 2: Voxel visualization with cubes
    # Create a cube glyph with the correct voxel size
    cube = pv.Cube(x_length=outputVoxelSize, y_length=outputVoxelSize, z_length=outputVoxelSize)

    # Create glyphs at each point
    glyphs = poly.glyph(geom=cube, orient=False, scale=False)

    # Add the glyphs to the plotter with a colormap based on a data attribute
    plotter.add_mesh(glyphs, scalars='resource_fallen log', cmap='turbo', 
                    show_scalar_bar=True)
    # Enable eye-dome lighting for better depth perception
    plotter.enable_eye_dome_lighting()

    # Add axes for reference
    plotter.add_axes()

    # Set a good camera position
    plotter.camera_position = 'xy'
    plotter.reset_camera()

    # Show the plot
    plotter.show()

    print(f'saving polydata to {polyfilePath}')
    poly.save(polyfilePath)
    print(f'exported poly to {polyfilePath}')
    
    print(f'processing complete')


    





    print(f'processing complete')
