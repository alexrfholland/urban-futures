"""
Colors and color mapping functionality for scenario visualization.

This module centralizes all color-related functionality:
- Color definitions for different element types
- Mapping functions to convert categorical data to color indices
- Colormap generation functions
"""
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

def get_lifecycle_colors():
    """
    Get color definitions for tree lifecycle states.
    
    Returns:
    dict: Dictionary mapping lifecycle stages to RGB color tuples (0-255 range)
    """
    return {
        "small": (170, 219, 94),        # Light green
        "medium": (154, 185, 222),      # Light blue
        "large": (249, 159, 118),       # Salmon/orange
        "senescing": (235, 155, 197),   # Pink
        "snag": (252, 227, 88),         # Yellow
        "fallen": (130, 203, 185),       # Teal/mint
        "artificial": (255, 0, 0)        # Red
    }


def get_resource_colours():
    """
    Get color definitions for resource categories.
    
    Returns:
    dict: Dictionary mapping resource categories to RGB color tuples (0-255 range)
    """
    return {
        "other": (158, 158, 158),
        "perch branch": (255, 152, 0),
        "dead branch": (33, 150, 243),
        "peeling bark": (255, 235, 59),
        "epiphyte": (139, 195, 74),
        "fallen log": (121, 85, 72),
        "hollow": (156, 39, 176)
    }



def get_bioenvelope_colors():
    """
    Get color definitions for bioenvelope categories.
    
    Returns:
    dict: Dictionary mapping bioenvelope categories to RGB color tuples (0-255 range)
    """
    return {
        'none': (255, 255, 255),        # White
        'rewilded': (27, 158, 119),     # Strong green (Dark2)
        'exoskeleton': (255, 217, 47),  # Yellow (from Set3, originally otherGround)
        'footprint-depaved': (251, 128, 114),  # Bright orange (Set3)
        'node-rewilded': (27, 158, 119),  # Same as rewilded (Dark2 green)
        'otherGround': (190, 186, 218),   # Soft blue (Set3, originally exoskeleton)
        'livingFacade': (179, 222, 105),  # Bright pink (Set3)
        'greenRoof': (127, 201, 127),     # Green from Accent
        'brownRoof': (253, 180, 98)       # Brown from Set2
    }

def normalize_rgb_colors(color_dict):
    """
    Normalize RGB colors from 0-255 range to 0-1 range.
    
    Parameters:
    color_dict (dict): Dictionary mapping categories to RGB tuples (0-255 range)
    
    Returns:
    dict: Dictionary mapping categories to normalized RGB tuples (0-1 range)
    """
    return {name: tuple(v/255 for v in color) for name, color in color_dict.items()}

def get_lifecycle_stages():
    """
    Get the ordered list of lifecycle stages.
    
    Returns:
    list: Ordered list of lifecycle stage names
    """
    return list(get_lifecycle_colors().keys())

def get_bioenvelope_stages():
    """
    Get the ordered list of bioenvelope categories.
    
    Returns:
    list: Ordered list of bioenvelope category names
    """
    return list(get_bioenvelope_colors().keys())

def get_lifecycle_to_int_mapping():
    """
    Get mapping from lifecycle stages to integer indices.
    
    Returns:
    dict: Dictionary mapping lifecycle stages to integer indices (starting from 1)
    """
    return {stage: i+1 for i, stage in enumerate(get_lifecycle_stages())}

def get_int_to_lifecycle_mapping():
    """
    Get mapping from integer indices to lifecycle stages.
    
    Returns:
    dict: Dictionary mapping integer indices to lifecycle stage names
    """
    return {v: k for k, v in get_lifecycle_to_int_mapping().items()}

def get_bioenvelope_to_int_mapping():
    """
    Get mapping from bioenvelope categories to integer indices.
    
    Returns:
    dict: Dictionary mapping bioenvelope categories to integer indices (starting from 1)
    """
    return {stage: i+1 for i, stage in enumerate(get_bioenvelope_stages())}

def get_int_to_bioenvelope_mapping():
    """
    Get mapping from integer indices to bioenvelope categories.
    
    Returns:
    dict: Dictionary mapping integer indices to bioenvelope category names
    """
    return {v: k for k, v in get_bioenvelope_to_int_mapping().items()}

def get_lifecycle_display_names():
    """
    Get user-friendly display names for lifecycle stages.
    
    Returns:
    dict: Dictionary mapping lifecycle stage keys to display names
    """
    return {
        "small": "Small Tree",
        "medium": "Medium Tree",
        "large": "Large Tree",
        "senescing": "Senescing Tree",
        "snag": "Snag", 
        "fallen": "Fallen Log"
    }

def get_bioenvelope_display_names():
    """
    Get user-friendly display names for bioenvelope categories.
    
    Returns:
    dict: Dictionary mapping bioenvelope category keys to display names
    """
    return {
        "none": "None",
        "rewilded": "Rewilded",
        "exoskeleton": "Exoskeleton",
        "footprint-depaved": "Depaved",
        "node-rewilded": "Node Rewilded",
        "otherGround": "Other Ground",
        "livingFacade": "Living Facade",
        "greenRoof": "Green Roof",
        "brownRoof": "Brown Roof"
    }

def create_colormap_from_dict(color_dict, name="custom_cmap"):
    """
    Create a matplotlib colormap from a dictionary of colors.
    
    Parameters:
    color_dict (dict): Dictionary mapping categories to RGB tuples (0-255 range)
    name (str): Name for the colormap
    
    Returns:
    matplotlib.colors.LinearSegmentedColormap: Custom colormap
    """
    # Get ordered categories (keys)
    categories = list(color_dict.keys())
    
    # Normalize colors to 0-1 range
    colors_norm = {cat: tuple(v/255 for v in color) for cat, color in color_dict.items()}
    
    # Create evenly spaced positions
    positions = np.linspace(0, 1, len(categories))
    
    # Create position-color pairs
    cmap_list = [(pos, colors_norm[cat]) for pos, cat in zip(positions, categories)]
    
    # Create and return the colormap
    return mcolors.LinearSegmentedColormap.from_list(name, cmap_list)

def get_lifecycle_colormap():
    """
    Get the colormap for tree lifecycle stages.
    
    Returns:
    matplotlib.colors.LinearSegmentedColormap: Colormap for tree lifecycle stages
    """
    return create_colormap_from_dict(get_lifecycle_colors(), "lifecycle_cmap")

def get_bioenvelope_colormap():
    """
    Get the colormap for bioenvelope categories.
    
    Returns:
    matplotlib.colors.LinearSegmentedColormap: Colormap for bioenvelope categories
    """
    return create_colormap_from_dict(get_bioenvelope_colors(), "bioenvelope_cmap")

def map_categories_to_indices(data, category_dict, default=0):
    """
    Map categorical data to integer indices using a provided dictionary.
    
    Parameters:
    data (array-like): Array of categorical values to map
    category_dict (dict): Dictionary mapping categories to integer indices
    default (int): Default value for categories not found in the dictionary
    
    Returns:
    numpy.ndarray: Array of integer indices
    """
    # Convert input to pandas Series for efficient mapping
    return pd.Series(data).map(category_dict).fillna(default).astype(np.int32).values

def map_numeric_to_lifecycle_indices(numeric_data):
    """
    Map numeric data to lifecycle stage indices based on threshold values.
    
    Parameters:
    numeric_data (numpy.ndarray): Array of numeric values
    
    Returns:
    numpy.ndarray: Array of lifecycle stage indices
    """
    lifecycle_to_int = get_lifecycle_to_int_mapping()
    result = np.zeros(len(numeric_data), dtype=np.int32)
    
    # Map ranges to lifecycle stages (these thresholds can be adjusted as needed)
    result[(numeric_data < 10)] = lifecycle_to_int.get('small', 0)
    result[(numeric_data >= 10) & (numeric_data < 20)] = lifecycle_to_int.get('medium', 0)
    result[(numeric_data >= 20) & (numeric_data < 30)] = lifecycle_to_int.get('large', 0)
    result[(numeric_data >= 30) & (numeric_data < 40)] = lifecycle_to_int.get('senescing', 0)
    result[(numeric_data >= 40) & (numeric_data < 50)] = lifecycle_to_int.get('snag', 0)
    result[(numeric_data >= 50)] = lifecycle_to_int.get('fallen', 0)
    
    return result

def map_to_lifecycle_indices(data):
    """
    Map data to lifecycle stage indices, handling both string and numeric data.
    
    Parameters:
    data (array-like): Array of values to map to lifecycle stages
    
    Returns:
    numpy.ndarray: Array of lifecycle stage indices
    """
    if hasattr(data, 'dtype') and data.dtype.kind in ['U', 'S']:  # String data
        return map_categories_to_indices(data, get_lifecycle_to_int_mapping())
    else:  # Numeric data
        return map_numeric_to_lifecycle_indices(data)

def map_to_bioenvelope_indices(data):
    """
    Map data to bioenvelope category indices.
    
    Parameters:
    data (array-like): Array of bioenvelope category names
    
    Returns:
    numpy.ndarray: Array of bioenvelope category indices
    """
    return map_categories_to_indices(data, get_bioenvelope_to_int_mapping())

# For backward compatibility, expose some constants
LIFECYCLE_COLORS = get_lifecycle_colors()
BIOENVELOPE_COLORS = get_bioenvelope_colors()
LIFECYCLE_STAGES = get_lifecycle_stages()
LIFECYCLE_TO_INT = get_lifecycle_to_int_mapping()
INT_TO_LIFECYCLE = get_int_to_lifecycle_mapping()
BIOENVELOPE_STAGES = get_bioenvelope_stages()
BIOENVELOPE_TO_INT = get_bioenvelope_to_int_mapping()
INT_TO_BIOENVELOPE = get_int_to_bioenvelope_mapping()
LIFECYCLE_DISPLAY_NAMES = get_lifecycle_display_names()
BIOENVELOPE_DISPLAY_NAMES = get_bioenvelope_display_names()

# --- Urban Elements --- #

def get_urban_element_colors():
    """
    Get color definitions for urban element categories.
    
    Returns:
    dict: Dictionary mapping urban element categories to RGB color tuples (0-255 range)
    """
    bio_colors = get_bioenvelope_colors() # Get bioenvelope colors for reuse
    return {
        'none': (255, 255, 255),               # White
        'open space': bio_colors.get('otherGround', (190, 186, 218)), # Soft blue from bioenvelope
        'green roof': bio_colors.get('greenRoof', (127, 201, 127)), # Green from bioenvelope
        'brown roof': bio_colors.get('brownRoof', (253, 180, 98)), # Brown from bioenvelope
        'facade': bio_colors.get('livingFacade', (179, 222, 105)), # Pink/Green from bioenvelope
        'roadway': (176, 58, 46),              # Burnt Red
        'busy roadway': (255, 0, 0),              # Bright Red
        'existing conversion': bio_colors.get('node-rewilded', (27, 158, 119)), # Green from bioenvelope
        'other street potential': bio_colors.get('exoskeleton', (255, 217, 47)), # Yellow from bioenvelope
        'parking': bio_colors.get('footprint-depaved', (251, 128, 114)) # Orange from bioenvelope
    }

def get_urban_element_stages():
    """
    Get the ordered list of urban element categories.
    
    Returns:
    list: Ordered list of urban element category names
    """
    # Ensure 'none' is first if present, otherwise maintain dict order
    colors = get_urban_element_colors()
    stages = list(colors.keys())
    if 'none' in stages:
        stages.insert(0, stages.pop(stages.index('none')))
    return stages

def get_urban_element_to_int_mapping():
    """
    Get mapping from urban element categories to integer indices.
    
    Returns:
    dict: Dictionary mapping urban element categories to integer indices (starting from 1)
    """
    # Assign index 0 to 'none' if it exists, others start from 1
    stages = get_urban_element_stages()
    mapping = {}
    current_index = 1
    for stage in stages:
        if stage == 'none':
            mapping[stage] = 0
        else:
            mapping[stage] = current_index
            current_index += 1
    return mapping
    # Original simple mapping: {stage: i+1 for i, stage in enumerate(get_urban_element_stages())}

def get_int_to_urban_element_mapping():
    """
    Get mapping from integer indices to urban element categories.
    
    Returns:
    dict: Dictionary mapping integer indices to urban element category names
    """
    return {v: k for k, v in get_urban_element_to_int_mapping().items()}

def get_urban_element_display_names():
    """
    Get user-friendly display names for urban element categories.
    
    Returns:
    dict: Dictionary mapping urban element category keys to display names
    """
    # Simple title case for now, can be customized later
    return {stage: stage.replace('_', ' ').title() for stage in get_urban_element_stages()}

def get_urban_element_colormap():
    """
    Get the colormap for urban element categories.
    
    Returns:
    matplotlib.colors.LinearSegmentedColormap: Colormap for urban element categories
    """
    return create_colormap_from_dict(get_urban_element_colors(), "urban_element_cmap")

def map_to_urban_element_indices(data):
    """
    Map data to urban element category indices.
    
    Parameters:
    data (array-like): Array of urban element category names
    
    Returns:
    numpy.ndarray: Array of urban element category indices
    """
    # Use the mapping that assigns 0 to 'none'
    return map_categories_to_indices(data, get_urban_element_to_int_mapping(), default=0)

# Add Urban Element Constants
URBAN_ELEMENT_COLORS = get_urban_element_colors()
URBAN_ELEMENT_STAGES = get_urban_element_stages()
URBAN_ELEMENT_TO_INT = get_urban_element_to_int_mapping()
INT_TO_URBAN_ELEMENT = get_int_to_urban_element_mapping()
URBAN_ELEMENT_DISPLAY_NAMES = get_urban_element_display_names() 