# Tree Data Structures

## Resources Dictionary

The `tree_resources_dict` contains resource counts for different tree configurations.

### Keys
Each key is a tuple with 4 elements:
1. `is_precolonial` (bool): Whether the tree is precolonial
2. `size` (str): Tree size - 'small', 'medium', or 'large'  
3. `control` (str): Control category - 'street-tree', 'park-tree', or 'reserve-tree'
4. `improvement` (bool): Whether improvement logic was applied

### Values 
Each value is a dictionary mapping resource names to their counts:
- 'peeling bark'
- 'dead branch'  
- 'fallen log'
- 'leaf litter'
- 'hollow'
- 'epiphyte'

## Eucalypt Template Dictionary

Located at `data/revised/trees/updated_tree_dict.pkl`

### Keys
Each key is a tuple with 4 elements:
1. `precolonial` (bool): Whether tree is precolonial
2. `size` (str): Tree size - 'small', 'medium', 'large', 'senescing', 'snag', 'fallen', or 'propped'
3. `control` (str): Control type - 'reserve-tree', 'park-tree', 'street-tree', or 'improved-tree'
4. `tree_id` (int): Unique tree identifier

## File Creation Process

###

### updated_trees_dict.pkl
Created by `final/tree_processing/euc_convertPickle.py`

This script processes `revised_tree_dict.pkl` by:
- Filtering out leaf litter
- Converting coordinate columns to lowercase
- Converting single 'resource' column to multiple boolean columns
- Creating VTK visualization files

### revised_tree_dict.pkl
Created by `f_temp_adjustTreeDict`

Combines and reformats trees from:
1. `data/treeOutputs/adjusted_tree_templates.pkl`
2. `data/treeOutputs/fallen_trees_dict.pkl`

Process:
- Loads pickle files
- Converts keys to 4-tuple structure:
  - 5-tuple keys: `(isPreColonial, size, control, isImproved, treeID)` → `(isPreColonial, size, control, treeID)`
    - If `isImproved=True`, control becomes 'improved-tree'
  - 3-tuple keys: `(isPreColonial, treeID, size)` → `(isPreColonial, size, 'improved-tree', treeID)`
- Saves as `data/revised/revised_tree_dict.pkl`

Source files:
- `adjusted_tree_templates.pkl`: Created by `modules/treeBake_recreateLogs.py`
- `fallen_trees_dict.pkl`: Created by `modules/treeBake_treeAging.py`
