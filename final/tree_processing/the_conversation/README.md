# Conversation Viewer

A PyVista-based viewer for visualizing individual tree meshes from VTK files.

## Usage

The viewer provides three main functions:

### 1. View Single Tree
```python
view_single_tree(precolonial=False, size='large', control='park-tree', tree_id=13)
```

### 2. Compare Tree Across Controls
Shows the same tree in reserve, park, and street contexts:
```python
view_tree_comparison(tree_id=13, size='large', precolonial=False)
```

### 3. View Multiple Trees in Grid
```python
tree_specs = [
    (False, 'large', 'park-tree', 13),
    (False, 'medium', 'street-tree', 2),
    (False, 'small', 'reserve-tree', 4),
    (True, 'large', 'park-tree', 12)
]
view_multiple_trees(tree_specs, layout=(2, 2))
```

## Tree Parameters

- **precolonial**: `True` or `False`
- **size**: `'small'`, `'medium'`, `'large'`, `'snag'`, `'senescing'`
- **control**: `'reserve-tree'`, `'park-tree'`, `'street-tree'`
- **tree_id**: Integer ID of the specific tree template

## Resource Colors

The viewer uses the same color scheme as the original treeBake system:
- **perch branch**: green
- **peeling bark**: orange  
- **dead branch**: purple
- **other**: light gray (#C5C5C5)
- **fallen log**: plum
- **leaf litter**: peachpuff
- **epiphyte**: cyan
- **hollow**: magenta
- **leaf cluster**: light green (#d5ffd1)

## File Structure

Tree meshes are expected to be in: `data/revised/final/treeMeshes/`
with naming convention: `precolonial.{bool}_size.{size}_control.{control}_id.{id}.vtk` 