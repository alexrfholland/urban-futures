import pyvista as pv
import numpy as np

# Function to compute the center and width of boxes with edges aligned to the bounds
def compute_boxes_with_edges(polydata, axis='x', num_boxes=5):
    """
    Divides the specified axis of the polydata bounds into equal-width boxes.

    Parameters:
    - polydata: The PyVista PolyData object.
    - axis: The axis along which to divide ('x', 'y', or 'z').
    - num_boxes: The number of boxes to create.

    Returns:
    - A list of dictionaries containing center positions and widths of the boxes.
    """
    # Get bounds of the polydata (xmin, xmax, ymin, ymax, zmin, zmax)
    bounds = polydata.bounds
    axis_indices = {'x': (0,1), 'y': (2,3), 'z': (4,5)}
    
    if axis not in axis_indices:
        raise ValueError("Axis must be one of 'x', 'y', or 'z'.")
    
    lower, upper = bounds[axis_indices[axis][0]], bounds[axis_indices[axis][1]]
    total_width = upper - lower
    
    if num_boxes <= 0:
        raise ValueError("Number of boxes must be a positive integer.")
    
    # Compute the edges using linspace to ensure precise division
    edges = np.linspace(lower, upper, num_boxes + 1)
    
    # Initialize list to store box centers and widths
    boxes = []
    
    for i in range(num_boxes):
        box_start = edges[i]
        box_end = edges[i+1]
        box_width = box_end - box_start
        center = (box_start + box_end) / 2
        boxes.append({
            'center': center,
            'width': box_width,
            'start': box_start,
            'end': box_end
        })
        print(f"Box {i+1}: Center = {center:.3f}, Width = {box_width:.3f}")
        print(f"         Start = {box_start:.3f}, End = {box_end:.3f}\n")
    
    return boxes

# Example usage
if __name__ == "__main__":
    # Load the polydata file (modify the file path as needed)

    site = 'city'
    #path = f"data/revised/final/{site}-roadVoxels-coloured.vtk"
    path = f"data/revised/final/{site}-siteVoxels-masked.vtk"
    path = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/city-roadVoxels-coloured.vtk'
    polydata = pv.read(path)

    # Print bounds of polydata
    bounds = polydata.bounds
    print(f"Bounds of the polydata:")
    print(f"X: {bounds[0]:.3f} to {bounds[1]:.3f}")
    print(f"Y: {bounds[2]:.3f} to {bounds[3]:.3f}")
    print(f"Z: {bounds[4]:.3f} to {bounds[5]:.3f}")

    # Set axis and number of boxes
    axis = 'y'  # Choose 'x', 'y', or 'z'
    num_boxes = 5  # Number of boxes
    
    # Compute boxes along the specified axis
    boxes = compute_boxes_with_edges(polydata, axis, num_boxes)
    
    # Visualization: Create and display the boxes alongside the polydata
    plotter = pv.Plotter()

    # Add the original polydata to the plot
    plotter.add_mesh(polydata, color='lightgrey', opacity=0.5, label='PolyData')

    # Determine sizes for the boxes based on axis
    bounds = polydata.bounds
    size = {
        'x': (boxes[0]['width'], bounds[5] - bounds[4], bounds[3] - bounds[2]),
        'y': (bounds[1] - bounds[0], boxes[0]['width'], bounds[5] - bounds[4]),
        'z': (bounds[1] - bounds[0], bounds[3] - bounds[2], boxes[0]['width'])
    }

    # Add each box to the plot
    for i, box in enumerate(boxes):
        if axis == 'x':
            center = (box['center'], (bounds[3] + bounds[2])/2, (bounds[5] + bounds[4])/2)
            box_size = size['x']
        elif axis == 'y':
            center = ((bounds[1] + bounds[0])/2, box['center'], (bounds[5] + bounds[4])/2)
            box_size = size['y']
        elif axis == 'z':
            center = ((bounds[1] + bounds[0])/2, (bounds[3] + bounds[2])/2, box['center'])
            box_size = size['z']
        
        box_mesh = pv.Box(center=center, x_length=box_size[0], y_length=box_size[1], z_length=box_size[2])
        plotter.add_mesh(box_mesh, opacity=0.3, label=f'Box {i+1}')
    
    # Add a legend to distinguish the boxes and polydata
    plotter.add_legend([('PolyData', 'lightgrey'),
                        *( (f'Box {i+1}', 'white') for i in range(num_boxes) )])
    
    # Set plot title and show the plot
    plotter.add_title("PolyData with Aligned Boxes")
    plotter.show()
