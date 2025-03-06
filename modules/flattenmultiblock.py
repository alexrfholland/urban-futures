import numpy as np
import pyvista as pv



def flattenBlock(multi_block):

    # Initialize an empty list to store the combined points
    combined_points = []
    # Initialize a dictionary to store attributes
    combined_attrs = {}
    total_points = 0

    # First loop to calculate the total number of points
    for i, block in enumerate(multi_block):
        if block is not None and block.n_points > 0:
            if block.n_cells == 0:
                print(f"Skipping block {i} as it contains 0 cells.")
                continue
            total_points += block.n_points

    # Initialize attribute arrays with NaN or empty strings, depending on dtype
    for i, block in enumerate(multi_block):
        if isinstance(block, pv.PolyData) and block.n_points > 0:
            for name, array in block.point_data.items():
                if name not in combined_attrs:
                    # Initialize with the correct shape and type
                    shape = (total_points,) if len(array.shape) == 1 else (total_points, array.shape[1])
                    init_value = np.nan if np.issubdtype(array.dtype, np.number) else ''
                    combined_attrs[name] = np.full(shape, init_value, dtype=array.dtype)


    # Initialize start_idx
    start_idx = 0 

    # Populate combined_points and Update attributes
    for i, block in enumerate(multi_block):
        if isinstance(block, pv.PolyData) and block.n_points > 0:
            if block.n_cells == 0:
                print(f"Skipping block {i} as it contains 0 cells.")
                continue
            
            # Append points from this block to combined_points
            combined_points.extend(block.points.tolist())
            
            end_idx = start_idx + block.n_points  # Ending index for updating attribute arrays
            for name, array in block.point_data.items():
                # Update the relevant slice of the destination array
                combined_attrs[name][start_idx:end_idx] = array

            start_idx = end_idx  # Update the starting index for the next iteration

    # Convert combined points to a NumPy array
    combined_points = np.array(combined_points)

    # Create a new PolyData object from the combined points
    combined_polydata = pv.PolyData(combined_points)

    # Add the attributes back to the combined PolyData
    for name, values in combined_attrs.items():
        combined_polydata.point_data[name] = values

    # Now, combined_polydata should contain all points and attributes from all individual blocks
    print(f'flattened polydata block contains {combined_polydata.point_data}')
    
    
    
    return combined_polydata


if __name__ == "__main__":
    # Read the VTM file
    site = 'city'
    multi_block = pv.read(f'data/{site}/{site}.vtm')
    combined_polydata = flattenBlock(multi_block)
    combined_multi_block = pv.MultiBlock()
    combined_multi_block.append(combined_polydata)

    # Save the MultiBlock dataset to a VTK file
    combined_multi_block.save(f'data/{site}/flattened-{site}.vtm')
    combined_polydata.save(f'data/{site}/flattened-{site}.vtk')
    

