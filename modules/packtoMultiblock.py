import pyvista as pv
import numpy as np

def generate_random_polydata(num_points: int) -> pv.PolyData:
    """Generate a random point cloud with specified number of points."""
    return pv.PolyData(np.random.rand(num_points, 3))

def create_polydata_dict() -> dict:
    """Create a dictionary of random PolyData objects."""
    polydata_dict = {
        'branches': generate_random_polydata(50),
        'canopy resources': generate_random_polydata(50),
        'ground resources': generate_random_polydata(50),
        'site': generate_random_polydata(50)
    }
    return polydata_dict

def create_or_extend_multiblock(polydata_dict: dict, multi_block: pv.MultiBlock = None, new_block_name: str = None) -> pv.MultiBlock:
    """Create a new MultiBlock or extend an existing MultiBlock with PolyData from a dictionary."""
    if multi_block is None:
        multi_block = pv.MultiBlock()
    
    if new_block_name is not None:  # If a new block name is provided, create a new MultiBlock
        new_multi_block = pv.MultiBlock()
        for name, polydata in polydata_dict.items():
            block_index = new_multi_block.n_blocks  # Get the next block index
            new_multi_block.append(polydata)  # Append the PolyData to the new MultiBlock
            new_multi_block.set_block_name(block_index, name)  # Set the block name

            print(f'Block name set to {new_multi_block.get_block_name(block_index)} in new MultiBlock')

        block_index = multi_block.n_blocks  # Get the next block index for the existing MultiBlock
        multi_block.append(new_multi_block)  # Append the new MultiBlock to the existing MultiBlock
        multi_block.set_block_name(block_index, new_block_name)  # Set the block name for the new MultiBlock
        
        print(f'New MultiBlock name set to {multi_block.get_block_name(block_index)} in existing MultiBlock')

    else:  # If no new block name is provided, insert the PolyData into the existing MultiBlock
        for name, polydata in polydata_dict.items():
            block_index = multi_block.n_blocks  # Get the next block index
            multi_block.append(polydata)  # Append the PolyData to the MultiBlock
            multi_block.set_block_name(block_index, name)  # Set the block name

            print(f'Block name set to {multi_block.get_block_name(block_index)} in existing MultiBlock')

    return multi_block

def create_or_extend_multiblockB(polydata_dict: dict, multi_block: pv.MultiBlock = None) -> pv.MultiBlock:
    """Create a new MultiBlock or extend an existing MultiBlock with PolyData from a dictionary."""
    if multi_block is None:
        multi_block = pv.MultiBlock()
    
    for name, polydata in polydata_dict.items():
        block_index = multi_block.n_blocks  # Get the next block index
        multi_block.append(polydata)  # Append the PolyData to the MultiBlock
        multi_block.set_block_name(block_index, name)  # Set the block name


        print(f'block name set to {multi_block.get_block_name(block_index)}')
    
    return multi_block

def main():
    polydata_dict = create_polydata_dict()
    multi_block = create_or_extend_multiblock(polydata_dict)
    
    # Optionally, extend the existing multi_block with more PolyData
    additional_polydata_dict = {
        'additional block 1': generate_random_polydata(50),
        'additional block 2': generate_random_polydata(50)
    }
    extended_multi_block = create_or_extend_multiblock(additional_polydata_dict, multi_block)
    
    # Now you can use multi_block or extended_multi_block in your visualization
    plotter = pv.Plotter()
    plotter.add_mesh(extended_multi_block)  # Example of how to add the extended_multi_block to a plotter
    plotter.show()

if __name__ == "__main__":
    main()
