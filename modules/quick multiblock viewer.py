import pyvista as pv

site = 'street'
vtm_name = 'buildings'
vtm_file_path = f'data/{site}/{vtm_name}.vtm'
multi_block = pv.read(vtm_file_path)

for block in multi_block:
    print(f'number of points in block is {block.n_points}')

plotter = pv.Plotter()
plotter.add_mesh(multi_block)  # This line replaces 'add multi_block'
plotter.show()  # This line replaces 'show plotter'
