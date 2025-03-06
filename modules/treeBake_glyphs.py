import numpy as np
import pyvista as pv

# Resource specifications dictionary
"""resource_specs = {
    'perchable branch': {
        'colour': 'green',
        'opacity': 0.1,
        'voxelSize': [0.15, 0.15, 0.15]
    },
    'peeling bark': {
        'colour': 'orange',
        'opacity': 0.1,
        'voxelSize': [0.15, 0.15, 0.15]
    },
    'dead branch': {
        'colour': 'purple',
        'opacity': 0.1,
        'voxelSize': [0.15, 0.15, 0.15]
    },
    'other': {
        #'colour': '#e8e8e8',
        'colour' : '#C5C5C5',
        'opacity': 0.2,
        'voxelSize': [0.15, 0.15, 0.15]
    },
    'fallen log': {
        'colour': 'plum',
        'opacity': 0.1,
        'voxelSize': [.2, .2, .2]
    },
    'leaf litter': {
        'colour': 'peachpuff',
        'opacity': .1,
        'voxelSize': [0.25, 0.25, 0.25]
    },
    'epiphyte': {
        'colour': 'cyan',
        'opacity': 1.0,
        'voxelSize': [0.3, 0.3, 0.3]
    },
    'hollow': {
        'colour': 'magenta',
        'opacity': 1.0,
        'voxelSize': [3, .3, .3]
    },
    'leaf cluster': {
        'colour': '#d5ffd1',
        'opacity': 0.1,
        'voxelSize': [.5, .5, .5]
    }
}
"""
isoSize = [0.15, 0.15, 0.15]

resource_specs = {
    'perchable branch': {
        'colour': 'green',
        'opacity': 0.5,
        'voxelSize': isoSize
    },
    'peeling bark': {
        'colour': 'orange',
        'opacity': 0.5,
        'voxelSize': isoSize
    },
    'dead branch': {
        'colour': 'purple',
        'opacity': 0.5,
        'voxelSize': isoSize
    },
    'other': {
        #'colour': '#e8e8e8',
        'colour' : '#C5C5C5',
        'opacity': 0.5,
        'voxelSize': isoSize
    },
    'fallen log': {
        'colour': 'plum',
        'opacity': 0.5,
        'voxelSize': isoSize
    },
    'leaf litter': {
        'colour': 'peachpuff',
        'opacity': 0.5,
        'voxelSize': [0.25, 0.25, 0.25]
    },
    'epiphyte': {
        'colour': 'cyan',
        'opacity': 1.0,
        'voxelSize': [0.3, 0.3, 0.3]
    },
    'hollow': {
        'colour': 'magenta',
        'opacity': 1.0,
        'voxelSize': [3, .3, .3]
    },
    'leaf cluster': {
        'colour': '#d5ffd1',
        'opacity': 0.1,
        'voxelSize': [.5, .5, .5]
    }
}



def extract_isosurface_from_polydata(polydata: pv.PolyData, spacing: tuple[float, float, float], resource_name, isovalue: float = 1.0) -> pv.PolyData:
    # Extracts an isosurface from PolyData using Marching Cubes
    if polydata is not None and polydata.n_points > 0:
        print(f'{resource_name} polyData has {polydata.n_points} points')
        
        points = polydata.points
        x, y, z = points.T

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        z_min, z_max = z.min(), z.max()

        dims = (
            int((x_max - x_min) / spacing[0]) + 1,
            int((y_max - y_min) / spacing[1]) + 1,
            int((z_max - z_min) / spacing[2]) + 1
        )

        grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=(x_min, y_min, z_min))
        scalars = np.zeros(grid.n_points)

        for px, py, pz in points:
            ix = int((px - x_min) / spacing[0])
            iy = int((py - y_min) / spacing[1])
            iz = int((pz - z_min) / spacing[2])
            grid_idx = ix + iy * dims[0] + iz * dims[0] * dims[1]
            scalars[grid_idx] = 2

        grid.point_data['values'] = scalars
        isosurface = grid.contour(isosurfaces=[isovalue], scalars='values')
        return isosurface
    else:
        print('{resource_name} polyData is empty or None')
        return None

def add_trees_to_plotter(plotter, multi_block: pv.MultiBlock, shift=None):
    def apply_shift(poly_data):
        if shift is not None:
            poly_data.points += shift

    def plot_resource(resource_name, polydata):
        specs = resource_specs[resource_name]

        # Create a shift if the resource is 'leaf litter'
        if resource_name == 'leaf litter':
            # Generate a random shift for the z-coordinate
            z_shift = np.random.uniform(0, 0.3, size=polydata.points.shape[0])
            
            # Initialize the shift array
            leafShift = np.zeros_like(polydata.points)
            
            # Apply the z_shift to the z-axis
            leafShift[:, 2] = z_shift

            polydata.points += leafShift

        apply_shift(polydata)
        isosurface = extract_isosurface_from_polydata(polydata, specs['voxelSize'], resource_name)

        if resource_name in ['hollow','epiphyte']:
            lineCol = '#525252'
            lineWidth = 10
            featureAngle=80
        else:
            lineCol = specs['colour']
            lineWidth = 3
            featureAngle=80

        if isosurface.n_points > 0:
            if resource_name not in ['other', 'leaf litter']:
                plotter.add_mesh(isosurface, 
                                opacity=specs['opacity'],  
                                color=specs['colour'],
                                #silhouette=dict(color=specs['colour'], line_width=3, feature_angle=55),
                                silhouette=dict(color=lineCol, line_width=lineWidth, feature_angle=featureAngle),
                                show_edges=False)
            else:
                plotter.add_mesh(isosurface, 
                                opacity=specs['opacity'],  
                                color=lineCol,
                                silhouette=dict(color=specs['colour'], line_width=3, feature_angle=80),
                                show_edges=False)
        else:
            print(f'Isosurface extraction for {resource_name} resulted in an empty mesh')

            

    for resource_name in multi_block.keys():
        polydata = multi_block.get(resource_name)
        if polydata is not None and polydata.n_points > 0:
            unique_resources = np.unique(polydata.point_data['resource'])
            for res_type in unique_resources:
                mask = polydata.point_data['resource'] == res_type
                resource_polydata = polydata.extract_points(mask)
                plot_resource(res_type, resource_polydata)
        else:
            print(f'No points found for {resource_name}')

if __name__ == '__main__':
    plotter = pv.Plotter()
    multiblock = pv.MultiBlock()
    # Add PolyData objects to the MultiBlock here...
    add_trees_to_plotter(plotter, multiblock, (0, 0, 0))
    plotter.show()
