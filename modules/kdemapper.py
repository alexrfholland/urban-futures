import numpy as np
import pyvista as pv
from scipy.stats import gaussian_kde
from multiprocessing import Pool, cpu_count
import searcher, loader
import numpy as np
from scipy.spatial import KDTree
import cameraSetUpRevised


def classifyTrees(poly_data):
    treeSearch = {
        "1": {  # small
            "AND": [
                {"_Tree_size": ["small"]}
            ]
        },
        "2": {  # medium notisPrecolonial street-tree
            "AND": [
                {"_Tree_size": ["medium"]},
                {"isPrecolonial": [False]},
                {"_Control": ["street-tree"]}
            ]
        },
        "3": {  # medium notisPrecolonial park-tree
            "AND": [
                {"_Tree_size": ["medium"]},
                {"isPrecolonial": [False]},
                {"_Control": ["park-tree"]}
            ]
        },
        "4": {  # medium isPrecolonial street-tree
            "AND": [
                {"_Tree_size": ["medium"]},
                {"isPrecolonial": [True]},
                {"_Control": ["street-tree"]}
            ]
        },
        "5": {  # medium isPrecolonial park-tree
            "AND": [
                {"_Tree_size": ["medium"]},
                {"isPrecolonial": [True]},
                {"_Control": ["park-tree"]}
            ]
        },
        "15": {  # medium isPrecolonial reserve-tree
            "AND": [
                {"_Tree_size": ["medium"]},
                {"isPrecolonial": [True]},
                {"_Control": ["reserve-tree"]}
            ]
        },
        "7": {  # large notisPrecolonial street-tree
            "AND": [
                {"_Tree_size": ["large"]},
                {"isPrecolonial": [False]},
                {"_Control": ["street-tree"]}
            ]
        },
        "8": {  # large notisPrecolonial park-tree
            "AND": [
                {"_Tree_size": ["large"]},
                {"isPrecolonial": [False]},
                {"_Control": ["park-tree"]}
            ]
        },
        "20": {  # large isPrecolonial street-tree
            "AND": [
                {"_Tree_size": ["large"]},
                {"isPrecolonial": [True]},
                {"_Control": ["street-tree"]}
            ]
        },
        "25": {  # large isPrecolonial park-tree
            "AND": [
                {"_Tree_size": ["large"]},
                {"isPrecolonial": [True]},
                {"_Control": ["park-tree"]}
            ]
        },
        "30": {  # large isPrecolonial reserve-tree
            "AND": [
                {"_Tree_size": ["large"]},
                {"isPrecolonial": [True]},
                {"_Control": ["reserve-tree"]}
            ]
        }
    }

    siteWeights = searcher.classify_points_poly(poly_data, treeSearch,unclassified="0")
    siteWeights = np.array(siteWeights).astype(float)
    treeMask = siteWeights != 0

    bounds = poly_data.bounds
    treeCenters = poly_data.points[treeMask]
    treeWeights = siteWeights[treeMask]



    return bounds, treeCenters, treeWeights


def classifyAnyCanopy(poly_data):
    treeSearch = {
        "1": {  # small
            "AND": [
                {"_Tree_size" : ["small", "medium"]}
            ]
        },
        "10": {  # small
            "AND": [
                {"_Tree_size" : ["small", "large"]}
            ]
        },
    }

def classifyAnyCanopy(poly_data):
    treeSearch = {
        "1": {  # small
            "AND": [
                {"_Tree_size" : ["small", "medium"]}
            ]
        },
        "10": {  # small
            "AND": [
                {"_Tree_size" : ["small", "large"]}
            ]
        },
    }

    siteWeights = searcher.classify_points_poly(poly_data, treeSearch,unclassified="0")
    siteWeights = np.array(siteWeights).astype(float)
    print(siteWeights)
    print(np.unique(siteWeights))
    treeMask = siteWeights != 0

    bounds = poly_data.bounds
    treeCenters = poly_data.points[treeMask]
    treeWeights = siteWeights[treeMask]

    return bounds, treeCenters, treeWeights




import numpy as np
from scipy.stats import gaussian_kde
from multiprocessing import Pool, cpu_count


def getDeployableGradient(poly_data):

    #Any canopy KDE
    search = {
        "1": {  # small
            "NOT": {"fortifiedStructures" : ["unassigned"]}
        },
    }

    #KDE for defensive structures
    bounds, centers, weights = doSearch(poly_data, search)
    kde, grid_points, kdvalues = weighted_2d_kde(bounds, centers, weights, 1,bandwidth_factor=.2)
    poly_data = assign_weights_to_polydata(poly_data, grid_points, kdvalues, 'fortifiedStructures')

    """plotter = pv.Plotter()
    cube = pv.Cube()  # Create a cube geometry for glyphing
    glyphs = polysite.glyph(geom=cube, scale=False, orient=False, factor=1)
    plotter.add_mesh(glyphs, scalars=f'fortifiedStructures-weights_log', cmap='viridis')


    # Settings for better visualization
    plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.5  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()
    plotter.show()
    """
    #Any canopy KDE
    canopySearch = {
        "1": {  # small
            "AND": [
                {"_Tree_size" : ["small", "medium"]}
            ]
        },
        "10": {  # small
            "AND": [
                {"_Tree_size" : ["small", "large"]}
            ]
        },
    }
    bounds, centers, weights = doSearch(poly_data, canopySearch)
    kde, grid_points, kdvalues = weighted_2d_kde(bounds, centers, weights, 1)
    poly_data = assign_weights_to_polydata(poly_data, grid_points, kdvalues, 'canopy')

    """plotter = pv.Plotter()
    cube = pv.Cube()  # Create a cube geometry for glyphing
    glyphs = polysite.glyph(geom=cube, scale=False, orient=False, factor=1)
    plotter.add_mesh(glyphs, scalars=f'canopy-weights_log', cmap='viridis')


    # Settings for better visualization
    plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.5  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()
    plotter.show()"""


    #get combinedscore
    #geometric_mean = np.sqrt(poly_data.point_data['fortifiedStructures-weights_log'] * poly_data.point_data['canopy-weights_log'] )

    max = np.maximum(poly_data.point_data['fortifiedStructures-weights_log'], poly_data.point_data['canopy-weights_log'])

    poly_data.point_data['offensiveScore'] = 1 - max

    from utilities.getCmaps import create_colormaps        
    colormaps = create_colormaps()


    plotter = pv.Plotter()
    cameraSetUpRevised.setup_camera(plotter, 50, 600)
    cube = pv.Cube()  # Create a cube geometry for glyphing
    glyphs = polysite.glyph(geom=cube, scale=False, orient=False, factor=1.5)
    plotter.add_mesh(glyphs, scalars=f'defensiveScore', cmap=colormaps['div5-asym-orange-blue'], flip_scalars=True)
    #plotter.add_mesh(glyphs, scalars=f'offensiveScore', cmap=colormaps['other-outl-5'])


    # Settings for better visualization
    plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.5  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()
    plotter.show()

    return polysite


def doSearch(poly_data, search, unclassified="0"):
    siteWeights = searcher.classify_points_poly(poly_data, search,unclassified="0")
    siteWeights = np.array(siteWeights).astype(float)
    print(siteWeights)
    print(np.unique(siteWeights))
    mask = siteWeights != 0

    bounds = poly_data.bounds
    centers = poly_data.points[mask]
    weights = siteWeights[mask]

    return bounds, centers, weights



def kde_evaluate(kde, centers):
    return kde.evaluate(centers.T)

def compute_bandwidth_factor(data, weights, method='scott', factor=0.5):
    kde_temp = gaussian_kde(data, weights=weights, bw_method=method)
    bandwidth = kde_temp.factor * factor
    return bandwidth

def create_evaluation_grid_2d(bounds, cell_size):
    min_x, max_x, min_y, max_y = bounds
    x_grid = np.arange(min_x, max_x, cell_size)
    y_grid = np.arange(min_y, max_y, cell_size)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T
    return grid_points

def weighted_2d_kde(bounds, points, weights, cell_size, bandwidth_factor=0.1):
    # Flatten points to 2D
    points_2d = points[:, :2]

    # Compute bandwidth factor
    bandwidth = compute_bandwidth_factor(points_2d.T, weights, method='scott', factor=bandwidth_factor)

    # Create gaussian_kde object
    kde = gaussian_kde(points_2d.T, weights=weights, bw_method=bandwidth)

    # Create evaluation grid
    grid_bounds = bounds[:4]  # Using only the X and Y bounds
    grid_points = create_evaluation_grid_2d(grid_bounds, cell_size)

    # Evaluate KDE for each point in the grid
    kdvalues = kde_evaluate(kde, grid_points)

    return kde, grid_points, kdvalues

import matplotlib.pyplot as plt
import numpy as np

def plot_kde(grid_points, kdvalues, plot_type='scatter', figsize=(10, 8), cmap='viridis'):
    """
    Plot the KDE values on a 2D grid.

    Parameters:
    - grid_points: numpy.ndarray of grid points (shape: Nx2).
    - kdvalues: numpy.ndarray of KDE values for each grid point.
    - plot_type: str, either 'scatter' or 'contour' for the type of plot.
    - figsize: tuple, size of the figure.
    - cmap: str, colormap to be used.

    Returns:
    - matplotlib figure object
    """
    plt.figure(figsize=figsize)
    
    if plot_type == 'scatter':
        sc = plt.scatter(grid_points[:, 0], grid_points[:, 1], c=kdvalues, cmap=cmap)
        plt.colorbar(sc, label='KDE Value')
    
    elif plot_type == 'contour':
        x_len = len(np.unique(grid_points[:, 0]))
        y_len = len(np.unique(grid_points[:, 1]))
        Z = kdvalues.reshape(x_len, y_len).T
        cp = plt.contourf(grid_points[:, 0].reshape(x_len, y_len), 
                          grid_points[:, 1].reshape(x_len, y_len), 
                          Z, 
                          levels=100, 
                          cmap=cmap)
        plt.colorbar(cp, label='KDE Value')

    else:
        raise ValueError("plot_type must be either 'scatter' or 'contour'.")

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('2D KDE Plot')

    plt.show()
    
    return plt

def assign_weights_to_polydata(polydata, grid_points, grid_weights, name, isFlip = False):
    """
    Assign weights to PolyData points based on the nearest grid point.

    Parameters:
    - polydata: PyVista PolyData object containing points to which weights will be assigned.
    - grid_points: numpy.ndarray of grid points (Nx2 or Nx3).
    - grid_weights: numpy.ndarray of weights corresponding to each grid point.

    Returns:
    - Modified PyVista PolyData with added 'tree-weight' point data.
    """

    # Extract the coordinates from PolyData
    polydata_points = polydata.points[:, :grid_points.shape[1]]  # Truncate to match grid points dimension

    # Create a KDTree with the grid points
    tree = KDTree(grid_points)

    # Find the nearest grid point for each PolyData point
    _, nearest_indices = tree.query(polydata_points)

    weights = grid_weights[nearest_indices]

    # Assign corresponding weights to PolyData points
    
    # Min-Max Normalization
    weights_minmax = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    

    

    # Quartile Normalization
    Q1 = np.percentile(weights, 25)
    Q3 = np.percentile(weights, 75)
    IQR = Q3 - Q1
    weights_quartile = (weights - Q1) / IQR
    

    # Z-Score Normalization
    mean_weight = np.mean(weights)
    std_weight = np.std(weights)
    weights_zscore = (weights - mean_weight) / std_weight

    # Logarithmic Scaling
    log_weights = np.log(weights + 1e-5)  # To avoid log(0)
    weights_log = (log_weights - np.min(log_weights)) / (np.max(log_weights) - np.min(log_weights))
    

    # Robust Scaling
    median_weight = np.median(weights)
    weights_robust = (weights - median_weight) / IQR

    # Assuming weights_minmax, weights_quartile, etc., are your arrays
    
    if isFlip:
        weightsList = [weights_minmax, weights_quartile, weights_zscore, weights_log, weights_robust]
        weights = [1 - weight for weight in weights]
        weights_minmax, weights_quartile, weights_zscore, weights_log, weights_robust = weightsList

    # Unpack the list back into individual variables if needed


    polydata.point_data[f'{name}-weights_minmax'] = weights_minmax
    polydata.point_data[f'{name}-weights_quartile'] = weights_quartile
    polydata.point_data[f'{name}-weights_quartile'] = weights_quartile
    polydata.point_data[f'{name}-weights_log'] = weights_log
    polydata.point_data[f'{name}-weights_zscore'] = weights_zscore
    polydata.point_data[f'{name}-weights_robust'] = weights_robust


    return polydata

# Example usage
# plot = plot_kde(grid_points, kdvalues, plot_type='scatter')
# plot.show()

def getTreeWeights(polysite, name):
    bounds, treeCentres, treeWeights = classifyTrees(polysite)

    kde, grid_points, kdvalues = weighted_2d_kde(bounds, treeCentres, treeWeights, 1,bandwidth_factor=.5)

    #plot_kde(grid_points, kdvalues)

    polysite = assign_weights_to_polydata(polysite, grid_points, kdvalues, name)



    """for weightName in ['weight', 'weights_minmax', 'weights_quartile', 'weights_zscore', 'weights_log', 'weights_robust']:
    
        plotter = pv.Plotter()
        cube = pv.Cube()  # Create a cube geometry for glyphing
        glyphs = polysite.glyph(geom=cube, scale=False, orient=False, factor=1)
        plotter.add_mesh(glyphs, scalars=f'{name}-{weightName}', cmap='viridis')


        # Settings for better visualization
        plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
        light2 = pv.Light(light_type='cameralight', intensity=.5)
        light2.specular = 0.5  # Reduced specular reflection
        plotter.add_light(light2)
        plotter.enable_eye_dome_lighting()
        plotter.show()"""

    return polysite


def getCanopyWeights(polysite, name):
    bounds, treeCentres, treeWeights = classifyAnyCanopy(polysite)
    kde, grid_points, kdvalues = weighted_2d_kde(bounds, treeCentres, treeWeights, 1)

    

    #plot_kde(grid_points, kdvalues)

    polysite = assign_weights_to_polydata(polysite, grid_points, kdvalues, name)

    #for weightName in ['weights_minmax', 'weights_quartile', 'weights_zscore', 'weights_log', 'weights_robust']:
    weightName = 'weights_log'
    
    """plotter = pv.Plotter()
    cube = pv.Cube()  # Create a cube geometry for glyphing
    glyphs = polysite.glyph(geom=cube, scale=False, orient=False, factor=1)
    plotter.add_mesh(glyphs, scalars=f'{name}-{weightName}', cmap='viridis')


    # Settings for better visualization
    plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.5  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()
    plotter.show()"""

    return polysite


if __name__ == "__main__":
    #sites = ['city', 'street', 'trimmed-parade']  # List of sites
    #sites = ['city']
    #sites = ['trimmed-parade']
    #sites = ['trimmed-parade']
    #sites = ['parade']
    #sites = ['trimmed-parade']
    sites = ['city', 'trimmed-parade','street']
    sites = ['city']
    #sites = ['city']
    #sites = ['city']
    #sites = ['city']
    sites = ['parade']
    states = ['potential']
    #states = ['baseline']
    polydata = loader.getSite(sites, states)[0]

    print(polydata)

    for i, polysite in enumerate(polydata):
        site = sites[i]
        poly = getTreeWeights(polysite, 'tree')
        poly = getDeployableGradient(polysite)
        polysite.save(f'data/{site}/offesnivePrep-{site}.vtk')
        


    print('done')





