import pyvista as pv
import numpy as np
import f_GetSiteMeshBuildings as gsb
import f_GeospatialModule as gm
import f_SiteCoordinates
from f_photoMesh import get_meshes_in_bounds
import f_segmentation_manager, rGeoVectorGetter
import f_resource_urbanForestParser
import f_resource_distributor_meshes
#from f_segmentation_VTK import process_vtk_meshes

def process_site(site_name):
    # Set up the common parameters
    file_path = "data/revised/experimental/DevelopmentActivityModel-trimmed-metric.glb"
    eastingsDim = 500
    northingsDim = 500

    # Get site coordinates
    easting, northing = f_SiteCoordinates.get_site_coordinates(site_name)
    print(f'processing site: {site_name} with center: {easting}, {northing} and dimensions: {eastingsDim}, {northingsDim}')

    # Get and add photo meshes within bounds
    photo_meshes = get_meshes_in_bounds(easting, northing, eastingsDim, northingsDim, manualSite=site_name)
    
    #Update eastings northings with bounds of returned photomeshes
    easting, northing, eastingsDim, northingsDim = f_SiteCoordinates.get_center_and_dims(photo_meshes)
    point = np.array([easting, northing, 0.0])

    # Get terrain mesh
    print("Getting initial terrain data...")
    terrain_mesh, contours_gdf = gm.handle_contours(easting, northing, eastingsDim, northingsDim)
    print(f"Terrain mesh created with {terrain_mesh.n_points} points and {terrain_mesh.n_cells} cells.")

    print('Get urban forest data')
    uf_gdf = gm.handle_urban_forest(easting, northing, eastingsDim, northingsDim)
    urbanForest_df = f_resource_urbanForestParser.create_urban_forest_df(uf_gdf, terrain_mesh)
    urbanForest_df.to_csv(f'data/revised/{site_name}-tree-locations.csv', index=False)

    print('Distribute resources')
    urbanForestVoxels = f_resource_distributor_meshes.distribute_meshes(urbanForest_df)
    treeVoxelSavePath = f'data/revised/{site_name}-treeVoxels.vtk'
    urbanForestVoxels.save(treeVoxelSavePath)
    print(f"Tree voxels saved to {treeVoxelSavePath}")
    
    print("Processing road data...")
    road_gdf =  gm.handle_road_segments(easting, northing, eastingsDim, northingsDim)
    
    print("Getting road voxels")
    roadVoxels = rGeoVectorGetter.getRoadVoxels(easting, northing, eastingsDim, northingsDim, road_gdf, terrain_mesh, resolution=0.25)
    roadVoxelSavePath = f'data/revised/{site_name}-roadVoxels.vtk'
    roadVoxels.save(roadVoxelSavePath)
    print(f"Road voxels saved to {roadVoxelSavePath}")
    # Get site mesh buildings
    print("Processing site mesh buildings...")
    initial_site_voxels = gsb.process_buildings(file_path, easting, northing, eastingsDim, northingsDim, terrain_mesh, buffer=300)
    print(f"Site mesh created with {initial_site_voxels.n_points} points and {initial_site_voxels.n_cells} cells.")

    segment_site_voxels = f_segmentation_manager.segmentFunction(easting, northing, eastingsDim, northingsDim, photo_meshes, initial_site_voxels, buffer=300)
    segmentedSiteSavePath = f'data/revised/{site_name}-siteVoxels.vtk'
    segment_site_voxels.save(segmentedSiteSavePath)
    print(f"Segmented site voxels saved to {segmentedSiteSavePath}")

    #canopy
    """print("Processing canopy data...")
    canopy_gdf = gm.handle_tree_canopies(easting, northing, eastingsDim, northingsDim)

    #building outlines
    print("Processing building data...")
    footprints_gdf = gm.handle_building_footprints(easting, northing, eastingsDim, northingsDim)

    # Create a plotter
    plotter = pv.Plotter()

    print("Adding meshes to the plot...")
    # Add terrain mesh to the plot
    gm.plot_terrain_mesh(terrain_mesh, plotter, opacity=0.7, show_edges=False, label='Terrain')
    print("Terrain mesh added to the plot.")

    # Add site mesh buildings to the plot
    plotter.add_mesh(initial_site_voxels, scalars='building_ID', opacity=0.5, show_edges=True, label='Site Mesh')
    print("Site mesh added to the plot.")

    # Add a point for the site center
    plotter.add_mesh(pv.PolyData(point), color='blue', point_size=10, render_points_as_spheres=True, label='Site Center')
    plotter.add_point_labels(pv.PolyData(point), ['Site Center'], font_size=24, point_size=1)
    #plotter.add_mesh(roadVoxels, scalars='road_ID', opacity=0.5, show_edges=True, label='Roads')


    
    #print(f"Processing {len(photo_meshes)} photo meshes with segmentation...")
    #processed_meshes = process_vtk_meshes(photo_meshes)
    ""for i, mesh in enumerate(photo_meshes):
        if 'RGB' in mesh.point_data:
            rgb = mesh.point_data['RGB']
            print(f"Mesh {i}: RGB array shape: {rgb.shape}, n_points: {mesh.n_points}")
        else:
            print(f"Mesh {i}: No RGB data found, n_points: {mesh.n_points}")
        
        # Add the processed mesh to the plot
        plotter.add_mesh(mesh, rgb=True)
    

    # Set up the camera
    plotter.camera_position = [
        (easting + eastingsDim/2, northing + northingsDim/2, max(eastingsDim, northingsDim)/2),  # Camera position
        (easting, northing, terrain_mesh.points[:, 2].mean()),  # Focal point
        (0, 0, 1)  # View up direction
    ]

    plotter.add_legend()
    plotter.add_scalar_bar('Elevation', vertical=True)
    
    print("Displaying the combined visualization...")
    plotter.show()"""

def main():
    #sites = ['trimmed-parade', 'uni', 'city', 'street']
    sites = ['city', 'trimmed-parade', 'uni']
    #sites = ['uni']
    #sites = ['city']
    for site in sites:
        process_site(site)
    
if __name__ == "__main__":
    main()