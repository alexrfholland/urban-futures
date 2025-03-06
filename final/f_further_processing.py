import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree
import fCamera as fCamera
import vtk
import pandas as pd
import f_SiteCoordinates, f_transferShapeFilestoVTK
import f_GeospatialModule
import matplotlib.pyplot as plt

def getSpatials(siteVTK):
    easting, northing, eastingsDim, northingsDim = f_SiteCoordinates.get_center_and_dims([siteVTK])

    greenRoofGDF, brownRoofGDF = f_GeospatialModule.handle_green_roofs(easting, northing, eastingsDim, northingsDim)

    print('Transferring green roofs...')
    siteVTK = f_transferShapeFilestoVTK.assign_gdf_attributes_to_polydata(
        polydata=siteVTK,
        gdf=greenRoofGDF,
        attribute_prefix='greenRoof_'
    )
    print('Green roof transfer complete!')

    print('Transferring brown roofs...')
    siteVTK = f_transferShapeFilestoVTK.assign_gdf_attributes_to_polydata(
        polydata=siteVTK,
        gdf=brownRoofGDF,
        attribute_prefix='brownRoof_'
    )
    print('Brown roof transfer complete!')

    f_transferShapeFilestoVTK.systematic_checks(siteVTK, attribute_prefixes=['greenRoof_', 'brownRoof_'])

    
    return siteVTK

def getTerrainSpatials(roadPoly, site):
    easting, northing, eastingsDim, northingsDim = f_SiteCoordinates.get_center_and_dims([roadPoly])
    
    print("Handling other road info...")
    gdfLittleStreets, gdfLaneways, gdfOpenSpace, gdfPrivateEmptySpace = f_GeospatialModule.handle_other_road_info(easting, northing, eastingsDim, northingsDim)
    
    if site == 'city':
        #print if any of gdfLittleStreets, gdfLaneways, gdfOpenSpace, gdfPrivateEmptySpace is None
        print("Assigning little streets attributes...")
        roadPoly = f_transferShapeFilestoVTK.assign_gdf_attributes_to_polydata(
            polydata=roadPoly,
            gdf=gdfLittleStreets,
            attribute_prefix='terrainInfo_',
            binaryMask='isLittleStreet'
        )
        print("Little streets attributes assigned successfully.")

        print("Assigning lanyway attributes...")
        roadPoly = f_transferShapeFilestoVTK.assign_gdf_attributes_to_polydata(
            polydata=roadPoly,
            gdf=gdfLaneways,
            attribute_prefix='terrainInfo_'
            )
        print("Lanyways assigned successfully.")

    print("Assigning open space attributes...")
    roadPoly = f_transferShapeFilestoVTK.assign_gdf_attributes_to_polydata(
        polydata=roadPoly,
        gdf=gdfOpenSpace,
        attribute_prefix='terrainInfo_',
        binaryMask='isOpenSpace'
    )
    print("Open space attributes assigned successfully.")

    print("Assigning private empty space attributes (with binary mask)...")
    roadPoly = f_transferShapeFilestoVTK.assign_gdf_attributes_to_polydata(
        polydata=roadPoly,
        gdf=gdfPrivateEmptySpace,
        attribute_prefix='terrainInfo_',
        binaryMask='gdfPrivateEmptySpace'
    )
    print("Private empty space attributes (with binary mask) assigned successfully.")

    print("Handling road corridors...")
    gdfRoadCorridors = f_GeospatialModule.handle_road_corridors(easting, northing, eastingsDim, northingsDim)

    print("Assigning road corridor attributes...")
    roadPoly = f_transferShapeFilestoVTK.assign_gdf_attributes_to_polydata(
        polydata=roadPoly,
        gdf=gdfRoadCorridors,
        attribute_prefix='terrainInfo_',
    )

    print("Handling parking...")
    gdfParking, gdfParkingBuffer = f_GeospatialModule.handle_parking(easting, northing, eastingsDim, northingsDim)
    roadPoly = f_transferShapeFilestoVTK.assign_gdf_attributes_to_polydata(
        polydata=roadPoly,
        gdf=gdfParking,
        attribute_prefix='terrainInfo_',
        binaryMask='isParking'
    )

    print("Handling parking median 3m buffer...")
    roadPoly = f_transferShapeFilestoVTK.assign_gdf_attributes_to_polydata(
        polydata=roadPoly,
        gdf=gdfParkingBuffer,
        attribute_prefix='terrainInfo_',
        binaryMask='isParkingMedian3mBuffer'
    )

    print("Road corridor attributes assigned successfully.")

    return roadPoly

def getCanopies(siteVTK,roadPoly):
    import f_GeospatialModule
    easting, northing, eastingsDim, northingsDim = f_SiteCoordinates.get_center_and_dims([siteVTK])
    canopyGDF = f_GeospatialModule.handle_tree_canopies(easting, northing, eastingsDim, northingsDim)
    print('transferring canopy information...')
    siteVTK = f_GeospatialModule.assign_gdf_attributes_to_polydata(siteVTK, canopyGDF, 'canopy_', binaryMask='isCanopy')
    roadPoly = f_GeospatialModule.assign_gdf_attributes_to_polydata(roadPoly, canopyGDF, 'canopy_', binaryMask='isCanopy')
    return siteVTK, roadPoly


infoDF = pd.DataFrame(columns=['site', 'centroid'])

# List of sites
#sites = ['uni', 'city', 'trimmed-parade']
sites = ['trimmed-parade', 'city','uni']
#sites = ['city']

# Loop through each site and process the corresponding VTK files
for site in sites:
    print(f"Processing site: {site}")

    centre = f_SiteCoordinates.get_site_coordinates(site)
    midpoint = np.array([centre[0], centre[1], 0])
    print(f"midpoint for {site} is {midpoint}")
    translation_vector = -midpoint
    print(f"Translation vector: {translation_vector}")
    
    # Load the VTK files for the current site
    site_vtk_path = f'data/revised/{site}-siteVoxels.vtk'
    road_vtk_path = f'data/revised/{site}-roadVoxels.vtk'
    tree_vtk_path = f'data/revised/{site}-treeVoxels.vtk'

    print("Loading VTK files...")
    site_vtk = pv.read(site_vtk_path)
    roadPoly = pv.read(road_vtk_path)

    roadPoly = getTerrainSpatials(roadPoly, site)




    treePoly = pv.read(tree_vtk_path)
    print("VTK files loaded successfully.")

    site_vtk, roadPoly = getCanopies(site_vtk, roadPoly)
    

    # List available point data arrays in site_vtk
    print(f"Available point data arrays in site_vtk:")
    for array_name in site_vtk.point_data.keys():
        print(f"  - {array_name}")

    print(f"Available point data arrays in tree vtk:")
    for array_name in treePoly.point_data.keys():
        print(f"  - {array_name}")

    print(f"Available point data arrays in road vtk:")
    for array_name in roadPoly.point_data.keys():
        print(f"  - {array_name}")

    # Step 1: Create a 2D cKDTree for roadPoly using only X and Y coordinates
    print("Creating 2D cKDTree for roadPoly...")
    road_xy = roadPoly.points[:, :2]  # Extract X, Y coordinates from roadPoly
    road_tree_2d = cKDTree(road_xy)
    print("2D cKDTree created.")

    # Step 2: Query the 2D cKDTree using the X and Y coordinates of site_vtk points
    print("Querying nearest roadPoly points for site_vtk points...")
    site_xy = site_vtk.points[:, :2]  # Extract X, Y coordinates from site_vtk
    distances, nearest_indices = road_tree_2d.query(site_xy)  # Find nearest roadPoly points
    print(f"Found nearest points for {site_vtk.n_points} site_vtk points.")

    # Step 3: Calculate Contours_HeightAboveGround as the difference in Z values
    road_z = roadPoly.points[nearest_indices, 2]  # Get Z values of the nearest roadPoly points
    site_z = site_vtk.points[:, 2]  # Get Z values of the site_vtk points

    contours_height = site_z - road_z  # Calculate the height difference

    # Step 4: Store the result in a new point_data array in site_vtk
    site_vtk.point_data['Contours_HeightAboveGround'] = contours_height
    print("Contours_HeightAboveGround assigned successfully.")
    print(f"Contours_HeightAboveGround range: {np.min(contours_height)} to {np.max(contours_height)}")

    treeMask = (site_vtk.point_data['canopy_isCanopy'] == True) & (site_vtk.point_data['Contours_HeightAboveGround'] > 5)
    # Initialize site_vtk.point_data['canopy_isCanopyCorrected'] as False
    site_vtk.point_data['canopy_isCanopyCorrected'] = np.zeros(site_vtk.n_points, dtype=bool)
    site_vtk.point_data['canopy_isCanopyCorrected'][treeMask] = True
    
    # Apply filtering based on masks
    print("Applying masks to filter points in site_vtk...")
    mask1 = site_vtk.point_data['LAS_Classification'] == -5
    mask2 = site_vtk.point_data['building_distance'] < 5
    mask3 = (site_vtk.point_data['Contours_HeightAboveGround'] > 10) & (site_vtk.point_data['building_distance'] < 5)
    mask4 = (site_vtk.point_data['Contours_HeightAboveGround'] > 20) & (site_vtk.point_data['building_distance'] < 15)
    mask5 = (site_vtk.point_data['Contours_HeightAboveGround'] > 30)
    
    maskOTHER = (site_vtk.point_data['Contours_HeightAboveGround'] > 0.5) & (site_vtk.point_data['canopy_isCanopyCorrected'] == False)

    combined_mask = (mask1 | (mask2 | mask3) | mask3 | mask4 | mask5) & maskOTHER


    # Extract the filtered points
    new_points = site_vtk.points[combined_mask]
    print(f"Filtered points: {new_points.shape[0]} points selected out of {site_vtk.points.shape[0]}")

    # Create new polydata with filtered points and copy data arrays
    filtered_site_voxels = pv.PolyData(new_points)
    for array_name in site_vtk.point_data.keys():
        filtered_site_voxels.point_data[array_name] = site_vtk.point_data[array_name][combined_mask]
    print("New polydata created with filtered points and corresponding data arrays.")
    
    
    
   

    # BUILDING PROPERTIES####
    #get spatials
    filtered_site_voxels  = getSpatials(filtered_site_voxels)
    print(f"Available point data arrays in filtered_site_voxels:")
    for array_name in filtered_site_voxels.point_data.keys():
        print(f"  - {array_name}")

    # Initialize point_data variables
    print("Initializing point_data variables...")
    filtered_site_voxels.point_data['building_isBuilding'] = np.full(filtered_site_voxels.n_points, True, dtype=bool)  # Initialize all as True

    # Initialize 'building_element' as a NumPy array filled with 'facade'
    building_element_np = np.full(filtered_site_voxels.n_points, 'facade', dtype=object)  # Initialize as 'facade'

    # Calculate the dip for building voxels
    print("Calculating building dip...")
    building_normals_x = filtered_site_voxels.point_data['building_normalX']
    building_normals_y = filtered_site_voxels.point_data['building_normalY']
    building_normals_z = filtered_site_voxels.point_data['building_normalZ']

    # Dip calculation (using arctan2 to compute the angle from the horizontal plane, in degrees)
    building_dip = np.rad2deg(np.arctan2(np.sqrt(building_normals_x**2 + building_normals_y**2), building_normals_z))
    filtered_site_voxels.point_data['building_dip'] = building_dip

    # Calculate and print the range of building_dip
    dip_min = np.nanmin(building_dip)
    dip_max = np.nanmax(building_dip)
    print(f"Range of building_dip: {dip_min} to {dip_max}")
    print(f"Dip type: {type(building_dip)}, Sample values: {building_dip[:5]}")

    # Assign 'facade' or 'roof' based on dip angle
    print("Assigning building elements ('facade' or 'roof')...")
    roof_mask = building_dip < 20  # Roofs are defined as having dip < 20 degrees

    # Apply roof_mask to assign 'roof'
    building_element_np[roof_mask] = 'roof'

    # Assign the updated building_element back to point_data
    filtered_site_voxels.point_data['building_element'] = building_element_np

    # Print unique elements in building_element and their counts
    unique_elements, counts = np.unique(filtered_site_voxels.point_data['building_element'], return_counts=True)
    print("Unique elements in building_element and their counts:")
    for element, count in zip(unique_elements, counts):
        print(f"  - {element}: {count}")

    print('adjusting green roof stats...')


    # Step 1: Get list of all attributes in filtered_site_voxels.point_data that begin with 'greenRoof' or 'brownRoof'
    roofAttributes = [attr for attr in filtered_site_voxels.point_data.keys() if attr.startswith('greenRoof') or attr.startswith('brownRoof')]

    print(f"Identified roof attributes: {roofAttributes}")

    # Step 2: Iterate over roofAttributes and check their dtype
    for roofAttributeName in roofAttributes:
        roofData = filtered_site_voxels.point_data[roofAttributeName]
        dtype = roofData.dtype
        
        if np.issubdtype(dtype, np.integer):
            adjustedRoofData = np.full_like(roofData, -1)
        else:
            adjustedRoofData = np.full_like(roofData, np.nan if dtype.kind == 'f' else None)
        
        adjustedRoofData[roof_mask] = roofData[roof_mask]
        filtered_site_voxels.point_data[roofAttributeName] = adjustedRoofData

    ##GET BUILDING FACADES
    
    print("Building-related attributes set successfully.")

    #print unique values and counts in greenRoof_ratingInt
    print(f"Unique values in greenRoof_ratingInt: {np.unique(filtered_site_voxels.point_data['greenRoof_ratingInt'])}")
    print(f"Counts of unique values in greenRoof_ratingInt: {np.unique(filtered_site_voxels.point_data['greenRoof_ratingInt'], return_counts=True)}")
    #filtered_site_voxels.plot(scalars = 'greenRoof_ratingInt')

    #####RGB
    # Create a new PolyData object called canopyVTK with points where canopy_isCanopy is True
    canopy_mask = site_vtk.point_data['canopy_isCanopyCorrected']
    canopyVTK = site_vtk.extract_points(canopy_mask)
    nonCanopyVTK = site_vtk.extract_points(~canopy_mask)

    #ROADS

    # Assign the RGB color of the nearest site_vtk point to each point in roadPoly
    print("Assigning nearest RGB color to each roadPoly point...")
    print("Building KD-tree from nonCanopyVTK points...")
    nonCanopy_tree = cKDTree(nonCanopyVTK.points)
    print("KD-tree built successfully.")
    print(f'finding for {roadPoly.points.shape[0]} points')
    distances, indices = nonCanopy_tree.query(roadPoly.points)
    roadPoly.point_data['colors'] = nonCanopyVTK.point_data['colors'][indices]
    print("Colors assigned successfully.")

    
    



    ##TREES
    print("Assigning colours to tree voxels")
    # Build a KD-tree with the site_vtk points
    print("Building KD-tree from canopyVTK points...")
    canopy_tree = cKDTree(canopyVTK.points)
    print("KD-tree built successfully.")
    print(f'finding for {treePoly.points.shape[0]} points')
    distances, indices = canopy_tree.query(treePoly.points)
    treePoly.point_data['colors'] = canopyVTK.point_data['colors'][indices]

    

    # Find the centroid of the filtered_site_voxels polydata (I am using pyvista)
    centroid = filtered_site_voxels.center

    infoDF.at[len(infoDF), 'site'] = site
    infoDF.at[len(infoDF), 'centroid'] = centroid
    # Apply the translation to all points in new_polydata and roadPoly
    
    filtered_site_voxels.points += translation_vector
    roadPoly.points += translation_vector
    treePoly.points += translation_vector
    canopyVTK.points += translation_vector

    print("Points translated successfully.")



    ########split roadVtk into roadVtk and terrainUnderBuildingVtk
    """print("Creating 2D cKDTree for buildings...")
    filtered_site_voxels_xy = filtered_site_voxels.points[:, :2]
    filtered_site_voxels_xy_Tree = cKDTree(filtered_site_voxels_xy)
    
    print(f"Finding nearest points for {roadPoly.points.shape[0]} road points...")
    distances, indices = filtered_site_voxels_xy_Tree.query(roadPoly.points[:, :2])
    
    print("Creating mask for road points...")
    roadMask = distances < .25
    
    print("Splitting roadPoly into roadVtk and terrainUnderBuildingVtk...")
    roadPoly = roadPoly.extract_points(roadMask)
    terrainUnderBuildingVtk = roadPoly.extract_points(~roadMask)

    print(f"Split complete. roadVtk: {roadPoly.n_points} points, terrainUnderBuildingVtk: {terrainUnderBuildingVtk.n_points} points")


    roadPoly.plot()
    terrainUnderBuildingVtk.plot()
    """

    # Create the 'final' folder inside 'data/revised'
    import os

    final_folder_path = "data/revised/final"
    
    # Check if the folder already exists
    if not os.path.exists(final_folder_path):
        # If it doesn't exist, create it
        os.makedirs(final_folder_path)
        print(f"Created folder: {final_folder_path}")
    else:
        print(f"Folder already exists: {final_folder_path}")

    # Update the save paths to use the new 'final' folder
    filtered_site_voxels.save(f"{final_folder_path}/{site}-siteVoxels-masked.vtk")
    canopyVTK.save(f"{final_folder_path}/{site}-canopyVoxels.vtk")
    roadPoly.save(f"{final_folder_path}/{site}-roadVoxels-coloured.vtk")
    treePoly.save(f"{final_folder_path}/{site}-treeVoxels-coloured.vtk")
    #terrainUnderBuildingVtk.save(f"{final_folder_path}/{site}-terrainUnderBuildingVoxels.vtk")

    print(f"Saved VTK files to {final_folder_path}")
  

output_csv_filepath = "data/revised/test_site_info.csv"  # You can customize this path
infoDF.to_csv(output_csv_filepath, index=False)

