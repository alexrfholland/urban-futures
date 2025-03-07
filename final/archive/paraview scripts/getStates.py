from paraview.simple import *
import os

# ----- Configuration -----
site = 'trimmed-parade'
voxel_size = 1
years_passed = [0, 10, 30, 60, 180]
base_path = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final'

# Template for scenario file names
def get_scenario_filename(year):
    return os.path.join(base_path, f"{site}_{voxel_size}_scenarioYR{year}.vtk")

# Number of scenarios
num_scenarios = len(years_passed)

# ----- Initialize the Render View -----
renderView = GetActiveViewOrCreate('RenderView')
renderView.ResetCamera()

# ----- Function to Load a VTK File and Apply Initial Filters -----
def load_vtk_pipeline(file_path, name_suffix):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    reader = LegacyVTKReader(FileNames=[file_path])
    
    # Since all data is point data, we don't need to set CellArrayStatus
    # Enable all available point arrays
    available_point_arrays = reader.PointArrayStatus
    # If you want to enable specific arrays, set them here
    # For example: reader.PointArrayStatus = ['Temperature', 'Pressure']
    # To enable all point arrays, ensure they are all set to 1
    for array_name in available_point_arrays:
        reader.PointArrayStatus[array_name] = 1

    # Apply Clip filter
    clip = Clip(Input=reader)
    clip.ClipType = 'Box'
    # Initial bounds; will be updated later
    clip.ClipType.Bounds = [0, 1, 0, 1, 0, 1]

    # Apply Transform filter
    transform = Transform(Input=clip)
    # Currently no transformation; set identity or leave as default
    # If needed, adjust transform.Transform properties here

    # Rename proxies for clarity
    reader.Name = f"{name_suffix}_Reader"
    clip.Name = f"{name_suffix}_Clip"
    transform.Name = f"{name_suffix}_Transform"

    return transform

# ----- Load Base Datasets: Roads and Site -----
roads_file = os.path.join(base_path, 'trimmed-parade-roadVoxels-coloured.vtk')
site_file = os.path.join(base_path, 'trimmed-parade-siteVoxels-coloured.vtk')  # Adjust if different

roads_pipeline = load_vtk_pipeline(roads_file, 'Roads')
site_pipeline = load_vtk_pipeline(site_file, 'Site')

if roads_pipeline is None or site_pipeline is None:
    raise FileNotFoundError("Base datasets (Roads or Site) not found. Please check the file paths.")

# ----- Function to Configure Clip Bounds -----
def configure_clip_bounds(pipeline, x_min, x_max):
    if pipeline is None:
        return

    # Access the Clip filter within the pipeline by name
    clip_filter = FindSource(f"{pipeline.GetName().replace('_Transform', '_Clip')}")
    if clip_filter is not None:
        clip_filter.ClipType.Bounds = [x_min, x_max,
                                      clip_filter.ClipType.Bounds[2], clip_filter.ClipType.Bounds[3],
                                      clip_filter.ClipType.Bounds[4], clip_filter.ClipType.Bounds[5]]
    else:
        print(f"Clip filter not found in pipeline: {pipeline.GetName()}")

# ----- Function to Setup Display Properties -----
def setup_display(pipeline, display_name):
    if pipeline is None:
        return
    display = Show(pipeline, renderView)
    display.Name = display_name
    display.Representation = 'Surface'

    # Identify available point data arrays
    data_info = servermanager.Fetch(pipeline)
    point_data = data_info.GetPointData()
    num_arrays = point_data.GetNumberOfArrays()

    if num_arrays > 0:
        # Use the first available point data array for coloring
        scalar_field = point_data.GetArrayName(0)
        display.ColorArrayName = ['POINTS', scalar_field]
        
        # Ensure the color transfer function exists
        color_transfer = GetColorTransferFunction(scalar_field)
        if not color_transfer:
            color_transfer = GetColorTransferFunction(scalar_field)
            color_transfer.ApplyPreset('Viridis', True)
        
        display.LookupTable = color_transfer
    else:
        # If no point data arrays are available, disable coloring
        display.ColorArrayName = [None, '']

    # Additional display properties can be set here as needed
    display.OSPRayScaleArray = scalar_field if num_arrays > 0 else ''
    display.OSPRayScaleFunction = 'PiecewiseFunction'
    display.SelectOrientationVectors = 'None'
    display.ScaleFactor = 1.0  # Adjust as needed
    display.GlyphType = 'Arrow'
    display.GlyphTableIndexArray = 'None'
    display.DataAxesGrid = 'GridAxesRepresentation'
    display.PolarAxes = 'PolarAxesRepresentation'

# ----- Iterate Over Scenarios -----
for year in years_passed:
    scenario_file = get_scenario_filename(year)
    if not os.path.exists(scenario_file):
        print(f"Scenario file not found: {scenario_file}")
        continue

    # Load Scenario VTK
    scenario_pipeline = load_vtk_pipeline(scenario_file, f"Scenario_{year}")
    if scenario_pipeline is None:
        continue

    # Get bounds to determine X range
    data_info = servermanager.Fetch(scenario_pipeline)
    bounds = data_info.GetBounds()
    x_total_min, x_total_max = bounds[0], bounds[1]
    x_range = x_total_max - x_total_min
    x_step = x_range / num_scenarios

    # Calculate bounds for current year
    year_index = years_passed.index(year)
    x_min = x_total_min + year_index * x_step
    x_max = x_min + x_step

    print(f"Year {year}: x_min = {x_min}, x_max = {x_max}")

    # Configure Clip for Scenario
    configure_clip_bounds(scenario_pipeline, x_min, x_max)

    # Clone and configure Roads and Site pipelines
    roads_clone = Clone(roads_pipeline)
    roads_clone.Name = f"Roads_Clip_{year}"
    configure_clip_bounds(roads_clone, x_min, x_max)

    site_clone = Clone(site_pipeline)
    site_clone.Name = f"Site_Clip_{year}"
    configure_clip_bounds(site_clone, x_min, x_max)

    # ----- Apply Additional Filters for Scenario -----
    # Assuming these are specific VTK files per scenario
    trees_file = os.path.join(base_path, f"Trees_scenarioYR{year}.vtk")
    rewilded_file = os.path.join(base_path, f"Rewilded_scenarioYR{year}.vtk")
    under_file = os.path.join(base_path, f"Under_scenarioYR{year}.vtk")

    trees_pipeline = load_vtk_pipeline(trees_file, f"Trees_{year}")
    rewilded_pipeline = load_vtk_pipeline(rewilded_file, f"Rewilded_{year}")
    under_pipeline = load_vtk_pipeline(under_file, f"Under_{year}")

    # ----- Set Display Properties -----
    setup_display(scenario_pipeline, f"Scenario_Display_{year}")
    setup_display(roads_clone, f"Roads_Display_{year}")
    setup_display(site_clone, f"Site_Display_{year}")
    setup_display(trees_pipeline, f"Trees_Display_{year}")
    setup_display(rewilded_pipeline, f"Rewilded_Display_{year}")
    setup_display(under_pipeline, f"Under_Display_{year}")

# ----- Final Render -----
RenderAllViews()
