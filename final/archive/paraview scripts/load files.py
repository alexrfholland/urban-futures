import os
import json
from paraview.simple import *

# Set the necessary variables
site = 'trimmed-parade'
voxel_size = 1
year = 60  # Testing for year 60 as requested
folderPath = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/trimmed-parade'

# Function to get the state file name
def get_state_file(year):
    file_path = f'{folderPath}/{site}_{voxel_size}_scenarioYR{year}.vtk'
    if os.path.exists(file_path):
        return file_path
    else:
        print(f"File not found: {file_path}")
        return None

# Function to load display settings from JSON
def load_all_display_settings(source, filename):
    with open(filename, 'r') as f:
        settings = json.load(f)
    
    display = GetDisplayProperties(source)
    for prop, value in settings.items():
        try:
            setattr(display, prop, value)
        except Exception as e:
            print(f"Could not apply property: {prop} - {e}")

# Function to apply a threshold filter and rename it
def apply_threshold(source, scalars_name, lower_bound, upper_bound, name):
    threshold = Threshold(Input=source)
    threshold.Scalars = ['POINTS', scalars_name]
    threshold.LowerThreshold = lower_bound
    threshold.UpperThreshold = upper_bound
    threshold.ThresholdMethod = 'Between'
    RenameSource(name, threshold)  # Rename the threshold for better clarity in the pipeline
    return threshold

# Load the state file for year 60
state_file = get_state_file(year)
if state_file is None:
    raise FileNotFoundError("The state file for year 60 could not be found.")

# Load the data using LegacyVTKReader and rename it
state_data = LegacyVTKReader(FileNames=[state_file])
RenameSource(f'{site}_YR{year}', state_data)  # Rename the loaded source to include the year

# Apply thresholds for Trees, Rewilded, and Under-tree treatments, with proper naming
trees_threshold = apply_threshold(state_data, 'maskforTrees', 1, 1, f'Trees_YR{year}')
rewilded_threshold = apply_threshold(state_data, 'scenario_rewildingEnabled', 0, 1e9, f'Rewilded_YR{year}')
under_tree_treatments_threshold = apply_threshold(state_data, 'voxel_I', 0, 1e9, f'UnderTreeTreatments_YR{year}')

# Load display settings for each sublayer
load_all_display_settings(trees_threshold, f'{folderPath}/trees_display_settings.json')
load_all_display_settings(rewilded_threshold, f'{folderPath}/rewilded_display_settings.json')
load_all_display_settings(under_tree_treatments_threshold, f'{folderPath}/under_tree_treatments_display_settings.json')

# Show the sublayers
Show(trees_threshold)
Show(rewilded_threshold)
Show(under_tree_treatments_threshold)

# Render the view to reflect changes
Render()
