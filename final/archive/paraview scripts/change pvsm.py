from paraview.simple import *

# Load the ParaView state file
state_file = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/city scenario.pvsm"
LoadState(state_file)

# Set the VTK file of interest
target_vtk_file = "trimmed-parade_1_scenarioYR0.vtk"

# Get all sources in the current pipeline
sources = GetSources()

# Loop over all sources to find those associated with the target VTK file
for source_name, source_proxy in sources.items():
    # Check if this source is related to the target VTK file
    try:
        # Fetch the file name property if it exists (some sources don't have a FileNames property)
        file_names = source_proxy.GetPropertyValue('FileNames')
        if file_names and target_vtk_file in file_names[0]:
            print(f"Source Name: {source_name}")
            print(f"Source Type: {source_proxy.SMProxy.GetXMLName()}")
            
            # Optionally, print more details of the source's properties
            for prop in source_proxy.ListProperties():
                print(f"  Property: {prop} = {source_proxy.GetPropertyValue(prop)}")
            
            # Recursively check other filters and transformations connected to this source
            connected_sources = [src for src in sources.items() if src[1].GetPropertyValue('Input') == source_proxy]
            for connected_name, connected_proxy in connected_sources:
                print(f"  Connected Source: {connected_name} ({connected_proxy.SMProxy.GetXMLName()})")
    except Exception as e:
        # Handle any sources without 'FileNames' property or input connections
        continue
