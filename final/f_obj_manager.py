import os
import numpy as np
import trimesh
import pyvista as pv
import csv

def visualize_geometries(scene):
    """
    Visualize individual geometries in the scene for inspection.
    """
    plotter = pv.Plotter()

    # Loop through each geometry and add it to the plotter
    for name, geom in scene.geometry.items():
        print(f"Visualizing geometry '{name}'...")
        mesh = pv.wrap(geom)
        plotter.add_mesh(mesh, show_edges=True, label=name)

    # Show the plot with labels to identify geometries
    plotter.add_legend()
    plotter.show()

def center_scene(scene):
    """
    Centers the scene so that its midpoint is at the origin (0,0,0).
    Saves the transformation vector as a CSV file.
    """
    print("Centering the scene...")

    # Compute the overall centroid of the scene
    all_vertices = np.vstack([geom.vertices for geom in scene.geometry.values()])
    centroid = all_vertices.mean(axis=0)
    print(f"Computed centroid: {centroid}")

    # Translation vector to move the centroid to the origin
    translation_vector = -centroid
    print(f"Translation vector: {translation_vector}")

    # Apply translation to each geometry in the scene
    for name, geom in scene.geometry.items():
        print(f"Translating geometry '{name}'...")
        geom.vertices += translation_vector  # NumPy vectorization

    # Save the translation vector to a CSV file
    csv_path = os.path.join(folder_name, f"{os.path.basename(folder_name)}_translation.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['dx', 'dy', 'dz'])
        csv_writer.writerow(translation_vector)
    print(f"Saved translation vector to {csv_path}")

    return scene

def convert_obj_to_gltf(folder_name):
    print(f"Loading OBJ file from {folder_name}...")

    obj_path = os.path.join(folder_name, f"{os.path.basename(folder_name)}.obj")
    obj_path = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/obj/C4-12.glb'
    
    # Load the OBJ file and convert to a trimesh scene
    scene = trimesh.load(obj_path, force='mesh')

    print(f'te')



    # Center the scene
    scene = center_scene(scene)
    
    # Visualize the geometries (optional)
    # visualize_geometries(scene)
    
    # Define output GLB path
    gltf_output = os.path.join(folder_name, f"{os.path.basename(folder_name)}.glb")
    
    # Export the scene as GLB (binary format)
    scene.export(gltf_output)
    
    print(f"Converted and saved centered scene to GLB at {gltf_output}.")

def load_and_view_glb(file_path):
    print(f"Loading GLB file from {file_path}...")
    
    # Load the GLB file, which might load as a MultiBlock dataset
    multiblock = pv.read(file_path)
    
    # Check if the mesh is a MultiBlock dataset
    if isinstance(multiblock, pv.MultiBlock):
        print(f"Loaded MultiBlock with {len(multiblock)} blocks.")
        
        plotter = pv.Plotter()
        
        # Loop through each block in the MultiBlock
        for i, block in enumerate(multiblock):
            if block is not None:
                print(f"Rendering block {i}...")
                plotter.add_mesh(block, show_edges=False)
        
        # Display the plot
        plotter.show()
    else:
        print("The loaded file is not a MultiBlock dataset.")

if __name__ == "__main__":
    folder_name = "data/revised/obj/C4-12"
    convert_obj_to_gltf(folder_name)
    #glb_file = os.path.join(folder_name, f"{os.path.basename(folder_name)}.glb")
    # Uncomment the line below to view the GLB file
    """file = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/obj/C4-12/C4-12.glb'
    load_and_view_glb(file)
    """