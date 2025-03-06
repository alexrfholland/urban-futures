import os
import trimesh
import pyvista as pv

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

def convert_obj_to_gltf(folder_name):
    print(f"Loading OBJ file from {folder_name}...")

    obj_path = os.path.join(folder_name, f"{os.path.basename(folder_name)}.obj")
    
    # Load the OBJ file and convert to a trimesh scene (which will include textures automatically)
    scene = trimesh.load(obj_path, force='scene')
    
    # Analyze the scene for grouping and textures
    #analyze_scene(scene)
    
    # Visualize the geometries
    visualize_geometries(scene)
    
    # Define output GLTF path
    gltf_output = os.path.join(folder_name, f"{os.path.basename(folder_name)}.glb")
    
    # Export the scene as GLTF (binary format)
    scene.export(gltf_output)
    
    print(f"Converted and saved to GLTF at {gltf_output}.")


if __name__ == "__main__":
    folder_name = "data/revised/obj/B4-22"
    convert_obj_to_gltf(folder_name)

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
                plotter.add_mesh(block, show_edges=False, texture=True)
        
        # Display the plot
        plotter.show()
    else:
        print("The loaded file is not a MultiBlock dataset.")

if __name__ == "__main__":
    folder_name = "data/revised/obj/B4-22"
    convert_obj_to_gltf(folder_name)

    glb_file = "data/revised/obj/B4-22/B4-22.glb"
    #load_and_view_glb(glb_file)
