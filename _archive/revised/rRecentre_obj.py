import trimesh
import numpy as np

def print_obj_centroid(obj_path):
    print(f"Loading OBJ file from: {obj_path}")
    scene = trimesh.load(obj_path)
    
    if isinstance(scene, trimesh.Scene):
        print("OBJ contains multiple geometries (Scene detected).")
        all_vertices = []
        for name, mesh in scene.geometry.items():
            all_vertices.extend(mesh.vertices)
        centroid = np.mean(all_vertices, axis=0)
    elif isinstance(scene, trimesh.Trimesh):
        print("OBJ contains a single mesh.")
        centroid = scene.centroid
    else:
        print("Unknown object type.")
        return
    
    print(f"Centroid of the entire OBJ file: {centroid}")

def center_obj_at_origin(input_obj_path, output_obj_path):
    # Load the OBJ file
    print(f"Loading OBJ file from: {input_obj_path}")
    scene = trimesh.load(input_obj_path)

    # Check if the loaded file is a scene or a single mesh
    if isinstance(scene, trimesh.Scene):
        print("OBJ contains multiple geometries (Scene detected). Processing each geometry.")
        
        # Iterate through each geometry in the scene
        for name, mesh in scene.geometry.items():
            centroid = mesh.centroid
            print(f"Original centroid of {name}: {centroid}")
            mesh.vertices -= centroid
            print(f"{name} has been centered to (0, 0, 0).")
        
        # Save the scene back to an OBJ file
        print(f"Saving centered scene to: {output_obj_path}")
        scene.export(output_obj_path)
    elif isinstance(scene, trimesh.Trimesh):
        print("OBJ contains a single mesh.")
        
        # Handle single mesh case
        centroid = scene.centroid
        print(f"Original centroid of the mesh: {centroid}")
        scene.vertices -= centroid
        print(f"Mesh has been translated to center the centroid at (0, 0, 0).")

        # Save the modified mesh back to an OBJ file
        print(f"Saving centered OBJ file to: {output_obj_path}")
        scene.export(output_obj_path, file_type='obj')
    
    print("Operation complete.")

# Example usage
input_obj_path = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/experimental/B4-21/B4-21.obj'
import os

output_folder = os.path.join(os.path.dirname(input_obj_path), 'centre')
os.makedirs(output_folder, exist_ok=True)
output_obj_path = os.path.join(output_folder, f'{os.path.basename(input_obj_path).split(".")[0]}-centre.obj')


#center_obj_at_origin(input_obj_path, output_obj_path)
print_obj_centroid(output_obj_path)
