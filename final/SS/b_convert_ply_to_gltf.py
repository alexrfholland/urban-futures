import trimesh
import os

# Specify the folder containing .ply files
input_folder = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/trimmed-parade/scenes/exports"
output_folder = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/trimmed-parade/scenes/exports"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".ply"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".ply", ".glb"))

        # Load the .ply file with trimesh
        mesh = trimesh.load(input_path)

        # Check if the mesh loaded successfully
        if mesh is not None:
            # Export the mesh to .glb (single file with all data included)
            mesh.export(output_path, file_type='glb')
            print(f"Converted {input_path} to {output_path}")
        else:
            print(f"Failed to load {input_path}")

print("Batch conversion to .glb completed.")
