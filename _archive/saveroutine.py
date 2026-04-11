
import rhino3dm
from typing import List, Any
import ghhops_server as hs

# Function to write mesh data to a PLY file
def write_ply(filename, vertices, faces):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for vertex in vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

        for face in faces:
            f.write(f"3 {' '.join(map(str, face))}\n")

# Assume `hops_mesh` is your ghhops_server.HopsMesh object
hops_mesh = ...

# Extract vertices and faces
vertices = [(v.X, v.Y, v.Z) for v in hops_mesh.Vertices]
faces = [(f.A, f.B, f.C) for f in hops_mesh.Faces]

# Write to a PLY file
write_ply("output.ply", vertices, faces)
