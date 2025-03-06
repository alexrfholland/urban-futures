import open3d as o3d

def load_and_visualize_textured_mesh(obj_path):
    # Load the textured mesh
    mesh = o3d.io.read_triangle_mesh(obj_path)
    
    # Print mesh information
    print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    print(f"Mesh has textures: {mesh.has_textures()}")
    print(f"Mesh has triangle uvs: {mesh.has_triangle_uvs()}")
    
    # Ensure the mesh has vertex colors for proper visualization
    if not mesh.has_vertex_colors():
        mesh.vertex_colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for _ in range(len(mesh.vertices))])
    
    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

if __name__ == "__main__":
    obj_path = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/experimental/B4-21/B4-21.obj"
    load_and_visualize_textured_mesh(obj_path)