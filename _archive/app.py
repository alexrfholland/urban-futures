"""Hops flask middleware example"""
import sys
import os
from flask import Flask

# load ghhops-server-py source from this directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import ghhops_server as hs

import rhino3dm
from typing import List, Any, Tuple, Union


# register hops app as middleware
app = Flask(__name__)
hops: hs.HopsFlask = hs.Hops(app)


import numpy as np
import pyvista as pv
import time


from typing import List
import rhino3dm
import pyvista as pv
import numpy as np

def PointstoPoints(rhino_points: List[rhino3dm.Point3d], name: str, site: str) -> None:
    
    multi_block = pv.MultiBlock()  # Initialize multi-block dataset
    vtk_file_path = f'data/{site}/{name}.vtm'

    centroids = []  # Initialize list to store centroids
    
    for idx, rhino_point in enumerate(rhino_points):
        #print(f"Processing point {idx + 1}...")
        
        # Create new point and append to centroids list
        new_point = np.array([rhino_point.X, rhino_point.Y, rhino_point.Z])
        centroids.append(new_point)
        
        #print(f"Successfully processed point {idx + 1}.")

    # Convert centroids list to Numpy array
    centroids = np.array(centroids)

    # Create a PyVista point cloud
    point_cloud = pv.PolyData(centroids)
    
    # Add to multi-block dataset
    multi_block.append(point_cloud)
            
    # Serialize multi-block dataset to a single VTK file
    multi_block.save(vtk_file_path)
    print(f"Saved point cloud to {vtk_file_path}.")
    
    # Visualize multi-block dataset
    plotter = pv.Plotter()
    plotter.add_mesh(multi_block, point_size=5.0, render_points_as_spheres=True)
    plotter.show()
    print("Visualization complete.")




def MeshesToPoints(rhino_meshes: List[rhino3dm.Mesh], metadata: dict, name: str, site: str) -> None:
    print(f"Metadata headers are {metadata.keys()}, metadata rows: {len(metadata)}, meshes rows: {len(rhino_meshes)}")
    
    multi_block = pv.MultiBlock()  # Initialize multi-block dataset
    vtk_file_path = f'data/{site}/{name}.vtm'

    
    for idx, rhino_mesh in enumerate(rhino_meshes):
        print(f"Processing mesh {idx + 1}...")
        
        # Initialize lists to store centroids, normals, strike and dip
        centroids = []
        averagedNormals = []
        strike_values = []
        dip_values = []
        
        # Loop through each face in the mesh to calculate its centroid and averaged normal
        for i in range(len(rhino_mesh.Faces)):
            face = rhino_mesh.Faces[i]
            face_vertices = np.array([(v.X, v.Y, v.Z) for v in [rhino_mesh.Vertices[i] for i in face]])
            centroid = np.mean(face_vertices, axis=0)
            centroids.append(centroid)

            # Average the normals of all vertices of the face
            normals = np.array([(n.X, n.Y, n.Z) for n in [rhino_mesh.Normals[i] for i in face]])
            averagedNormal = np.mean(normals, axis=0)
            averagedNormals.append(averagedNormal)

            # Calculate strike and dip
            strike = np.arctan2(averagedNormal[1], averagedNormal[0])
            dip = np.arccos(averagedNormal[2] / np.linalg.norm(averagedNormal))
            strike_values.append(strike)
            dip_values.append(dip)
        
        # Convert centroids and normals to Numpy arrays
        centroids = np.array(centroids)
        averagedNormals = np.array(averagedNormals)
        strike_values = np.array(strike_values)
        dip_values = np.array(dip_values)
        
        # Create a PyVista point cloud
        point_cloud = pv.PolyData(centroids)
        
        # Add metadata as point attributes
        for key, value_list in metadata.items():
            value = value_list[idx]  # Assuming one value per mesh
            point_cloud.point_data[key] = np.full(centroids.shape[0], value)

        # Add normals, strike, and dip as attributes
        point_cloud.point_data['averagedNormals'] = averagedNormals
        point_cloud.point_data['strike'] = strike_values
        point_cloud.point_data['dip'] = dip_values
        
        # Add to multi-block dataset
        multi_block.append(point_cloud)
        
        print(f"Successfully processed mesh {idx + 1}.")
    
    # Verify point attributes before saving
    for idx, block in enumerate(multi_block):
        print(f"Attributes for block {idx + 1}: {list(block.point_data.keys())}")
    
    # Serialize multi-block dataset to a single VTK file
    multi_block.save(vtk_file_path)
    print(f"Saved all point clouds to {vtk_file_path}.")
    
    # Visualize multi-block dataset
    plotter = pv.Plotter()
    plotter.add_mesh(multi_block, point_size=5.0, render_points_as_spheres=True)
    plotter.show()
    print("Visualization complete.")

def MeshesToPoints2(rhino_meshes: List[rhino3dm.Mesh], metadata: dict, name: str) -> None:
    print(f'metadata headers are {metadata.keys()}, metadata rows: {len(metadata)}, meshes rows: {len(rhino_meshes)}')
    
    multi_block = pv.MultiBlock()  # Initialize multi-block dataset
    
    vtk_file_path = f'data/{name}.vtm'
    
    for idx, rhino_mesh in enumerate(rhino_meshes):
        print(f"Processing mesh {idx + 1}...")
        
        # Initialize list to store centroids
        centroids = []
        
        # Loop through each face in the mesh to calculate its centroid
        num_triangles = rhino_mesh.Faces.TriangleCount
        num_quads = rhino_mesh.Faces.QuadCount
        for i in range(num_triangles + num_quads):
            face = rhino_mesh.Faces[i]
            face_vertices = np.array([(v.X, v.Y, v.Z) for v in [rhino_mesh.Vertices[i] for i in face]])
            centroid = np.mean(face_vertices, axis=0)
            centroids.append(centroid)
        
        # Convert centroids to a Numpy array
        centroids = np.array(centroids)
        
        # Create a PyVista point cloud
        point_cloud = pv.PolyData(centroids)
        
        # Add metadata as point attributes
        for key, value_list in metadata.items():
            value = value_list[idx]  # Assuming one value per mesh
            point_cloud.point_data[key] = np.full(centroids.shape[0], value)
        
        # Add to multi-block dataset
        multi_block.append(point_cloud)
        
        print(f"Successfully processed mesh {idx + 1}.")
    
    # Verify point attributes before saving
    for idx, block in enumerate(multi_block):
        print(f"Attributes for block {idx + 1}: {list(block.point_data.keys())}")
    
    # Serialize multi-block dataset to a single VTK file
    multi_block.save(vtk_file_path.replace('.vtk', '.vtm'))
    print(f"Saved all point clouds to {vtk_file_path.replace('.vtk', '.vtm')}.")
    
    # Visualize multi-block dataset
    plotter = pv.Plotter()
    plotter.add_mesh(multi_block, point_size=5.0, render_points_as_spheres=True)
    plotter.show()
    print("Visualization complete.")



@hops.component(
    "/getTopo",
    name="getTopo",
    nickname="Get Meshes",
    description="Get Meshes",
    icon="pointat.png",
    inputs=[
       hs.HopsString("regen", "R", "G"),
       hs.HopsString("site", "S", "S"),
       hs.HopsString("Name", "N", "Name"),
       hs.HopsMesh("Mesh", "M", "Mesh", hs.HopsParamAccess.LIST),
    ],
    outputs=[hs.HopsString("EP", "P", "Point on curve at t")],
)
def getTopo(gen: str, site: str, name: str, meshes: List[hs.HopsMesh]):
    metadata = {'type' : 'ground'}
    MeshesToPoints(meshes, metadata, name, site)
    #write_rhino3dm_mesh_to_ply("output.ply", meshes[1])
    print("extracted mesh")
    return "done"


@hops.component(
    "/getPts",
    name="Get Points",
    nickname="Get Points",
    description="Get Points",
    icon="pointat.png",
    inputs=[
       hs.HopsString("regen", "R", "G"),
       hs.HopsString("site", "S", "S"),
       hs.HopsString("Name", "N", "Name"),
       hs.HopsPoint("Point", "P", "P", hs.HopsParamAccess.LIST),
    ],
    outputs=[hs.HopsString("EP", "P", "Point on curve at t")],
)
def get_Points(gen: str, site: str, name: str, points: List[hs.HopsPoint]) -> dict:
    print("starting to extract points")
    PointstoPoints(points, name, site)
    print("extracted points")
    return "done"


@hops.component(
    "/getMeshes",
    name="Get Meshes",
    nickname="Get Meshes",
    description="Get Meshes",
    icon="pointat.png",
    inputs=[
       hs.HopsString("regen", "R", "G"),
       hs.HopsString("site", "S", "S"),
       hs.HopsString("Name", "N", "Name"),
       hs.HopsMesh("Mesh", "M", "Mesh", hs.HopsParamAccess.LIST),
       hs.HopsString("Headers", "H", "Headers", hs.HopsParamAccess.LIST),
       hs.HopsString("Values", "V", "Values", hs.HopsParamAccess.TREE),
    ],
    outputs=[hs.HopsString("EP", "P", "Point on curve at t")],
)
def get_Meshes(gen: str, site: str, name: str, meshes: List[hs.HopsMesh], headers: List[str], values: Any) -> dict:
    metadata = {}
    print(f'headers are {headers}')
    for i in range(len(headers)):
        key = headers[i]
        value = values[f"{{{i}}}"]  # Assuming values is a list or array-like object
        metadata.update({key: value})

    print('inspecting')
    print(f'mesh is of type {type(meshes[1])}')
    # Write to a PLY file
    print("starting to extract mesh")
   

    MeshesToPoints(meshes, metadata, name, site)
    #write_rhino3dm_mesh_to_ply("output.ply", meshes[1])
    print("extracted mesh")
    return "done"


@hops.component(
    "/getbuilding3",
    name="Get Buildings",
    nickname="Get Buildings",
    description="Get buildings",
    icon="pointat.png",
    inputs=[
        hs.HopsBrep("Brep", "C", "Curve to evaluate"),
    ],
    outputs=[hs.HopsPoint("P", "P", "Point on curve at t")],
)
def getbuilding(buildings: rhino3dm.Brep):
    print('hi')
    print(len(buildings))
    print('test')
    print('boo')




@hops.component(
    "/tester",
    name="Get Buildings",
    nickname="Get Buildings",
    description="Get buildings",
    icon="pointat.png",
    inputs=[
       hs.HopsBrep("Brep", "C", "Curve to evaluate", hs.HopsParamAccess.TREE),
       #hs.HopsNumber("Brep", "C", "Curve to evaluate", hs.HopsParamAccess.LIST),
        
    ],
    outputs=[hs.HopsBrep("EP", "P", "Point on curve at t")],
)
def getbuilding(no: [hs.HopsBrep]):
    print('hi-poo4')
    return no["{0}"]



# flask app can be used for other stuff drectly
@app.route("/help")
def help():
    return "Welcome to Grashopper Hops for CPython!"


@app.route("/update", methods=["POST"])
def update():
    return "Update example!"


# /solve is a builtin method and can not be overrriden
@app.route("/solve", methods=["GET", "POST"])
def solve():
    return "Oh oh! this should not happen!"


@hops.component(
    "/binmult",
    inputs=[hs.HopsNumber("A"), hs.HopsNumber("B")],
    outputs=[hs.HopsNumber("Multiply")],
)
def BinaryMultiply(a: float, b: float):
    return a * b


@hops.component(
    "/add",
    name="Add",
    nickname="Add",
    description="Add numbers with CPython",
    inputs=[
        hs.HopsNumber("A", "A", "First number"),
        hs.HopsNumber("B", "B", "Second number"),
    ],
    outputs=[hs.HopsNumber("Sum", "S", "A + B")],
)
def add(a: float, b: float):
    return a + b



@hops.component(
    "/pointat",
    name="PointAt",
    nickname="PtAt",
    description="Get point along curve",
    icon="pointat.png",
    inputs=[
        hs.HopsCurve("Curve", "C", "Curve to evaluate"),
        hs.HopsNumber("t", "t", "Parameter on Curve to evaluate"),
    ],
    outputs=[hs.HopsPoint("P", "P", "Point on curve at t")],
)
def pointat(curve: rhino3dm.Curve, t=0.0):
    return curve.PointAt(t)


@hops.component(
    "/srf4pt",
    name="4Point Surface",
    nickname="Srf4Pt",
    description="Create ruled surface from four points",
    inputs=[
        hs.HopsPoint("Corner A", "A", "First corner"),
        hs.HopsPoint("Corner B", "B", "Second corner"),
        hs.HopsPoint("Corner C", "C", "Third corner"),
        hs.HopsPoint("Corner D", "D", "Fourth corner"),
    ],
    outputs=[hs.HopsSurface("Surface", "S", "Resulting surface")],
)
def ruled_surface(
    a: rhino3dm.Point3d,
    b: rhino3dm.Point3d,
    c: rhino3dm.Point3d,
    d: rhino3dm.Point3d,
):
    edge1 = rhino3dm.LineCurve(a, b)
    edge2 = rhino3dm.LineCurve(c, d)
    return rhino3dm.NurbsSurface.CreateRuledSurface(edge1, edge2)


@hops.component(
    "/test.IntegerOutput",
    name="tIntegerOutput",
    nickname="tIntegerOutput",
    description="Add numbers with CPython and Test Integer output.",
    inputs=[
        hs.HopsInteger("A", "A", "First number"),
        hs.HopsInteger("B", "B", "Second number"),
    ],
    outputs=[hs.HopsInteger("Sum", "S", "A + B")],
)
def test_IntegerOutput(a, b):
    return a + b


if __name__ == "__main__":
    app.run()
