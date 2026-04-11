import pyvista as pv
import numpy as np

# Sample data: Replace this with your actual PolyData
mesh = pv.Cube()

plotter = pv.Plotter()

# Add the mesh to the plotter
plotter.add_mesh(mesh)

# Function to change the camera view to an isometric angle
def set_isometric_view(angle):
    plotter.camera.position = [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 1]
    plotter.camera.focal_point = [0, 0, 0]
    plotter.camera.view_up = [0, 0, 1]
    plotter.reset_camera()

# Function to translate the camera position along x-axis
def translate_camera_x(translation):
    x, y, z = plotter.camera.position
    plotter.camera.position = [x + translation, y, z]
    plotter.reset_camera()

# Bind keys to functions
plotter.add_key_event("1", lambda: set_isometric_view(45))
plotter.add_key_event("2", lambda: set_isometric_view(135))
plotter.add_key_event("3", lambda: set_isometric_view(225))
plotter.add_key_event("4", lambda: set_isometric_view(315))
plotter.add_key_event("q", lambda: translate_camera_x(1))  # +1 along x-axis
plotter.add_key_event("w", lambda: translate_camera_x(-1)) # -1 along x-axis

# Show the plotter
plotter.show()
