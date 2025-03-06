import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import json

# Global variables for camera settings
camera_settings = []
current_view_index = -1
current_angle = 45
current_target = [0, 0, 0]
i, j = 0, 0
coordinates = np.array([[[i*200, j*200, 0] for j in range(3)] for i in range(3)])
camera_distance = 200  # Initial camera distance
light2 = None  
light = None

# Function to update the global camera settings
def update_camera_settings(plotter):
    global camera_settings
    camera_settings['camera_position'] = plotter.camera_position
    camera_settings['view_angle'] = plotter.camera.view_angle

def save_camera_settings(plotter):
    global camera_settings
    camera_settings.append({
        'camera_position': plotter.camera_position,
        'view_angle': plotter.camera.view_angle
    })
    print(f"View saved. Total saved views: {len(camera_settings)}")

def apply_camera_settings(plotter, index):
    global camera_settings
    global current_view_index
    if 0 <= index < len(camera_settings):
        settings = camera_settings[index]
        plotter.camera_position = settings['camera_position']
        plotter.camera.view_angle = settings['view_angle']
        plotter.render()
    else:
        print(f"No camera settings found at index: {index}, looping back to 0")
        current_view_index = -1

def next_view(plotter):
    global current_view_index
    current_view_index += 1
    apply_camera_settings(plotter, current_view_index)

def previous_view(plotter):
    global current_view_index
    current_view_index = max(0, current_view_index - 1)
    apply_camera_settings(plotter, current_view_index)

def serialize_camera_settings(camera_settings):
    serialized_settings = []
    for setting in camera_settings:
        camera_pos, focal_point, view_up = setting['camera_position']
        serialized_setting = {
            'camera_position': {
                'camera_pos': list(camera_pos),
                'focal_point': list(focal_point),
                'view_up': list(view_up)
            },
            'view_angle': setting['view_angle']
        }
        serialized_settings.append(serialized_setting)
    return serialized_settings

def save_camera_settings_to_file(filename="camera_views.json"):
    global camera_settings
    serialized_settings = serialize_camera_settings(camera_settings)
    with open(filename, "w") as file:
        json.dump(serialized_settings, file, indent=4)
    print("Camera views saved to file.")

def deserialize_camera_settings(serialized_settings):
    camera_settings = []
    for setting in serialized_settings:
        camera_pos = tuple(setting['camera_position']['camera_pos'])
        focal_point = tuple(setting['camera_position']['focal_point'])
        view_up = tuple(setting['camera_position']['view_up'])
        camera_position = (camera_pos, focal_point, view_up)
        view_angle = setting['view_angle']
        camera_settings.append({'camera_position': camera_position, 'view_angle': view_angle})
    return camera_settings

def load_camera_settings_from_file(filename="camera_views.json"):
    global camera_settings
    with open(filename, "r") as file:
        serialized_settings = json.load(file)
    camera_settings = deserialize_camera_settings(serialized_settings)
    print("Camera views loaded from file.")

def zoom_in(plotter, zoom_factor=1.1):
    # Zoom in by a factor (greater than 1)
    plotter.zoom_camera(zoom_factor)

def zoom_out(plotter, zoom_factor=0.9):
    # Zoom out by a factor (less than 1)
    plotter.zoom_camera(zoom_factor)


def add_colored_cubes(plotter):
    for i in range(3):
        for j in range(3):
            center = coordinates[i, j]
            color_index = (i * 3 + j) % 20  # Cycle through the 20 colors in the Tab20 colormap
            color = plt.cm.tab20.colors[color_index]
            cube = pv.Cube(center=center, x_length=20, y_length=20, z_length=20)
            plotter.add_mesh(cube, color=color)

def set_camera_to_location(plotter, coordinates, i, j, distance):
    target_coordinate = coordinates[i, j]
    camera_pos = [
        target_coordinate[0] + distance * np.sqrt(2)/2,
        target_coordinate[1] + distance * np.sqrt(2)/2,
        target_coordinate[2] + distance
    ]
    plotter.camera_position = (camera_pos, target_coordinate, [0, 0, 1])

def setup_camera(plotter, gridDistance=200, camDistance=200):
    global coordinates, camera_distance, light, light2

    light = pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=0.7)
    plotter.add_light(light)
    
    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.1  # Set reduced specular reflection
    plotter.add_light(light2)

    coordinates = np.array([[[i*gridDistance, j*gridDistance, 0] for j in range(3)] for i in range(3)])
    camera_distance = camDistance

    plotter.add_key_event("z", lambda: zoom_in(plotter))  # Key for zooming in
    plotter.add_key_event("x", lambda: zoom_out(plotter))  # Key for zooming out
    plotter.add_key_event("s", lambda: save_camera_settings(plotter))  # Press 's' to save current view
    plotter.add_key_event("Right", lambda: next_view(plotter))  # Press 'Right arrow' to go to next view
    plotter.add_key_event("Left", lambda: previous_view(plotter))  # Press 'Left arrow' to go to previous view
    plotter.add_key_event("S", lambda: save_camera_settings_to_file())  # Press 'S' to save camera views to file
    plotter.add_key_event("L", lambda: load_camera_settings_from_file())  # Press 'L' to load camera views from file

    set_camera_to_location(plotter, coordinates, 0, 0, camera_distance)

if __name__ == "__main__":
    plotter = pv.Plotter()
    add_colored_cubes(plotter)
    setup_camera(plotter, 200, 250)
    save_camera_settings(plotter)  # Save initial camera setting
    plotter.show()
