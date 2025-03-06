import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import json

# Global variables for camera settings
global camera_settings
camera_settings = []
global current_view_index
current_view_index = -1  # Initialize with -1, indicating no view is currently selected

global savename
savename = 'camera_views.json'



# Function to update the global camera settings
def update_camera_settings(plotter):
    global camera_settings
    camera_settings['camera_position'] = plotter.camera_position
    camera_settings['view_angle'] = plotter.camera.view_angle


"""def save_camera_settings(plotter):
    global camera_settings
    camera_settings.append({
        'camera_position': plotter.camera_position,
        'view_angle': plotter.camera.view_angle
    })
    print(f"View saved. Total saved views: {len(camera_settings)}")"""

def save_camera_settings(plotter):
    global camera_settings, light, light2
    camera_settings.append({
        'camera_position': plotter.camera_position,
        'view_angle': plotter.camera.view_angle,
        'light_intensity': light.intensity if light else None,
        'light2_intensity': light2.intensity if light2 else None
    })
    print(f"View saved. Total saved views: {len(camera_settings)}")



def apply_camera_settings(plotter, index):
    global camera_settings, light, light2
    global current_view_index

    # Adjust the index to loop within the range
    if len(camera_settings) > 0:
        current_view_index = index % len(camera_settings)
        settings = camera_settings[current_view_index]
        plotter.camera_position = settings['camera_position']
        plotter.camera.view_angle = settings['view_angle']
        if 'light_intensity' in settings and light:
            light.intensity = settings['light_intensity']
        if 'light2_intensity' in settings and light2:
            light2.intensity = settings['light2_intensity']

        plotter.render()
        print(f'Applied camera settings at {current_view_index}: {camera_settings[current_view_index]}')
    else:
        print(f'No camera settings available at {current_view_index}')
        current_view_index = -1

def next_view(plotter):
    global current_view_index
    current_view_index += 1
    apply_camera_settings(plotter, current_view_index)

def previous_view(plotter):
    global current_view_index
    current_view_index -= 1
    apply_camera_settings(plotter, current_view_index)



def zoom_in(plotter):
    global camera_settings, camera_distance
    if camera_settings:
        # Retrieve the last camera position and view_up vector
        last_camera_position, last_focal_point, last_view_up = camera_settings[-1]['camera_position']
        camera_distance = max(100, camera_distance - 50)  # Decrease distance

        # Calculate the new camera position while maintaining the same angle
        new_camera_pos = [
            last_camera_position[0] - last_view_up[0] * 50,
            last_camera_position[1] - last_view_up[1] * 50,
            last_camera_position[2] - last_view_up[2] * 50
        ]
        plotter.camera_position = (new_camera_pos, last_focal_point, last_view_up)
        plotter.render()
    else:
        print("No saved camera settings found. Please save a camera setting first.")

def zoom_out(plotter):
    global camera_settings, camera_distance
    if camera_settings:
        # Retrieve the last camera position and view_up vector
        last_camera_position, last_focal_point, last_view_up = camera_settings[-1]['camera_position']
        camera_distance += 50  # Increase distance

        # Calculate the new camera position while maintaining the same angle
        new_camera_pos = [
            last_camera_position[0] + last_view_up[0] * 50,
            last_camera_position[1] + last_view_up[1] * 50,
            last_camera_position[2] + last_view_up[2] * 50
        ]
        plotter.camera_position = (new_camera_pos, last_focal_point, last_view_up)
        plotter.render()
    else:
        print("No saved camera settings found. Please save a camera setting first.")



###saving views to file
"""def serialize_camera_settings(camera_settings):
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
"""

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
            'view_angle': setting['view_angle'],
            'light_intensity': setting['light_intensity'],
            'light2_intensity': setting['light2_intensity']
        }
        serialized_settings.append(serialized_setting)
    return serialized_settings


def save_camera_settings_to_file():
    global camera_settings, savename
    serialized_settings = serialize_camera_settings(camera_settings)
    with open(savename, "w") as file:
        json.dump(serialized_settings, file, indent=4)
    print(f'Camera views saved to file. {savename}\n{camera_settings}')

"""def deserialize_camera_settings(serialized_settings):
    camera_settings = []
    for setting in serialized_settings:
        camera_pos = tuple(setting['camera_position']['camera_pos'])
        focal_point = tuple(setting['camera_position']['focal_point'])
        view_up = tuple(setting['camera_position']['view_up'])
        camera_position = (camera_pos, focal_point, view_up)
        view_angle = setting['view_angle']
        camera_settings.append({'camera_position': camera_position, 'view_angle': view_angle})
    return camera_settings"""

def deserialize_camera_settings(serialized_settings):
    camera_settings = []
    for setting in serialized_settings:
        camera_pos = tuple(setting['camera_position']['camera_pos'])
        focal_point = tuple(setting['camera_position']['focal_point'])
        view_up = tuple(setting['camera_position']['view_up'])
        camera_position = (camera_pos, focal_point, view_up)
        view_angle = setting['view_angle']
        light_intensity = setting['light_intensity']
        light2_intensity = setting['light2_intensity']
        camera_settings.append({
            'camera_position': camera_position,
            'view_angle': view_angle,
            'light_intensity': light_intensity,
            'light2_intensity': light2_intensity
        })
    return camera_settings



def load_camera_settings_from_file(plotter):
    global camera_settings, current_view_index, savename
    with open(savename, "r") as file:
        serialized_settings = json.load(file)
    camera_settings = deserialize_camera_settings(serialized_settings)
    print(f'Camera views loaded from file {savename}')


    # Set current_view_index to the last index in the camera_settings
    current_view_index = 0 
    
    len(camera_settings) - 1



def zoom_in(plotter):
    global camera_settings, camera_distance
    if camera_settings:
        last_view_up = camera_settings[-1]['camera_position'][2]
        camera_distance = max(100, camera_distance - 50)  # Decrease distance
        update_camera_position(plotter, view_up=last_view_up)

def zoom_out(plotter):
    global camera_settings, camera_distance
    if camera_settings:
        last_view_up = camera_settings[-1]['camera_position'][2]
        camera_distance += 50  # Increase distance
        update_camera_position(plotter, view_up=last_view_up)

def update_camera_position(plotter, view_up):
    global current_target, current_angle, camera_distance
    camera_pos = [
        current_target[0] + camera_distance * np.sin(np.radians(current_angle)),
        current_target[1] + camera_distance * np.cos(np.radians(current_angle)),
        current_target[2]
    ]
    plotter.camera_position = (camera_pos, current_target, view_up)
    plotter.render()

current_angle = 45
current_target = [0, 0, 0]
i, j = 0, 0
coordinates = np.array([[[i*200, j*200, 0] for j in range(3)] for i in range(3)])
camera_distance = 200  # Initial camera distance, you can adjust this value
light2 = None  
light = None


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


def increment_i(plotter):
    global i, j, current_target
    i = (i + 1) % 3
    current_target = coordinates[i, j]
    set_camera_to_location(plotter, coordinates, i, j)
    plotter.render()  # Fixed this line


def increment_j(plotter):
    global i, j, current_target
    j = (j + 1) % 3
    current_target = coordinates[i, j]
    set_camera_to_location(plotter, coordinates, i, j)
    plotter.render()

def reset_ij(plotter):
    global i, j, current_target
    i, j = 0, 0
    current_target = coordinates[i, j]
    set_camera_to_location(plotter, coordinates, i, j)
    plotter.render()

def on_key_y_pressed(plotter):
    global current_angle, current_target, camera_distance
    current_angle += 90  # Rotate the camera by 90 degrees
    camera_pos = [
        current_target[0] + camera_distance * np.sin(np.radians(current_angle)) / np.sqrt(2),
        current_target[1] + camera_distance * np.cos(np.radians(current_angle)) / np.sqrt(2),
        current_target[2] + camera_distance
    ]

    plotter.camera_position = (camera_pos, current_target, (0, 0, 1))
    plotter.render()  # Force a render update

"""def on_key_y_pressed(plotter):
    # Get the current camera settings directly from the plotter
    camera_pos, focal_point, view_up = plotter.camera_position

    # Calculate the direction vector from camera_pos to focal_point
    direction_vector = np.array(focal_point) - np.array(camera_pos)

    # Rotate the direction vector by 90 degrees around the Z-axis (up direction)
    rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90-degree rotation matrix around Z-axis
    rotated_direction_vector = np.dot(rotation_matrix, direction_vector)

    # Calculate the new camera position to maintain the distance between camera and focal point
    new_camera_pos = np.array(focal_point) - rotated_direction_vector

    # Apply the new camera position
    plotter.camera_position = (new_camera_pos.tolist(), focal_point, view_up)
    plotter.render()"""





def on_key_n_pressed(plotter):
    global light2
    light2.intensity += 0.1  # Increase light intensity
    plotter.render()

def on_key_m_pressed(plotter):
    global light2
    light2.intensity = max(light2.intensity - 0.1, 0)  # Decrease light intensity, limit to a minimum of 0
    plotter.render()

def on_key_v_pressed(plotter):
    global light
    light.intensity += 0.1  # Increase light intensity
    plotter.render()

def on_key_b_pressed(plotter):
    global light
    light.intensity = max(light.intensity - 0.1, 0)  # Decrease light intensity, limit to a minimum of 0
    plotter.render()


                                                                                                


def setup_camera(plotter, gridDistance=200, camDistance=200, name=None):
    global coordinates, camera_distance, light, light2, savename
    if name is not None:
        savename = f'{name}_{savename}'
        print(f'name changed to {savename}')

    light = pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=0.7)
    plotter.add_light(light)
    

    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.1  # Set reduced specular reflection
    plotter.add_light(light2)

    coordinates = np.array([[[i*gridDistance, j*gridDistance, 0] for j in range(3)] for i in range(3)])
    camera_distance = camDistance  # Set the initial camera distance

    # Function to update camera position based on the current i, j, and camera distance
    def set_camera_to_location(plotter, coordinates, i, j, distance):
        target_coordinate = coordinates[i, j]
        camera_pos = [
            target_coordinate[0] + distance * np.sqrt(2)/2,
            target_coordinate[1] + distance * np.sqrt(2)/2,
            target_coordinate[2] + distance
        ]
        plotter.camera_position = (camera_pos, target_coordinate, [0, 0, 1])


    # Assign key events for navigation and zoom
    #plotter.add_key_event("i", lambda: increment_i(plotter))
    #plotter.add_key_event("j", lambda: increment_j(plotter))
    #plotter.add_key_event("r", lambda: reset_ij(plotter))
    plotter.add_key_event("y", lambda: on_key_y_pressed(plotter))

    #plotter.add_key_event("z", lambda: zoom_in(plotter))  # Key for zooming in
    #plotter.add_key_event("x", lambda: zoom_out(plotter))  # Key for zooming out
    plotter.add_key_event("n", lambda: on_key_n_pressed(plotter))  # Key for increasing light intensity
    plotter.add_key_event("m", lambda: on_key_m_pressed(plotter))  # Key for decreasing light intensity
    plotter.add_key_event("0", lambda: on_key_v_pressed(plotter))  # Key for increasing light intensity
    plotter.add_key_event("9", lambda: on_key_b_pressed(plotter))  # Key for decreasing light intensity
    #plotter.add_key_event("s", lambda: save_current_view(plotter))  # Press 'S' to save current view
    #plotter.add_key_event("l", lambda: load_saved_view(plotter))    # Press 'L' to load a saved view

    plotter.add_key_event("s", lambda: save_camera_settings(plotter))  # Press 's' to save current view
    plotter.add_key_event("Right", lambda: next_view(plotter))         # Press 'Right arrow' to go to next view
    plotter.add_key_event("Left", lambda: previous_view(plotter))      # Press 'Left arrow' to go to previous view

    plotter.add_key_event("S", lambda: save_camera_settings_to_file())  # Press 'S' to save camera views to file
    plotter.add_key_event("L", lambda: load_camera_settings_from_file(plotter))  # Press 'L' to load camera views from file


    # Initialize camera position
    set_camera_to_location(plotter, coordinates, 0, 0, camera_distance)


if __name__ == "__main__":
    plotter = pv.Plotter()
    add_colored_cubes(plotter)
    setup_camera(plotter, 200, 250)
    plotter.show()