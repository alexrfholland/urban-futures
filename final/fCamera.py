import pyvista as pv
import numpy as np
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

###saving views to file
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
    current_view_index = 0

def on_key_n_pressed(plotter):
    global light2
    light2.intensity += 0.1  # Increase light2 intensity
    plotter.render()

def on_key_m_pressed(plotter):
    global light2
    light2.intensity = max(light2.intensity - 0.1, 0)  # Decrease light2 intensity, limit to a minimum of 0
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

    # Assign key events for lights and saving/loading views
    plotter.add_key_event("[", lambda: on_key_n_pressed(plotter))  # Increase light2 intensity (Light 2)
    plotter.add_key_event("]", lambda: on_key_m_pressed(plotter))  # Decrease light2 intensity (Light 2)
    plotter.add_key_event("-", lambda: on_key_v_pressed(plotter))  # Increase light intensity (Light 1)
    plotter.add_key_event("=", lambda: on_key_b_pressed(plotter))  # Decrease light intensity (Light 1)
    plotter.add_key_event("s", lambda: save_camera_settings(plotter))  # Save current view
    plotter.add_key_event("Right", lambda: apply_camera_settings(plotter, current_view_index + 1))  # Next view
    plotter.add_key_event("Left", lambda: apply_camera_settings(plotter, current_view_index - 1))  # Previous view
    plotter.add_key_event("S", lambda: save_camera_settings_to_file())  # Save camera views to file
    plotter.add_key_event("L", lambda: load_camera_settings_from_file(plotter))  # Load camera views from file

if __name__ == "__main__":
    plotter = pv.Plotter()
    setup_camera(plotter, 200, 250)
    plotter.show()
