import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

# Global variables
current_angle = 45
current_target = [0, 0, 0]
i, j = 0, 0
coordinates = np.array([[[i*200, j*200, 0] for j in range(3)] for i in range(3)])
camera_distance = 200  # Initial camera distance, you can adjust this value


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


def setup_camera(plotter, gridDistance=200, camDistance=200):
    global coordinates, camera_distance
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

    # Zoom in and zoom out functions
    def zoom_in(plotter):
        global camera_distance
        camera_distance = max(100, camera_distance - 50)  # Decrease distance, limit to a minimum value
        set_camera_to_location(plotter, coordinates, i, j, camera_distance)
        plotter.render()

    def zoom_out(plotter):
        global camera_distance
        camera_distance += 50  # Increase distance
        set_camera_to_location(plotter, coordinates, i, j, camera_distance)
        plotter.render()

    # Assign key events for navigation and zoom
    plotter.add_key_event("i", lambda: increment_i(plotter))
    plotter.add_key_event("j", lambda: increment_j(plotter))
    plotter.add_key_event("r", lambda: reset_ij(plotter))
    plotter.add_key_event("y", lambda: on_key_y_pressed(plotter))
    plotter.add_key_event("z", lambda: zoom_in(plotter))  # Key for zooming in
    plotter.add_key_event("x", lambda: zoom_out(plotter))  # Key for zooming out

    # Initialize camera position
    set_camera_to_location(plotter, coordinates, 0, 0, camera_distance)


if __name__ == "__main__":
    plotter = pv.Plotter()
    add_colored_cubes(plotter)
    setup_camera(plotter, 200, 250)
    plotter.show()