import pyvista as pv
import numpy as np

# Global variables to track lights
global scene_light, camera_light
scene_light = None
camera_light = None

def add_default_lighting(plotter, scene_light_intensity=0.7, camera_light_intensity=0.5):
    """
    Add default lighting setup to a plotter.
    
    Parameters:
    plotter (pyvista.Plotter): The plotter to add lighting to
    scene_light_intensity (float): Intensity of the scene light (default 0.7)
    camera_light_intensity (float): Intensity of the camera light (default 0.5)
    
    Returns:
    tuple: (scene_light, camera_light) - The two light objects added to the scene
    """
    global scene_light, camera_light
    
    # Create a positioned light (illuminates the scene from a specific position)
    scene_light = pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=scene_light_intensity)
    plotter.add_light(scene_light)
    
    # Create a camera light (follows the camera)
    camera_light = pv.Light(light_type='cameralight', intensity=camera_light_intensity)
    camera_light.specular = 0.1  # Reduce specular reflection
    plotter.add_light(camera_light)
    
    return scene_light, camera_light

def set_isometric_view(plotter, focus_on_bounds=True, negative=False):
    """
    Set the camera to PyVista's built-in isometric view.
    
    Parameters:
    plotter (pyvista.Plotter): The plotter to set view for
    focus_on_bounds (bool): Whether to focus on the dataset bounds
    negative (bool): View from the other isometric direction
    
    Returns:
    pyvista.Plotter: The configured plotter
    """
    if focus_on_bounds and plotter.renderer.actors:
        # Get the bounds of all actors in the scene
        bounds = plotter.bounds
        
        # Use PyVista's built-in isometric view with explicit bounds
        plotter.view_isometric(negative=negative, render=False, bounds=bounds)
    else:
        # Use PyVista's built-in isometric view without explicit bounds
        plotter.view_isometric(negative=negative, render=False)
    
    # Ensure the camera is positioned correctly
    if focus_on_bounds:
        plotter.reset_camera()
    
    # Update the render
    plotter.render()
    
    return plotter

def increase_scene_light_intensity(plotter, increment=0.1):
    """Increase the intensity of the scene light and update the plotter."""
    global scene_light
    if scene_light:
        scene_light.intensity += increment
        plotter.render()
        return scene_light.intensity
    return None

def decrease_scene_light_intensity(plotter, decrement=0.1):
    """Decrease the intensity of the scene light and update the plotter."""
    global scene_light
    if scene_light:
        scene_light.intensity = max(scene_light.intensity - decrement, 0)
        plotter.render()
        return scene_light.intensity
    return None

def increase_camera_light_intensity(plotter, increment=0.1):
    """Increase the intensity of the camera light and update the plotter."""
    global camera_light
    if camera_light:
        camera_light.intensity += increment
        plotter.render()
        return camera_light.intensity
    return None

def decrease_camera_light_intensity(plotter, decrement=0.1):
    """Decrease the intensity of the camera light and update the plotter."""
    global camera_light
    if camera_light:
        camera_light.intensity = max(camera_light.intensity - decrement, 0)
        plotter.render()
        return camera_light.intensity
    return None

def setup_key_bindings(plotter):
    """Setup keyboard shortcuts for controlling lights and camera views."""
    plotter.add_key_event("n", lambda: increase_camera_light_intensity(plotter))  # Increase camera light
    plotter.add_key_event("m", lambda: decrease_camera_light_intensity(plotter))  # Decrease camera light
    plotter.add_key_event("0", lambda: increase_scene_light_intensity(plotter))   # Increase scene light
    plotter.add_key_event("9", lambda: decrease_scene_light_intensity(plotter))   # Decrease scene light
    
    # Add key binding to reset to isometric view
    plotter.add_key_event("i", lambda: set_isometric_view(plotter))
    
    # Add other standard view bindings
    plotter.add_key_event("x", lambda: plotter.view_xz())  # X-Z plane view
    plotter.add_key_event("y", lambda: plotter.view_yz())  # Y-Z plane view
    plotter.add_key_event("z", lambda: plotter.view_xy())  # X-Y plane view

def setup_plotter_with_lighting(plotter, auto_isometric=True):
    """
    Setup a plotter with default lighting and key bindings.
    
    Parameters:
    plotter (pyvista.Plotter): The plotter to set up
    auto_isometric (bool): Whether to automatically set isometric view (default: True)
    
    Returns:
    pyvista.Plotter: The configured plotter
    """
    # Add lighting
    add_default_lighting(plotter)
    
    # Add key bindings
    setup_key_bindings(plotter)
    
    # Set isometric view if requested
    if auto_isometric:
        set_isometric_view(plotter)
    
    return plotter

def estimate_scene_center_and_size(polydata):
    """
    Estimate the center and size of a scene based on polydata.
    
    Parameters:
    polydata (pyvista.PolyData): The polydata to analyze
    
    Returns:
    tuple: (center, size) - Center point and approximate size of the scene
    """
    if polydata is None:
        return (0, 0, 0), 100
    
    # Get bounds
    bounds = polydata.bounds
    
    # Calculate center
    center = (
        (bounds[0] + bounds[1]) / 2,  # x center
        (bounds[2] + bounds[3]) / 2,  # y center
        (bounds[4] + bounds[5]) / 2   # z center
    )
    
    # Calculate approximate size (diagonal of bounding box)
    size = np.sqrt(
        (bounds[1] - bounds[0])**2 + 
        (bounds[3] - bounds[2])**2 + 
        (bounds[5] - bounds[4])**2
    )
    
    # Ensure a minimum size
    size = max(size, 100)
    
    return center, size

def setup_camera_for_scene(plotter, polydata):
    """
    Setup the camera based on scene contents.
    
    Parameters:
    plotter (pyvista.Plotter): The plotter to configure
    polydata (pyvista.PolyData): The polydata to view
    
    Returns:
    pyvista.Plotter: The configured plotter
    """
    # Add all polydata to plotter temporarily to ensure bounds are set correctly
    if polydata is not None:
        actor = plotter.add_mesh(polydata, opacity=0)  # Add temporarily with opacity 0
    
    # Set up lighting
    add_default_lighting(plotter)
    
    # Use PyVista's built-in isometric view and reset camera to bounds
    set_isometric_view(plotter, focus_on_bounds=True)
    
    # Remove the temporary actor
    if polydata is not None:
        plotter.remove_actor(actor)
    
    # Add key bindings
    setup_key_bindings(plotter)
    
    return plotter 