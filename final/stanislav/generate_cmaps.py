import json
from matplotlib import cm, colors
from pathlib import Path

def generate_paraview_colormap(name, cmap, num_points=256):
    """
    Generate a ParaView-compatible colormap definition from a matplotlib colormap.
    
    Args:
        name (str): Name of the colormap
        cmap: Matplotlib colormap object
        num_points (int): Number of points in the colormap (default: 256)
    
    Returns:
        dict: ParaView colormap definition
    """
    # Ensure we have valid RGB values
    color_points = []
    for i in range(num_points):
        x = i / (num_points - 1)  # Normalized position
        rgba = cmap(x)
        
        # Ensure RGB values are within [0, 1]
        r = max(0.0, min(1.0, float(rgba[0])))
        g = max(0.0, min(1.0, float(rgba[1])))
        b = max(0.0, min(1.0, float(rgba[2])))
        
        color_points.extend([float(x), r, g, b])

    return {
        "Name": name,
        "RGBPoints": color_points,
        "NanColor": [0.5, 0.5, 0.5],
        "ColorSpace": "RGB",
        "DefaultMap": False,
        "Categories": ["Brewer"],
        "Points": [0, 0, 0.5, 0, 1, 1, 0.5, 0]  # Required by ParaView
    }

def create_colormaps_file(output_path):
    """
    Create a JSON file containing ParaView-compatible Brewer colormaps.
    
    Args:
        output_path (Path): Path where the JSON file will be saved
    """
    # Get only valid Brewer colormaps
    brewer_colormaps = {}
    for name in cm.cmaps_listed:
        if "brewer" in name.lower():
            try:
                cmap = cm.get_cmap(name)
                brewer_colormaps[name] = cmap
            except ValueError:
                print(f"Warning: Could not load colormap {name}")

    # Create the colormap definitions
    paraview_colormaps = {
        "ColorMaps": [
            generate_paraview_colormap(name, cmap) 
            for name, cmap in brewer_colormaps.items()
        ]
    }

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to JSON file
    with open(output_path, "w") as f:
        json.dump(paraview_colormaps, f, indent=2)

    print(f"Successfully wrote {len(brewer_colormaps)} Brewer colormaps to {output_path}")

# Usage
if __name__ == "__main__":
    output_path = Path('data/revised/final/stanislav/brewer_colormaps.json')
    create_colormaps_file(output_path)