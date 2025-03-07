import json
from paraview.simple import *

# Function to save key display properties to a JSON file
def save_important_display_settings(source, filename):
    display = GetDisplayProperties(source)
    
    # Retrieve key display properties
    settings = {}
    key_properties = [
        'Representation',  # Glyphs
        'MapScalars',  # Scalar coloring: yes/no
        'InterpolateScalarsBeforeMapping',  # Interpolation yes/no
        'Lighting',  # Lighting enabled
        'AmbientColor', 'DiffuseColor', 'SpecularColor',  # Colors
        'Ambient', 'Diffuse', 'Specular', 'SpecularPower',  # Lighting properties
        'Luminosity', 'Opacity',  # Opacity and Luminosity
        'GlyphType', 'ScaleMode', 'ScaleArray',  # Glyph settings
        'Transform',  # Transform
        'Position', 'Scale', 'Orientation',  # Transform positions and rotations
        # Add other critical properties here as needed
    ]
    
    # Attempt to save each key property
    for prop in key_properties:
        try:
            value = getattr(display, prop)
            # Attempt to serialize the property; skip if not serializable
            json.dumps(value)  # Try serializing the value to see if it's JSON compatible
            settings[prop] = value
        except Exception as e:
            print(f"Could not save property: {prop} - {e}")

    # Write properties to file
    with open(filename, 'w') as f:
        json.dump(settings, f, indent=4)

# Specify the sources for Trees, Rewilded, and Under_tree_treatments
trees_source = FindSource('Trees')
rewilded_source = FindSource('Rewilded')
under_tree_treatments_source = FindSource('Under_tree_treatments')

# Save the most important display settings for each source to the specified directory
save_important_display_settings(trees_source, '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/trimmed-parade/trees_display_settings.json')
save_important_display_settings(rewilded_source, '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/trimmed-parade/rewilded_display_settings.json')
save_important_display_settings(under_tree_treatments_source, '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/trimmed-parade/under_tree_treatments_display_settings.json')
