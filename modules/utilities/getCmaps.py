
import os
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import zipfile
import numpy as np

def create_colormaps():
    def xml_to_cmap(file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()

        cmap_entries = []
        for colormap in root.findall('ColorMap'):
            for point in colormap.findall('Point'):
                x = float(point.get('x'))
                r = float(point.get('r'))
                g = float(point.get('g'))
                b = float(point.get('b'))
                cmap_entries.append((x, (r, g, b)))

        cmap_entries.sort(key=lambda entry: entry[0])
        colors = [entry[1] for entry in cmap_entries]
        positions = [entry[0] for entry in cmap_entries]

        return LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))

    # Unzipping the file
    extraction_path = 'data'  # Change this to your folder path


    # Checking the contents of the extracted folder
    key_colormaps_path = os.path.join(extraction_path, 'cmaps')
    colormap_files = os.listdir(key_colormaps_path)

    # Filtering out only XML files from the list
    xml_colormap_files = [file for file in colormap_files if file.endswith('.xml')]

    # Creating colormaps from each XML file
    colormaps = {}
    for xml_file in xml_colormap_files:
        file_path = os.path.join(key_colormaps_path, xml_file)
        file_name = os.path.splitext(xml_file)[0]
        try:
            colormaps[file_name] = xml_to_cmap(file_path)
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")

    # Listing the names of successfully created colormaps
    print(list(colormaps.keys()))

    return colormaps

def plot_with_colormap(cmap_name, colormaps):
    """
    Plots mock data using the specified colormap.

    Parameters:
    cmap_name (str): The name of the colormap to use.
    colormaps (dict): Dictionary of colormaps.
    """
    if cmap_name not in colormaps:
        print(f"Colormap '{cmap_name}' not found.")
        return

    cmap = colormaps[cmap_name]
    # Generate some mock data
    x = np.random.rand(100)
    y = np.random.rand(100)
    colors = np.random.rand(100)

    # Create a scatter plot with the specified colormap
    plt.scatter(x, y, c=colors, cmap=cmap)
    plt.colorbar()  # Show color scale
    plt.title(f"Scatter Plot with '{cmap_name}' Colormap")
    plt.show()

"""colormaps = create_colormaps()
plot_with_colormap('other-outl-3', colormaps)"""