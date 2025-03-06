import pyvista as pv
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import os

def get_user_tree_attributes():
    # Get precolonial status
    while True:
        pre_choice = input("Select precolonial status:\n1) precolonial = False\n2) precolonial = True\nChoice: ")
        if pre_choice in ['1', '2']:
            precolonial = True if pre_choice == '2' else False
            break
        print("Invalid choice. Please select 1 or 2.")

    # Get control type
    while True:
        control_choice = input("\nSelect control type:\n1) street-tree\n2) park-tree\n3) reserve-tree\nChoice: ")
        if control_choice in ['1', '2', '3']:
            control_map = {'1': 'street-tree', '2': 'park-tree', '3': 'reserve-tree'}
            control = control_map[control_choice]
            break
        print("Invalid choice. Please select 1, 2, or 3.")

    # Get size
    while True:
        size_choice = input("\nSelect size:\n1) small\n2) medium\n3) large\nChoice: ")
        if size_choice in ['1', '2', '3']:
            size_map = {'1': 'small', '2': 'medium', '3': 'large'}
            size = size_map[size_choice]
            break
        print("Invalid choice. Please select 1, 2, or 3.")

    return precolonial, control, size

def main():
    # Get site name
    site = input("Enter site name: ")
    filePATH = f'data/revised/final/{site}'

    # Ask whether to reset or add
    while True:
        mode = input("\nDo you want to:\n1) Reset additional poles and trees\n2) Add to existing poles and trees\nChoice: ")
        if mode in ['1', '2']:
            break
        print("Invalid choice. Please select 1 or 2.")

    # Initialize data storage
    pole_positions = []
    tree_data = []

    # Load existing data if in add mode
    if mode == '2':
        pole_file = f'{filePATH}/{site}-extraPoleDF.csv'
        tree_file = f'{filePATH}/{site}-extraTreeDF.csv'
        
        if os.path.exists(pole_file):
            pole_df = pd.read_csv(pole_file)
            pole_positions = pole_df[['x', 'y', 'z']].values.tolist()
            print(f"Loaded {len(pole_positions)} existing poles")
            
        if os.path.exists(tree_file):
            tree_df = pd.read_csv(tree_file)
            tree_data = tree_df.to_dict('records')
            print(f"Loaded {len(tree_data)} existing trees")

    # Load the VTK file
    vtk_file = f'{filePATH}/{site}_positive_1_scenarioYR0.vtk'
    polydata = pv.read(vtk_file)

    # Create KDTree for vertex picking
    all_vertices = np.array(polydata.points)
    vertex_tree = KDTree(all_vertices)

    # Ask user which type to add
    while True:
        add_type = input("\nWhat would you like to add?\n1) Poles\n2) Trees\n3) Both\nChoice: ")
        if add_type in ['1', '2', '3']:
            break
        print("Invalid choice. Please select 1, 2, or 3.")

    def pick_poles():
        nonlocal pole_positions
        
        plotter = pv.Plotter()
        plotter.add_mesh(polydata, 
                        scalars = 'analysis_combined_resistance',
                        point_size=5, 
                        render_points_as_spheres=True, 
                        opacity=0.5)
        plotter.add_axes()

        # Show existing poles with larger size
        if pole_positions:
            pole_points = np.array(pole_positions)
            plotter.add_points(pole_points, 
                             color='red', 
                             point_size=20, 
                             render_points_as_spheres=True)
            for i, pos in enumerate(pole_positions):
                plotter.add_point_labels([pos], 
                                       [f"Pole {i+1}"], 
                                       point_size=30, 
                                       font_size=14)

        def callback(point, picker):
            if picker.GetActor() is None:
                print("No mesh picked")
                return

            picked_position = picker.GetPickPosition()
            distance, index = vertex_tree.query(picked_position)
            nearest_vertex = all_vertices[index]

            pole_positions.append(list(nearest_vertex))
            plotter.add_point_labels([nearest_vertex], 
                                   [f"Pole {len(pole_positions)}"], 
                                   point_size=20, 
                                   font_size=10)
            print(f"Pole added at {nearest_vertex}")
            plotter.render()

        plotter.enable_point_picking(
            callback=callback,
            show_message=True,
            tolerance=0.01,
            use_picker=True,
            pickable_window=False,
            show_point=True,
            point_size=20,
            picker='point',
            font_size=20
        )

        plotter.add_text("Select pole locations.\nPress 'q' to finish", 
                        position='upper_left', 
                        font_size=12)
        
        plotter.enable_eye_dome_lighting()
        
        plotter.show()

    def pick_trees():
        nonlocal tree_data
        
        plotter = pv.Plotter()
        plotter.add_mesh(polydata, 
                        scalars = 'analysis_combined_resistance',
                        point_size=5, 
                        render_points_as_spheres=True, 
                        opacity=0.5)
        plotter.add_axes()

        # Show existing poles with larger size
        if pole_positions:
            pole_points = np.array(pole_positions)
            plotter.add_points(pole_points, 
                             color='red', 
                             point_size=20, 
                             render_points_as_spheres=True)
            for i, pos in enumerate(pole_positions):
                plotter.add_point_labels([pos], 
                                       [f"Pole {i+1}"], 
                                       point_size=30, 
                                       font_size=14)

        # Show existing trees
        if tree_data:
            tree_points = np.array([[tree['x'], tree['y'], tree['z']] for tree in tree_data])
            plotter.add_points(tree_points, 
                             color='green', 
                             point_size=20, 
                             render_points_as_spheres=True)
            for i, tree in enumerate(tree_data):
                plotter.add_point_labels([[tree['x'], tree['y'], tree['z']]], 
                                       [f"Tree {i+1}"], 
                                       point_size=30, 
                                       font_size=14)

        def callback(point, picker):
            if picker.GetActor() is None:
                print("No mesh picked")
                return

            picked_position = picker.GetPickPosition()
            distance, index = vertex_tree.query(picked_position)
            nearest_vertex = all_vertices[index]
            position = list(nearest_vertex)

            choice = input("\nSelect tree type:\n1) Default (street-tree, medium, non-precolonial)"
                          "\n2) Precolonial park-tree (small)"
                          "\n3) Large park-tree (non-precolonial)"
                          "\n4) Custom precolonial"
                          "\n5) Fully custom\nChoice: ")
            
            if choice == '1':
                tree_data.append({
                    'x': position[0], 'y': position[1], 'z': position[2],
                    'control': 'street-tree', 'size': 'medium', 'precolonial': False
                })
                print("Added tree with default values")
            elif choice == '2':
                tree_data.append({
                    'x': position[0], 'y': position[1], 'z': position[2],
                    'control': 'park-tree', 'size': 'small', 'precolonial': True
                })
                print("Added precolonial park-tree (small)")
            elif choice == '3':
                tree_data.append({
                    'x': position[0], 'y': position[1], 'z': position[2],
                    'control': 'park-tree', 'size': 'large', 'precolonial': False
                })
                print("Added large park-tree")
            elif choice == '4':
                _, control, size = get_user_tree_attributes()
                tree_data.append({
                    'x': position[0], 'y': position[1], 'z': position[2],
                    'control': control, 'size': size, 'precolonial': True
                })
                print("Added custom precolonial tree")
            elif choice == '5':
                precolonial, control, size = get_user_tree_attributes()
                tree_data.append({
                    'x': position[0], 'y': position[1], 'z': position[2],
                    'control': control, 'size': size, 'precolonial': precolonial
                })
                print("Added fully custom tree")

            plotter.add_point_labels([nearest_vertex], 
                                   [f"Tree {len(tree_data)}"], 
                                   point_size=20, 
                                   font_size=10)
            plotter.render()

        plotter.enable_point_picking(
            callback=callback,
            show_message=True,
            tolerance=0.01,
            use_picker=True,
            pickable_window=False,
            show_point=True,
            point_size=20,
            picker='point',
            font_size=20
        )

        plotter.add_text("Select tree locations.\nPress 'q' to finish", 
                        position='upper_left', 
                        font_size=12)
        
        plotter.enable_eye_dome_lighting()
        plotter.show()

    # Run the selected phases
    if add_type in ['1', '3']:
        print("\nPhase: Select locations for artificial trees (poles)")
        pick_poles()
    
    if add_type in ['2', '3']:
        print("\nPhase: Select locations for trees")
        pick_trees()

    # Save and verify results
    if pole_positions:
        pole_df = pd.DataFrame(pole_positions, columns=['x', 'y', 'z'])
        pole_df.to_csv(f'{filePATH}/{site}-extraPoleDF.csv', index=False)
        # Verify the save
        saved_pole_df = pd.read_csv(f'{filePATH}/{site}-extraPoleDF.csv')
        print(f"Saved and verified {len(saved_pole_df)} poles to extraPoleDF.csv")
        if len(saved_pole_df) != len(pole_positions):
            print("WARNING: Number of saved poles doesn't match original data!")

    if tree_data:
        tree_df = pd.DataFrame(tree_data)
        tree_df.to_csv(f'{filePATH}/{site}-extraTreeDF.csv', index=False)
        # Verify the save
        saved_tree_df = pd.read_csv(f'{filePATH}/{site}-extraTreeDF.csv')
        print(f"Saved and verified {len(saved_tree_df)} trees to extraTreeDF.csv")
        if len(saved_tree_df) != len(tree_data):
            print("WARNING: Number of saved trees doesn't match original data!")

if __name__ == "__main__":
    main()