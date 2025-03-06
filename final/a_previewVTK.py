import pyvista as pv

# Step 1: Prompt the user for a file path to a VTK file
file_path = input("Enter the file path to the VTK file: ")

# Step 2: Read the VTK file using PyVista
mesh = pv.read(file_path)

# Step 3: Print all point_data attribute names
print("\nAvailable point data attributes:")
for name in mesh.point_data.keys():
    print(f"- {name}")

# Step 4: Prompt the user for a scalar field name
scalar_field = input("\nEnter the name of the scalar field to visualize: ")

# Step 5: Print all values of the selected scalar field
if scalar_field in mesh.point_data:
    scalar_values = mesh.point_data[scalar_field]
    print(f"\nValues of the '{scalar_field}' attribute:")
    print(scalar_values)
else:
    print(f"\nError: The scalar field '{scalar_field}' is not found in the point data.")
    exit()

# Step 6: Plot the VTK file with the entered scalar field
mesh.plot(scalars=scalar_field, show_scalar_bar=True)
