import rUtilities
import rUtilities_kde
import matplotlib.pyplot as plt

# Define search conditions
urban_system_search = {
        # Urban system classification conditions
        # Each key is a system type, and its value is a condition to be evaluated
        # Conditions use dataset attributes and can include:
        # - Comparisons: ==, !=, <, >, <=, >=
        # - Logical operators: &, |, ~ (NOT)
        # - Functions: .isin() for membership tests
        # - Attribute access: dataset['attribute_name']
        "Adaptable Vehicle Infrastructure": (
            "dataset['parkingmedian-isparkingmedian'] == True | "
            "(dataset['disturbance-potential'].isin([4, 2, 3])) & "
            "~((dataset['little_streets-islittle_streets'] == True) & "
            "(dataset['road_types-type'] == 'Footway'))"
        ),
        "Private empty space": (
            "dataset['disturbance-potential'] == 1"
        ),
        "Existing Canopies": (
            "dataset['_Tree_size'].isin(['large', 'medium']) & "
            "(dataset['road_types-type'] != 'Carriageway')"
        ),
        "Existing Canopies Under Roads": (
            "dataset['_Tree_size'].isin(['large', 'medium']) & "
            "(dataset['road_types-type'] == 'Carriageway')"
        ),
        "Street pylons": (
            "dataset['isstreetlight'] | dataset['ispylons']"
        ),
        "Load bearing roof": (
            "(dataset['buildings-dip'] >= 0.0) & "
            "(dataset['buildings-dip'] <= 0.1) & "
            "dataset['extensive_green_roof-RATING'].isin(['Excellent', 'Good', 'Moderate']) & "
            "(dataset['elevation'] >= -20) & "
            "(dataset['elevation'] <= 80)"
        ),
        "Lightweight roof": (
            "(dataset['buildings-dip'] >= 0.0) & "
            "(dataset['buildings-dip'] <= 0.1) & "
            "dataset['intensive_green_roof-RATING'].isin(['Excellent', 'Good', 'Moderate']) & "
            "(dataset['elevation'] >= -20) & "
            "(dataset['elevation'] <= 80)"
        ),
        "Ground floor facade": (
            "(dataset['buildings-dip'] >= 0.8) & "
            "(dataset['buildings-dip'] <= 1.7) & "
            "(dataset['solar'] >= 0.2) & "
            "(dataset['solar'] <= 1.0) & "
            "(dataset['elevation'] >= 0) & "
            "(dataset['elevation'] <= 10)"
        ),
        "Upper floor facade": (
            "(dataset['buildings-dip'] >= 0.8) & "
            "(dataset['buildings-dip'] <= 1.7) & "
            "(dataset['solar'] >= 0.2) & "
            "(dataset['solar'] <= 1.0) & "
            "(dataset['elevation'] >= 10) & "
            "(dataset['elevation'] <= 80)"
        )
    }


def main():
    # Example usage
    sites = ['street', 'city', 'trimmed-parade']
    site = sites[0]
    filename = f'data/{site}/updated-{site}.vtk'

    # Convert VTK to xarray
    site_data = rUtilities.vtk_to_xarray(filename)

    # Apply search conditions
    #classified_dataset = rUtilities.search(site_data, urban_system_search, 'urban systems')

    # Plot results
    rUtilities.plot_site(site_data, 'tree-weights')

def test_kde():
    # Example usage
    sites = ['street', 'city', 'trimmed-parade']
    site = sites[0]
    filename = f'data/{site}/updated-{site}.vtk'

    # Convert VTK to xarray
    site_data = rUtilities.vtk_to_xarray(filename)

    # Call classifyTrees
    bounds, tree_centers, tree_weights = rUtilities_kde.classifyTrees(site_data)
    
    # Compute and plot KDE for trees
    kde, grid_points, kdvalues = rUtilities_kde.weighted_2d_kde(bounds, tree_centers, tree_weights, cell_size=1)
    plt.figure(figsize=(10, 8))
    rUtilities_kde.plot_kde(grid_points, kdvalues, plot_type='contour')
    plt.title('Tree KDE')
    plt.show()

    # Call getDeployableGradient
    site_data_with_gradient = rUtilities_kde.getDeployableGradient(site_data)

    # Plot offensive score
    rUtilities.plot_site(site_data_with_gradient, 'offensiveScore')
    plt.title('Offensive Score')
    plt.show()

if __name__ == "__main__":
    test_kde()
    #main()
    