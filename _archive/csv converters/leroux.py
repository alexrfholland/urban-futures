import pandas as pd

# Load the LeRoux dataset from the provided CSV file
leroux_df = pd.read_csv('data/csvs/lerouxdata-long.csv')

# Create a mapping dictionary for renaming attributes
attribute_mapping = {
    '% of peeling bark cover on trunk/limbs': 'peeling bark',
    'Number of epiphytes': 'epiphyte',
    'Number of hollows': 'hollow',
    'Number of fallen logs (> 10 cm DBH 10 m radius of tree)': 'fallen log',
    '% of litter cover (10 m radius of tree)': 'leaf litter',
    '% of dead branches in canopy': 'dead branch'
}

# Add the new 'name' column based on the 'Attribute' column
leroux_df['name'] = leroux_df['Attribute'].map(attribute_mapping)

# Display the updated DataFrame
leroux_df.head()

print(leroux_df)
leroux_df.to_csv('data/csvs/lerouxdata-update.csv')