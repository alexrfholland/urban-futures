import pandas as pd
import numpy as np

# The classify_points function provided
def classify_points(df, search, unclassified, returnIndex=True):
    """
    Classifies points in a DataFrame based on complex nested search criteria involving logical 
    conditions 'AND', 'OR', and 'NOT'. The function optionally returns an array indicating the 
    classification of each index in the DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data to be classified. It must have columns 
                             that correspond to the attributes specified in the search criteria.
    - search (dict): A nested dictionary enclosed in parentheses, defining the classification 
                     categories and their respective logical conditions. The dictionary structure 
                     facilitates intricate categorization based on multiple, interrelated conditions.
    - unclassified (str): the string to give unclassified points
    - returnIndex (bool, optional): If True, the function returns an array with each index in the 
                                    DataFrame labeled according to the classification criteria, or as 
                                    'unclassified' if it doesn't match any criteria. Defaults to False.

    Returns:
    - dict or numpy.ndarray: If returnIndex is False, the function returns a dictionary with category 
                             names as keys and sets of indices as values. If returnIndex is True, 
                             it returns an array with each DataFrame index labeled as per the 
                             classification or 'unclassified'.

    Search Criteria Structure:
    - The 'search' parameter should be structured as a nested dictionary enclosed in parentheses. 
      Each key represents a classification category with its value being another dictionary. This 
      inner dictionary should have a single key ('AND', 'OR', 'NOT') representing the logical 
      condition, and its value should be a list of conditions or further nested dictionaries for 
      more complex criteria.
    - Conditions are expressed as dictionaries with the DataFrame attribute name as the key and the 
      condition (specified as a range, exact value, etc.) as the value.
    - Example structure:
        (
            {
                "CategoryName": 
                {
                    "LogicalOperator": 
                    [
                        {"AttributeName": ["Value1", "Value2"]},
                        {"NOT": {"AttributeName": ["ExcludedValue"]}}
                    ]
                }
            }
        )
      In this structure, each category defines its criteria using logical operators and 
      attribute-based conditions.

    Example:
    - An example search criteria to classify points where "_Tree_size" is either "large" or "medium", 
      but not where "road_types-type" is "Carriageway" or "elevation" is between -20 and 80:
        search_criteria = (
            {
                "TreeSizeFilter": {
                    "AND": [
                        {"_Tree_size": ["large", "medium"]},
                        {"OR": [
                            {"NOT": {"road_types-type": ["Carriageway"]}},
                            {"NOT": {"elevation": [[-20, 80]]}}
                        ]}
                    ]
                }
            }
        )
        classified_array = classify_points(df, search_criteria, returnIndex=True)

      This example demonstrates how the function classifies rows in the DataFrame based on the 
      specified complex criteria and returns an array indicating the classification for each row.
    """
     
    def process_condition(df, condition, mode):
        all_indices = set(df.index)
        
        if mode in ['AND', 'OR']:
            valid_indices = all_indices if mode == 'AND' else set()
            for sub_condition in condition:
                if isinstance(sub_condition, dict):
                    sub_mode = next(iter(sub_condition))
                    sub_criteria = sub_condition[sub_mode]
                    indices = process_condition(df, sub_criteria, sub_mode)
                    valid_indices = valid_indices.intersection(indices) if mode == 'AND' else valid_indices.union(indices)
            return valid_indices

        elif mode == 'NOT':
            if isinstance(condition, dict):
                not_mode = next(iter(condition))
                not_criteria = condition[not_mode]
                not_indices = process_condition(df, not_criteria, not_mode)
                return all_indices.difference(not_indices)
            else:
                print("Expected a dictionary for NOT condition but got something else.")

        else:  # Direct attribute-value condition
            return set(process_search_values(df, mode, condition))

    def process_search_values(df, attr, values):
        indices = set()
        if attr not in df.columns:
            return indices

        for value in values:
            if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
                indices.update(df[(df[attr] >= value[0]) & (df[attr] <= value[1])].index)
            elif isinstance(value, (int, float, str)):
                indices.update(df[df[attr] == value].index)
            else:
                raise ValueError(f"Unsupported value type: {type(value)} in attribute '{attr}'")

        return indices

    if returnIndex:
        classification_array = [unclassified] * len(df)
    
    classifications = {}
    for category, criteria in search.items():
        mode = next(iter(criteria))
        category_indices = process_condition(df, criteria[mode], mode)
        classifications[category] = category_indices
        if returnIndex:
            for idx in category_indices:
                classification_array[idx] = category

    return classification_array if returnIndex else classifications



if __name__ == "__main__":
    # Sample DataFrame to simulate poly_data
    data = {
        'buildings-dip': np.random.uniform(0, 0.15, 100),
        'extensive_green_roof-RATING': np.random.choice(['Excellent', 'Good', 'Average', 'Poor'], 100),
        'intensive_green_roof-RATING': np.random.choice(['Excellent', 'Good', 'Average', 'Poor'], 100),
        'elevation': np.random.uniform(-30, 100, 100)
    }

    dataframe = pd.DataFrame(data)

    # Search criteria provided
    search_criteria = {
        "Load bearing": {
            "AND": [
                {"buildings-dip": [[0.0, 0.1]]},
                {"OR": [
                    {"extensive_green_roof-RATING": ["Excellent", "Good"]},
                    {"elevation": [[-20, 80]]}
                ]}
            ]
        },
        "Lightweight": {
            "AND": [
                {"buildings-dip": [[0.0, 0.1]]},
                {"NOT": {"intensive_green_roof-RATING": ["Poor"]}},
                {"elevation": [[-20, 80]]}
            ]
        }
    }


    search_criteria = {
        "Load bearing": {
            "AND": [
                {"buildings-dip": [[0.0, 0.1]]},
                {"OR": [
                    {"extensive_green_roof-RATING": ["Excellent", "Good"]},
                    {"elevation": [[-20, 80]]}
                ]}
            ]
        },
        "Lightweight": {
            "AND": [
                {"elevation": [[-20, 80]]}
            ]
        }
    }

    

    # Testing the updated function with returnIndex set to True
    classified_array = classify_points(dataframe, search_criteria, 'unclassified', returnIndex=True)

    # Displaying the first 20 elements of the classified_array to check the classifications
    print(classified_array[:20])





