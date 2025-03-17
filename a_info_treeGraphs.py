import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def load_voxelized_templates(voxel_size):
    """
    Load the voxelized templates dataframe based on the specified voxel size.
    
    Args:
        voxel_size (float): The voxel size used for the templates
        
    Returns:
        pd.DataFrame: The loaded voxelized templates dataframe
    """
    # Format voxel size correctly: remove decimal for integers, keep decimal for floats
    if voxel_size == int(voxel_size):
        # It's an integer value like 1.0, format as "1"
        formatted_voxel = str(int(voxel_size))
    else:
        # It's a float value like 0.5, keep as is
        formatted_voxel = str(voxel_size)
    
    file_path = Path(f'data/revised/trees/combined_voxelSize_{formatted_voxel}_templateDF.pkl')
    
    if not file_path.exists():
        print(f"Looking for file: {file_path}")
        raise FileNotFoundError(f"Could not find file: {file_path}")
    
    print(f"Loading voxelized templates from {file_path}")
    return pd.read_pickle(file_path)

def aggregate_tree_stats(templates_df):
    """
    Aggregate statistics for each tree in the templates dataframe.
    
    Args:
        templates_df (pd.DataFrame): DataFrame containing tree templates
        
    Returns:
        pd.DataFrame: Aggregated statistics for each tree
    """
    stats_rows = []
    
    for idx, row in templates_df.iterrows():
        template = row['template']
        
        if template.empty:
            print(f"Skipping empty template for tree {row['tree_id']}")
            continue
        
        # Get metadata columns
        metadata = {
            'precolonial': row['precolonial'],
            'size': row['size'],
            'control': row['control'],
            'tree_id': row['tree_id']
        }
        
        # Get all stat columns
        stat_columns = [col for col in template.columns if col.startswith('stat_')]
        
        # If no stat columns, try using resource columns
        if not stat_columns:
            stat_columns = [col for col in template.columns if col.startswith('resource_')]
            
        if not stat_columns:
            print(f"No stat or resource columns found for tree {row['tree_id']}")
            continue
            
        # Calculate total voxels
        total_voxels = len(template)
        metadata['total_voxels'] = total_voxels
        
        # Calculate counts and percentages for each stat
        for stat_col in stat_columns:
            resource_name = stat_col.replace('stat_', '').replace('resource_', '')
            count = template[stat_col].sum()
            percentage = (count / total_voxels) * 100 if total_voxels > 0 else 0
            
            metadata[f'count_{resource_name}'] = count
            metadata[f'percent_{resource_name}'] = percentage
            
        stats_rows.append(metadata)
    
    # Create DataFrame from collected data
    stats_df = pd.DataFrame(stats_rows)
    
    return stats_df

def create_visualizations(stats_df, output_dir):
    """
    Create bar graphs for each resource count, faceted by tree size and precolonial status.
    
    Args:
        stats_df (pd.DataFrame): Aggregated statistics dataframe
        output_dir (Path): Directory to save the visualizations
    """
    # Get all count columns
    count_columns = [col for col in stats_df.columns if col.startswith('count_')]
    
    if not count_columns:
        print("No count columns found in the statistics dataframe")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the style
    sns.set(style="whitegrid")
    
    # Create a plot for each resource
    for count_col in count_columns:
        resource_name = count_col.replace('count_', '')
        percent_col = f'percent_{resource_name}'
        
        # Skip if the column doesn't exist
        if percent_col not in stats_df.columns:
            continue
            
        print(f"Creating visualization for {resource_name}")
        
        # Create a figure for this resource
        plt.figure(figsize=(15, 10))
        
        # Create the facet grid for counts
        g = sns.catplot(
            data=stats_df,
            x="size",
            y=count_col,
            hue="control",
            col="precolonial",
            kind="bar",
            palette="viridis",
            height=5,
            aspect=1.2
        )
        
        # Set titles and labels
        g.fig.suptitle(f"Count of {resource_name} by Tree Type", fontsize=16, y=1.02)
        g.set_axis_labels("Tree Size", f"Count of {resource_name}")
        g.set_titles(col_template="{col_name}")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        output_file = output_dir / f"{resource_name}_count.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        # Create the facet grid for percentages
        plt.figure(figsize=(15, 10))
        g = sns.catplot(
            data=stats_df,
            x="size",
            y=percent_col,
            hue="control",
            col="precolonial",
            kind="bar",
            palette="viridis",
            height=5,
            aspect=1.2
        )
        
        # Set titles and labels
        g.fig.suptitle(f"Percentage of {resource_name} by Tree Type", fontsize=16, y=1.02)
        g.set_axis_labels("Tree Size", f"Percentage of {resource_name}")
        g.set_titles(col_template="{col_name}")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        output_file = output_dir / f"{resource_name}_percent.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Saved visualizations for {resource_name}")
        
    # Create a summary visualization showing all resources by tree type
    print("Creating summary visualization")
    
    # Melt the dataframe to get it in the right format for faceting
    percent_cols = [col for col in stats_df.columns if col.startswith('percent_')]
    
    melted_df = pd.melt(
        stats_df,
        id_vars=['precolonial', 'size', 'control', 'tree_id', 'total_voxels'],
        value_vars=percent_cols,
        var_name='resource_type',
        value_name='percentage'
    )
    
    # Extract resource name from resource_type
    melted_df['resource'] = melted_df['resource_type'].str.replace('percent_', '')
    
    # Create a facet grid
    g = sns.catplot(
        data=melted_df,
        x="resource",
        y="percentage",
        hue="control",
        col="precolonial",
        row="size",
        kind="bar",
        palette="viridis",
        height=4,
        aspect=1.5,
        legend_out=True
    )
    
    # Rotate x-axis labels
    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Set titles
    g.fig.suptitle("Resource Distribution by Tree Type", fontsize=16, y=1.02)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels("Resource Type", "Percentage")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_file = output_dir / "resource_distribution_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved summary visualization to {output_file}")

def main():
    # Ask user for voxel size
    while True:
        try:
            voxel_size = float(input("Enter voxel size (e.g., 0.25, 0.5, 1.0): "))
            if voxel_size <= 0:
                print("Voxel size must be positive")
                continue
            break
        except ValueError:
            print("Please enter a valid number")
    
    # Define output directory
    output_dir = Path('data/revised/final/stats/arboreal-future-stats/data')
    
    try:
        # Load voxelized templates
        templates_df = load_voxelized_templates(voxel_size)
        
        # Aggregate statistics
        print("Aggregating tree statistics...")
        stats_df = aggregate_tree_stats(templates_df)
        
        if stats_df.empty:
            print("No statistics could be generated from the templates")
            return
        
        # Save statistics to CSV
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"tree_stats_{voxel_size}_voxel.csv"
        stats_df.to_csv(csv_path, index=False)
        print(f"Statistics saved to {csv_path}")
        
        # Create visualizations
        print("Creating visualizations...")
        create_visualizations(stats_df, output_dir)
        
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 