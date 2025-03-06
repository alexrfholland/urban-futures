import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq
import json
from pathlib import Path

def export_nested_templates(template_df, output_dir, format='split', prefix='template'):
    """
    Export a DataFrame containing nested DataFrames in the 'template' column.
    
    Args:
        template_df (pd.DataFrame): DataFrame containing nested DataFrames in 'template' column
        output_dir (str or Path): Directory to save the outputs
        format (str): Export format - one of 'split', 'json', or 'parquet'
        prefix (str): Prefix for file names (default: 'template')
        
    Returns:
        dict: Dictionary containing paths to exported files
        
    Examples:
        >>> paths = export_nested_templates(
        ...     voxelised_templates, 
        ...     'data/revised/trees',
        ...     format='split'
        ... )
    """
    # Convert output_dir to Path object
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store output paths
    output_paths = {}
    
    if format == 'split':
        # Create templates subdirectory
        template_dir = output_dir / f'{prefix}_templates'
        template_dir.mkdir(exist_ok=True)
        
        # Save metadata (everything except template column)
        metadata = template_df.drop('template', axis=1)
        metadata_path = output_dir / f'{prefix}_metadata.feather'
        metadata.to_feather(metadata_path)
        output_paths['metadata'] = metadata_path
        
        # Save template index mapping
        template_mapping = pd.DataFrame({
            'index': template_df.index,
            'template_file': [f'template_{i}.feather' for i in range(len(template_df))]
        })
        mapping_path = output_dir / f'{prefix}_mapping.feather'
        template_mapping.to_feather(mapping_path)
        output_paths['mapping'] = mapping_path
        
        # Save each template DataFrame separately
        template_paths = []
        for idx, row in template_df.iterrows():
            template_path = template_dir / f'template_{idx}.feather'
            row['template'].to_feather(template_path)
            template_paths.append(template_path)
        output_paths['templates'] = template_paths
            
    elif format == 'json':
        # Function to convert DataFrame to dict
        def df_to_dict(df):
            return df.to_dict(orient='records')
        
        # Convert template DataFrames to dictionaries
        json_df = template_df.copy()
        json_df['template'] = json_df['template'].apply(df_to_dict)
        
        # Save as JSON
        json_path = output_dir / f'{prefix}.json'
        json_df.to_json(json_path, orient='records')
        output_paths['json'] = json_path
        
    elif format == 'parquet':
        try:
            # Attempt direct parquet conversion
            parquet_path = output_dir / f'{prefix}.parquet'
            template_df.to_parquet(parquet_path)
            output_paths['parquet'] = parquet_path
        except Exception as e:
            print(f"Direct parquet conversion failed: {e}")
            print("Falling back to split format...")
            return export_nested_templates(template_df, output_dir, format='split', prefix=prefix)
            
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'split', 'json', or 'parquet'")
    
    # Print summary of exports
    print(f"\nExported nested templates in {format} format:")
    for key, path in output_paths.items():
        if isinstance(path, list):
            print(f"{key}: {len(path)} files in {path[0].parent}")
        else:
            print(f"{key}: {path}")
            
    return output_paths

def import_nested_templates(input_dir, format='split', prefix='template'):
    """
    Import previously exported nested template structure.
    
    Args:
        input_dir (str or Path): Directory containing the exported files
        format (str): Format to import - must match export format
        prefix (str): Prefix used in export (default: 'template')
        
    Returns:
        pd.DataFrame: Reconstructed DataFrame with nested templates
        
    Examples:
        >>> df = import_nested_templates(
        ...     'data/revised/trees',
        ...     format='split'
        ... )
    """
    input_dir = Path(input_dir)
    
    if format == 'split':
        # Load metadata
        metadata = pd.read_feather(input_dir / f'{prefix}_metadata.feather')
        
        # Load mapping
        mapping = pd.read_feather(input_dir / f'{prefix}_mapping.feather')
        
        # Load templates
        template_dir = input_dir / f'{prefix}_templates'
        templates = {}
        for idx in mapping['index']:
            template_path = template_dir / f'template_{idx}.feather'
            templates[idx] = pd.read_feather(template_path)
            
        # Reconstruct DataFrame
        metadata['template'] = metadata.index.map(templates.get)
        return metadata
        
    elif format == 'json':
        # Load JSON
        df = pd.read_json(input_dir / f'{prefix}.json', orient='records')
        # Convert template dictionaries back to DataFrames
        df['template'] = df['template'].apply(pd.DataFrame.from_records)
        return df
        
    elif format == 'parquet':
        try:
            return pd.read_parquet(input_dir / f'{prefix}.parquet')
        except Exception as e:
            print(f"Direct parquet import failed: {e}")
            print("Falling back to split format...")
            return import_nested_templates(input_dir, format='split', prefix=prefix)
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'split', 'json', or 'parquet'")