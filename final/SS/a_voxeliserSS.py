def verify_new_voxel_dataframe(new_voxel_df, ds):
    """
    Verifies that all required variables are present in new_voxel_df and have correct data types.

    Parameters:
        new_voxel_df (pd.DataFrame): DataFrame containing new voxels with data variables.
        ds (xr.Dataset): Existing xarray Dataset.

    Raises:
        ValueError: If any required variable is missing or has incorrect data type.
    """
    required_vars = [var for var in ds.data_vars if var not in ['voxel', 'voxel_I', 'voxel_J', 'voxel_K', 'centroid_x', 'centroid_y', 'centroid_z']]
    missing_vars = [var for var in required_vars if var not in new_voxel_df.columns]
    if missing_vars:
        logger.error(f"Missing variables in new_voxel_df: {missing_vars}")
        raise ValueError(f"Missing variables in new_voxel_df: {missing_vars}")
    else:
        logger.info("All required variables are present in new_voxel_df.")

    # Ensure data types match
    for var in required_vars:
        expected_dtype = ds[var].dtype
        actual_dtype = new_voxel_df[var].dtype
        if expected_dtype != actual_dtype:
            try:
                new_voxel_df[var] = new_voxel_df[var].astype(expected_dtype)
                logger.info(f"Converted '{var}' to dtype '{expected_dtype}'.")
            except Exception as e:
                logger.error(f"Error converting '{var}' to dtype '{expected_dtype}': {e}")
                raise