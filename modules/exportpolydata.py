import pyvista as pv
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import pyvista as pv
import pyarrow.parquet as pq
import os

def process_block(block, parent_name='', data_accumulator=[]):
    for i in range(block.n_blocks):
        current_block = block[i]

        if isinstance(current_block, pv.MultiBlock):
            new_parent_name = f"{parent_name}{block.get_block_name(i)}_" if parent_name else ''
            process_block(current_block, new_parent_name, data_accumulator)
        else:
            points = current_block.points
            data_dict = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
            
            for name in current_block.point_data.keys():
                adjusted_name = f"{parent_name}{name}"
                attribute_data = current_block.point_data[name]

                if attribute_data.ndim > 1:
                    for j in range(attribute_data.shape[1]):
                        data_dict[f"{adjusted_name}_{j}"] = attribute_data[:, j]
                else:
                    data_dict[adjusted_name] = attribute_data

            data_accumulator.append(pd.DataFrame(data_dict))

def save_as_parquet(multi_block, output_file):

    # Process the blocks and accumulate the data in a list of DataFrames
    data_frames = []
    process_block(multi_block, data_accumulator=data_frames)

    # Concatenate all DataFrames
    full_df = pd.concat(data_frames, ignore_index=True)

    # Convert pandas DataFrame to PyArrow Table
    table = pa.Table.from_pandas(full_df)

    # Save as Parquet
    pq.write_table(table, output_file)
def loadSite(path):
        siteMultiBlock = pv.read(path)
        return siteMultiBlock

def saver(sites, states):
    # Usage:
    for site in sites:
        for state in states:
            print(f'loading vtm for {site}-{state}...')
            siteMultiBlock = loadSite(f'data/{site}/combined-{site}-{state}.vtm')
            print(f'finished loading vtm for {site}-{state}')
            outputData = f'data/{site}/resource_stats_{site}_{state}.parquet'
            save_as_parquet(siteMultiBlock, outputData)
            print(f'finished saving parquet for {site}-{state}')
            

sites = ['city', 'street', 'trimmed-parade','parade']
sites = ['parade']
states = ['now', 'trending','baseline']
#states = ['baseline']

def saverPreferred():
    # Usage:
    sites = ['city', 'street', 'trimmed-parade']

    for site in sites:
        print(f'loading vtm for {site}-preferred...')
        siteMultiBlock = loadSite(f'data/{site}/all_resources-{site}-now.vtm')
        print(f'finished loading vtm for {site}-preferred')
        outputData = f'data/{site}/resource_stats_{site}_preferred.parquet'
        save_as_parquet(siteMultiBlock, outputData)
        print(f'finished saving parquet for {site}-preferred')



#saver(sites, states)
saverPreferred()