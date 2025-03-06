import pyvista as pv
import h5py
import time
import numpy as np
import glyphs
import cameraSetUp



def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = te - ts  # Now logging time in seconds
        else:
            print(f'time: {method.__name__!r}  {((te - ts)):2.2f} s')  # Now printing time in seconds with 's' for seconds
        return result
    return timed

def save_multiblock_to_hdf5(multiblock, file_name):
    with h5py.File(file_name, 'w') as hdf_file:
        _save_multiblock(hdf_file, multiblock)

def _save_multiblock(hdf_group, multiblock):
    for idx, block_name in enumerate(multiblock.keys()):
        block = multiblock[block_name]
        if isinstance(block, pv.MultiBlock):
            subgroup = hdf_group.create_group(f'MultiBlock_{block_name}')
            _save_multiblock(subgroup, block)
        elif isinstance(block, pv.PolyData):
            poly_group = hdf_group.create_group(f'PolyData_{block_name}')
            poly_group.create_dataset('points', data=block.points)
            for field_name, array in block.point_data.items():
                if array.dtype.kind == 'U':  # Check if the data type is Unicode string
                    encoded_array = np.char.encode(array, 'utf-8')  # Encode to UTF-8
                    poly_group.create_dataset(f'point_data/{field_name}', data=encoded_array)
                else:
                    poly_group.create_dataset(f'point_data/{field_name}', data=array)
            for field_name, array in block.field_data.items():
                if array.dtype.kind == 'U':  # Check if the data type is Unicode string
                    encoded_array = np.char.encode(array, 'utf-8')  # Encode to UTF-8
                    poly_group.create_dataset(f'field_data/{field_name}', data=encoded_array)
                else:
                    poly_group.create_dataset(f'field_data/{field_name}', data=array)

def load_multiblock_from_hdf5(file_name):
    with h5py.File(file_name, 'r') as hdf_file:
        multiblock = _load_multiblock(hdf_file)
    return multiblock

def _load_multiblock(hdf_group):
    multiblock = pv.MultiBlock()
    for group_name, group in hdf_group.items():
        if group_name.startswith('MultiBlock_'):
            block_name = group_name[len('MultiBlock_'):]
            multiblock[block_name] = _load_multiblock(group)
        elif group_name.startswith('PolyData_'):
            block_name = group_name[len('PolyData_'):]
            polydata = pv.PolyData(group['points'][:])
            for dataset_name, dataset in group['point_data'].items():
                if dataset.dtype.kind == 'S':  # Check if the data type is a byte string
                    decoded_array = np.char.decode(dataset[:], 'utf-8')  # Decode from UTF-8
                    polydata.point_data[dataset_name] = decoded_array
                else:
                    polydata.point_data[dataset_name] = dataset[:]
            if 'field_data' in group:  # Check if 'field_data' group exists before iterating
                for dataset_name, dataset in group['field_data'].items():
                    if dataset.dtype.kind == 'S':  # Check if the data type is a byte string
                        decoded_array = np.char.decode(dataset[:], 'utf-8')  # Decode from UTF-8
                        polydata.field_data[dataset_name] = decoded_array
                    else:
                        polydata.field_data[dataset_name] = dataset[:]
            multiblock[block_name] = polydata
    return multiblock

# Usage:
# multi_block = load_multiblock_from_hdf5('data.h5')


# Usage:
# Assume 'multi_block' is your MultiBlock object
# save_multiblock_to_hdf5(multi_block, 'data.h5')

def read_multiblock_from_hdf5(file_name):
    with h5py.File(file_name, 'r') as hdf_file:
        return _read_multiblock(hdf_file)

def _read_multiblock(hdf_group):
    multiblock = pv.MultiBlock()
    for key, group in hdf_group.items():
        if key.startswith('MultiBlock_'):
            block_name = key[len('MultiBlock_'):]
            multiblock[block_name] = _read_multiblock(group)
        elif key.startswith('PolyData_'):
            block_name = key[len('PolyData_'):]
            polydata = pv.PolyData(group['points'][:])
            for field_name, dataset in group['point_data'].items():
                polydata.point_data[field_name] = dataset[:]
            for field_name, dataset in group['field_data'].items():
                polydata.field_data[field_name] = dataset[:]
            multiblock[block_name] = polydata
    return multiblock

# Usage:
# multi_block = read_multiblock_from_hdf5('data.h5')

@timeit
def main(sites, states):

    @timeit
    def loadSite(path):
        siteMultiBlock = pv.read(path)
        return siteMultiBlock

    # Usage:
    for siteNo, site in enumerate(sites):

        if (site != 'parade'):
            gridDist = 300
        else:
            gridDist = 800

        for stateNo, state in enumerate(states):
            print(f'processing {site} - {state}...')

            #siteMultiBlock = pv.read(f'data/{site}/combined-{site}-{state}.vtm')
            siteMultiBlock = loadSite(f'data/{site}/combined-{site}-{state}.vtm')
            print(f'VTM {site} - {state} loaded')
            save_multiblock_to_hdf5(siteMultiBlock, f'data/{site}/combined-{site}-{state}.h5')
            print(f'{site} - {state} HDf5 save')



@timeit
def view(sites, states):

    @timeit
    def loadSite(path):
        siteMultiBlock = pv.read(path)
        return siteMultiBlock

    plotter = pv.Plotter()

    # Usage:
    for siteNo, site in enumerate(sites):

        if (site != 'parade'):
            gridDist = 300
        else:
            gridDist = 800

        for stateNo, state in enumerate(states):
            print(f'processing {site} - {state}...')
            siteMultiBlock = load_multiblock_from_hdf5(f'data/{site}/combined-{site}-{state}.h5')
            print(f'{site} - {state} loaded')
            #print_block_info(siteMultiBlock)


            z_translation = siteMultiBlock.get('rest of points').bounds[4]
            translation_amount = np.array([gridDist * stateNo, gridDist * siteNo, z_translation])
            glyphs.plotSite(plotter, siteMultiBlock, translation_amount)
            print(f'added to plotter: {site} - {state}')

        

    # Additional settings for better visualization
    plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.4))
    light2 = pv.Light(light_type='cameralight', intensity=.4)
    light2.specular = 0.3  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()

    cameraSetUp.setup_camera(plotter, gridDist, 600)




    plotter.show()
if __name__ == "__main__":
    #sites = ['city', 'street', 'trimmed-parade']  # List of sites
    #sites = ['city', 'trimmed-parade']
    #sites = ['parade']
    #state = ['baseline', 'now', 'trending']
    sites = ['street']
    state = ['now', 'trending']
    #main(sites, state)
    view(sites, state)