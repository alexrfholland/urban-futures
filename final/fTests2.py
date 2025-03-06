import pickle
voxel_size = .5
print(f'Loading tree templates of voxel size {voxel_size}')
tree_templates = pickle.load(open(f'data/revised/trees/{voxel_size}_voxel_tree_dict.pkl', 'rb'))
#print all keys in tree_templates
print(tree_templates.keys())