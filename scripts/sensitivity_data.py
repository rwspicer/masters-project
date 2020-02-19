import os
import numpy as np
from multigrids import TemporalMultiGrid, TemporalGrid

def add_empty_grids(self, new_grids):
    rows = self.config['grid_shape'][0]
    cols = self.config['grid_shape'][1]
    num_ts = self.config['num_timesteps']
    n_grids = self.config['num_grids'] + len(new_grids)
    
    old_grids = list(self.config['grid_name_map'].keys())
    #print(  n_grids, old_grids, new_grids, list(old_grids) + list(new_grids))
    superset = TemporalMultiGrid(rows, cols, n_grids, num_ts,
        grid_names = old_grids + new_grids,
        data_type=self.config['data_type'],
        mask = self.config['mask'],
    )
    
    try:
        superset.config['description'] = \
            self.config['description'] + ' superset.'
    except KeyError:
        superset.config['description'] = 'Unknown superset.'

    try:
        superset.config['dataset_name'] = \
            superset.config['dataset_name'] + ' superset.'
    except KeyError:
        superset.config['dataset_name'] = 'Unknown superset.'

    superset.config['start_timestep'] = self.config['start_timestep']
    superset.config['timestep'] = self.config['timestep']

    for idx, grid in enumerate(old_grids):
        superset[grid][:] = self[grid][:]

    return superset

TemporalMultiGrid.add_empty_grids = add_empty_grids

def save_data (name, mg, save_dir):
    save_loc = os.path.join(save_dir, name, 'multigrid')
    try:
        os.makedirs(save_loc)
    except:
        pass
    mg.save(os.path.join(save_loc, name+'.yml'))

def generate(base_features, tweaks, save_dir ):

    for tweak in tweaks :
        subset_grids = list(base_features.config['grid_name_map'].keys())
        
        for g in tweak['remove']:
            subset_grids.remove(g)

        new_mg = base_features.create_subset(subset_grids)

        added = []
        for g in tweak['add']:
            added.append(g)
            new_mg = base_features.add_empty_grids(['labels',])
            new_mg['labels'][:] = tweak['add'][g]

        new_mg.config['dataset_name'] = 'ACP training data - ' + tweak['name']
        new_mg.config['description'] = 'ACP training data without ' + \
            str(tweak['remove']) + ", and with " + str(added)
        save_data(tweak['name'].replace(' ','-'), new_mg, save_dir)
        del(new_mg)

def build_tweaks(data_dir, label_file, shape = 'real_shape'):
    label_file = os.path.join(data_dir, label_file)

    labels = TemporalGrid(label_file)
    tweaks = [
        {
            "name": "no geolocation",
            "remove": ['lat','long'],
            "add": {}
        },
        {
            "name": "without elevation",
            "remove": ['aspect','slope', 'elevation'],
            "add": {}
        },
        {
            "name": "with labels",
            "remove": [],
            "add": {"labels": labels.grids.reshape(labels.config[shape])}
        },
        {
            "name": "with random",
            "remove": [],
            "add": {"random": np.random.random(labels.config[shape])}
        },
    ]
    return tweaks

