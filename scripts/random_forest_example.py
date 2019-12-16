"""random_forest_example.py

Example on how to train a random forest
"""
from atm.images import raster
from multigrids import temporal_grid, temporal
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz

from datetime import datetime

data_path = 'data/degree-day/ACP'
fdd = temporal_grid.TemporalGrid(os.path.join(data_path, 'freezing', 'TemporalGrid', 'ACP-FDD.yml'))

tdd = temporal_grid.TemporalGrid(os.path.join(data_path, 'thawing', 'TemporalGrid', 'ACP-TDD.yml'))

data_path = 'data/precipitation/ACP'
ewp = temporal_grid.TemporalGrid(os.path.join(data_path, 'early-winter', 'TemporalGrid', 'precip-1901-2015-early-winter.yml'))

fwp = temporal_grid.TemporalGrid(os.path.join(data_path, 'full-winter', 'TemporalGrid', 'winter_precip.yml'))
fwp.config['start_timestep'] = 1901



## create a new mg temporal grid for the tki
tki = temporal_grid.TemporalGrid(ewp.config['grid_shape'][0], ewp.config['grid_shape'][1],ewp.config['num_timesteps'])
# tki.config['']
tki.config['start_timestep'] = 1901

tki_files = sorted(
    glob.glob(
        'data/thermokarst-initiation-regions/method-2/EWP_FWP_M2/1901-1950/*_initialization_areas.tif'
    )
)

for ix, val in enumerate(tki_files):
    yr = ix
    data, md = raster.load_raster(val)
    tki[yr] = data
    tki.config['raster_metadata'] = md

# full = range(1901, 2016)
train_sum = range(1902, 1951) # ends at 1950 traing range for summers
train_wint = range(1901, 1950)

mask = np.logical_not(np.isnan(ewp[1901]))

ewp_feat = ewp.get_as_ml_features(None, mask, train_wint)
fwp_feat = fwp.get_as_ml_features(None, mask, train_wint)
fdd_feat = fdd.get_as_ml_features(None, mask, train_wint)

tdd_feat = tdd.get_as_ml_features(None, mask, train_sum)

tki_labels = tki.get_as_ml_features(None, mask, train_sum)

train_features = np.array([ewp_feat,fwp_feat,fdd_feat,tdd_feat])
train_features.shape


def train_rf(train_features, tki_labels):
    """
    """
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42, verbose=1, n_jobs=4, max_depth=20)

    start = datetime.now()
    rf.fit(train_features.T, tki_labels)
    total = datetime.now() - start

    return rf, total
