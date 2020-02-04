## imports 

import sys
import os
sys.path.insert(0, os.path.abspath('../forestpy/forestpy'))

import forest
from importlib import reload
reload(forest)
import os
from pandas import read_csv
import numpy as np

from datetime import datetime, timedelta

from multigrids import TemporalMultiGrid, TemporalGrid
import random_forest_tools as tools
reload(tools)
import matplotlib.pyplot as plt

data_dir = "/Users/rwspicer/Desktop/data/V1/"
feature_file = os.path.join(
    data_dir,
    "master-project/training/ACP/v1/temporal-multigrid/rf_traing_set_v1.yml"
)
label_file = os.path.join(
    data_dir,
    "thermokarst/initiation-regions/ACP/v3-1/PDM-5var/without_predisp/temporal-grid/ia_grid_5var.yml"
)

to_td = lambda x: timedelta(hours = int(x.split(':')[0]),minutes = int(x.split(':')[1]), seconds = float(x.split(':')[2]))
to_seconds = lambda x: int(x.split(':')[0]) *60 *60 + int(x.split(':')[1]) * 60 + float(x.split(':')[2])
to_min = lambda x: to_seconds(x) / 60
to_hour = lambda x: to_min(x) / 60


import pickle
import joblib
from pandas import DataFrame, read_csv

feature_importance_list = []
model_list = []

grid_names = ['fdd', 'tdd', 'tdd+1', 'ewp', 'fwp', 'sp', 'lsp', 'sp+1', 'lat', 'long','aspect','slope', 'elev' ]



def show_feature_importances(model, feature_list, show=False):
    # Get numerical feature importances
    importances = list(model.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance*100, 3)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    if show:
        [print('{:20} {}%'.format(*pair)) for pair in feature_importances];
        
    feature_importances = {f[0]:f[1] for f in feature_importances} 
    return feature_importances

hp_lookup = {
    'e': "number estimators", 
    'md': "max depth", 
    'mf': "max features", 
    'mln': "max leaf nodes", 
    'msl': "min samples leaf", 
    'mss': "min samples split", 
    'tdp' : "training data percent", 
    'computer': "computer"
}


import copy
for key, value in copy.copy(hp_lookup).items():
    hp_lookup[value] = key
