import sys
import os
sys.path.insert(0, os.path.abspath('../forestpy/forestpy'))
from datetime import datetime
import os
import joblib
import forest

import numpy as np
from multigrids import TemporalMultiGrid, TemporalGrid

# data_path = '/Volumes/toshi-stati/data/V1/'

# feature_root = os.path.join(data_path, 'master-project/training/ACP/v2/')

# label_file = os.path.join(
#     data_path,
#     'thermokarst/initiation-regions/ACP/v4/PDM-5var/without_predisp/multigrid/',
#     'ACP-TKI-PDM5.yml'
# )

from ocotal_sp_paths import *

items = [
    {
        'name': 'rfm_e4_md2_mfAUTO_tdp05.yml',
        'features_file': os.path.join( 
            feature_root,'with-labels/multigrid/with-labels.yml' 
        ),
        'label_file': label_file ,
        'percent':.05

    },
    {
        'name': 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss5_tdp.75.yml',
        'features_file': os.path.join( 
            feature_root,'with-labels/multigrid/with-labels.yml' 
        ),
        'label_file': label_file ,
        'percent':.75

    },
    {
        'name': 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss5_tdp.75.yml',
        'features_file': os.path.join( 
            feature_root,'with-random/multigrid/with-random.yml' 
        ),
        'label_file': label_file ,
        'percent':.75

    },
    {
        'name': 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss5_tdp.75.yml',
        'features_file': os.path.join( 
            feature_root,'no-geolocation/multigrid/no-geolocation.yml' 
        ),
        'label_file': label_file ,
        'percent':.75

    },
    {
        'name': 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss5_tdp.75.yml',
        'features_file': os.path.join( 
            feature_root,'without-elevation/multigrid/without-elevation.yml' 
        ),
        'label_file': label_file ,
        'percent':.75

    },
]


    



        

def go(items=items):

    grades = {}
 
    for sam in items:
        print(sam['name'])
        hyperparameters = forest.RFParams(sam['name'])

        f_grid = TemporalMultiGrid(sam['features_file'])
        l_grid = TemporalGrid(sam['label_file'])

        # if sam['percent'] in loaded_data:
        #     pass
        # else:
        #     print('loading data')
        print ('loading')
        features, labels, index = forest.setup(f_grid,l_grid, sam['percent'])
        test_features, labels_true = forest.format_data(f_grid, l_grid)
            #  = 
        print ("running_model")
        

        try:
            start = datetime.now()
            model = forest.create_model(
                features, 
                labels, 
                hyperparameters, 
                2, #verbosity
                12  #n_jobs
            )
            total_train = datetime.now() - start
        except:
            grades[save_name] = "this caused an error"
            continue


        start = datetime.now()
        labels_predicted = model.predict(test_features.T)
        total_predict = datetime.now() - start

        diff = labels_predicted - labels_true

        sa_name = os.path.split(sam['features_file'])[1]
        score = {
            'train_time': total_train,
            'predict_time': total_predict,
            'features': sa_name,
            'mean_error' : np.nanmean(diff),
            'median_error' : np.nanmedian(diff),
            'var_error' : np.nanvar(diff),
            'mean_abs_error' : np.abs(diff).mean(),
            'median_abs_error' : np.nanmedian(np.abs(diff)),
            'var_abs_error' : np.nanvar(np.abs(diff)),
            'r2' :  model.score(test_features.T, labels_true),
        }
        save_name = sa_name.split('.')[0] + '_' + sam['name'].split('.')[0] + '.joblib'
        grades[save_name] = score
        joblib.dump(model, os.path.join(save_path,save_name))
        print (score)

    return grades
