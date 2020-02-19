from datetime import datetime
import os
import joblib

data_path = '/home/rwspicer/data/V1'

feature_root = os.path.join(data_path, 'master-project/training/ACP/v2/')

label_file = os.path.join(
    data_path,
    'thermokarst/initiation-regions/ACP/v4/PDM-5var/without_predisp/multigrid/',
    'ACP-TKI-PDM5.yml'
)


items = [
    {
        'name': 'rfm_e4_md2_mfAUTO_tdp25.yml',
        'features_file': os.path.join( 
            feature_root,'with-labels/multigrid/with-labels.yml' 
        ),
        'label_file': label_file 
        'precent':75

    },
    {
        'name': 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss5_tdp75.yml',
        'features_file': os.path.join( 
            feature_root,'with-labels/multigrid/with-labels.yml' 
        ),
        'label_file': label_file 
        'precent':75

    },
    {
        'name': 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss5_tdp75.yml',
        'features_file': os.path.join( 
            feature_root,'with-random/multigrid/with-random.yml' 
        ),
        'label_file': label_file 
        'precent':75

    },
    {
        'name': 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss5_tdp75.yml',
        'features_file': os.path.join( 
            feature_root,'no-geolocation/multigrid/no-geolocation.yml' 
        ),
        'label_file': label_file 
        'precent':75

    },
    {
        'name': 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss5_tdp75.yml',
        'features_file': os.path.join( 
            feature_root,'without-elevation/multigrid/without-elevation.yml' 
        ),
        'label_file': label_file 
        'precent':75

    },
]


    


save_path = '../'
        

def go():

    grades = {}
    for sam in items:
        hyperparameters = RFParams(sam['name'])
        f_gird = TemporalMultiGrid(sam['features_file'])
        l_grid = TemporalGrid(sam['label_file'])



        features, labels, index = setup(f_gird,l_grid, sam['percent'])
        start = datetime.now()
        model = create_model(
            features, 
            labels, 
            hyperparameters, 
            2, # verbosity
            10 # n_jobs
        )
        total_train = datetime.now() - start
        
        test_features, labels_true = format_data(f_grid, l_grid)
        start = datetime.now()
        labels_predicted = model.predict(test_features.T)
        total_predict = datetime.now() - start

        diff = labels_predicted - labels_true

        sa_name =  os.path.split(sam['features_file'])[1]
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
            
            'r2' :  model.score(features.T, labels_true),
        }
        save_name = sa_name.split('.')[0] +'_'+ sam['name'].split('.')[0]+'.joblib'
        grades[save_name] = score
        joblib.dump(model, os.path.join(save_path,save_name))

    return grades
