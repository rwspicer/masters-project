import os

data_path = '/home/ross/rf-data'

feature_root = os.path.join(data_path, 'features')

label_file = os.path.join(
    data_path,
    'labels',
    'ACP-TKI-PDM5.yml'
)


baseline_file = 'baseline/ACP-training-base.yml' 

save_path = '../../rf-data'
