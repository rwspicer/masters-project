import os

data_path = '/home/rspicer/rf-data-sp'

feature_root = os.path.join(data_path, 'features')

label_file = os.path.join(
    data_path,
    'labels',
    'SP-TKI-PDM5.yml'
)


baseline_file = 'baseline/SP-training-base.yml' 

save_path = '../../rf-data-sp'
