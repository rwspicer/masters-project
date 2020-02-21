import os

data_path = '/home/ross/rf-data-sp'

feature_root = os.path.join(data_path, 'features')

label_file = os.path.join(
    data_path,
    'labels',
    'SP-TKI-PDM5.yml'
)


baseline_file = 'baseline/SP-rf-training-set.yml'

save_path = '../../rf-data-sp'
