import os
data_path = '/Volumes/toshi-stati/data/V1/'

feature_root = os.path.join(data_path, 'master-project/training/ACP/v2/')

label_file = os.path.join(
    data_path,
    'thermokarst/initiation-regions/ACP/v4/PDM-5var/without_predisp/multigrid/',
    'ACP-TKI-PDM5.yml'
)

baseline_file = 'baseline/multigrid/ACP-training-base.yml' 

save_path = './'
