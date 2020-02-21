
import sys
import os
sys.path.insert(0, os.path.abspath('../forestpy/forestpy'))
import forest

# data_path = '/Volumes/toshi-stati/data/V1/'

# feature_root = os.path.join(data_path, 'master-project/training/ACP/v2/')

# label_file = os.path.join(
#     data_path,
#     'thermokarst/initiation-regions/ACP/v4/PDM-5var/without_predisp/multigrid/',
#     'ACP-TKI-PDM5.yml'
# )

# baseline_file = 'baseline/multigrid/ACP-training-base.yml' 

# from harddrive_acp_paths import *
# from ocotal_acp_paths import *

from ocotal_sp_paths import *

all_vars = []

# baseline 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss5_tdp.75.yml',
delta_e = [

    {
        'name': 'rfm_e'+ str(n) + '_md100_mfAUTO_mln50000_msl8_mss5_tdp75.yml',
        'features_file': os.path.join( 
            feature_root,baseline_file
        ),
        'label_file': label_file ,
        'percent':.75
    } for n in [1, 4, 12, 25, 60, 75, 100]

]

all_vars += delta_e

# baseline 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss5_tdp.75.yml',
delta_md = [

    {
        'name': 'rfm_e50_md'+ str(n) + '_mfAUTO_mln50000_msl8_mss5_tdp75.yml',
        'features_file': os.path.join( 
            feature_root,baseline_file
        ),
        'label_file': label_file ,
        'percent':.75
    } for n in  range(17)

] 

delta_md.append(
    {
        'name': 'rfm_e50_md17_mfAUTO_msl8_mss5_tdp75.yml',
        'features_file': os.path.join( 
            feature_root,baseline_file
        ),
        'label_file': label_file ,
        'percent':.75
    },
)

all_vars += delta_md


# baseline 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss5_tdp.75.yml',
delta_mf = [

    {
        'name': 'rfm_e50_md100_mf'+ str(n) + '_mln50000_msl8_mss5_tdp75.yml',
        'features_file': os.path.join( 
            feature_root,baseline_file
        ),
        'label_file': label_file ,
        'percent':.75
    } for n in  ['AUTO', 'SQRT', 'LOG2', 1,3,4,5,6, 8, 10, 12]

] 

all_vars += delta_mf

# baseline 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss5_tdp.75.yml',
delta_mln = [

    {
        'name': 'rfm_e50_md100_mfAUTO_mln'+ str(n) + '_msl8_mss5_tdp75.yml',
        'features_file': os.path.join( 
            feature_root,baseline_file
        ),
        'label_file': label_file ,
        'percent':.75
    } for n in  [50000*s for s in [.25, .5, .75, 1.25, 1.5, 2]]

] 
all_vars += delta_mln

# baseline 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss5_tdp.75.yml',
# min sample leaf
delta_msl = [

    {
        'name': 'rfm_e50_md100_mfAUTO_mln50000_msl'+ str(n) + '_mss5_tdp75.yml',
        'features_file': os.path.join( 
            feature_root,baseline_file
        ),
        'label_file': label_file ,
        'percent':.75
    } for n in  [1, 4, 12, 16]

] 
all_vars += delta_msl

# baseline 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss5_tdp.75.yml',
# min sample split
delta_mss = [

    {
        'name': 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss'+ str(n) + '_tdp75.yml',
        'features_file': os.path.join( 
            feature_root,baseline_file
        ),
        'label_file': label_file ,
        'percent':.75
    } for n in  [2, 7, 10, 15, 25 ]

] 
all_vars += delta_mss


# baseline 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss5_tdp.75.yml',
delta_tdp = [

    {
        'name': 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss5_tdp'+ str(n) + '.yml',
        'features_file': os.path.join( 
            feature_root,baseline_file
        ),
        'label_file': label_file ,
        'percent': n/100
    } for n in  [25, 50 ]

] 
all_vars += delta_tdp

# baseline 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss5_tdp.75.yml',
# max samples
delta_ms = [

    {
        'name': 'rfm_e50_md100_mfAUTO_mln50000_msl8_mss5_ms'+str(n)+'_tdp75.yml',
        'features_file': os.path.join( 
            feature_root,baseline_file
        ),
        'label_file': label_file ,
        'percent': .75
    } for n in  [.25, .50, .75]

] 
all_vars += delta_ms



for sam in all_vars:
    print(sam['name'])
    forest.RFParams(sam['name'])
