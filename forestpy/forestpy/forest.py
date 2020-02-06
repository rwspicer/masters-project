"""

"""
from sklearn.ensemble import RandomForestRegressor
import random_forest_tools as rft
import yaml
import copy
import os
from pandas import read_csv

import time

from multigrids import TemporalMultiGrid, TemporalGrid

from datetime import datetime, timedelta

import numpy as np

import rank_models as rank
import log_cleanup

class RFParams (dict):
    """
    """
    def __init__(self, parameters = {}):
        """Function Docs 
        Parameters
        ----------
        Returns
        -------
        """
        self.name_substitutions = {
            # 'estimators': 'e',
            'criterion': 'c',
            'random_state': 'rs',
            'max_depth': 'md',
            'min_samples_split': 'mss',
            'min_samples_leaf': 'msl',
            'min_weight_fraction_leaf': 'mwfl',
            'max_features': 'mf',
            'max_leaf_nodes': 'mln',
            'min_impurity_decrease': 'mid',
            'bootstrap': 'b',
            'oob_score': 'oobs',
            'warm_start': 'ws',
            'ccp_alpha': 'ccpa',
            'max_samples': 'ms',

            # 'training_data_percent': 'tdp',
            # 'version': 'v',
        }
        #setup reverse_map
        for k in list(self.name_substitutions.keys()):
            v = self.name_substitutions[k]
            self.name_substitutions[v] = k

        self.defaults = {
            'estimators': 10,
            'criterion': 'mse',
            'random_state': 42,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0,
            'max_features': 'auto',
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0,
            'bootstrap': True,
            'oob_score': False,
            'warm_start': False,
            'ccp_alpha': 0,
            'max_samples': None,
            'training_data_percent': 'x',
            'version': 1,
        }

        self.update(self.defaults)

        if type(parameters) is str:
            try:
                
                parameters = yaml.load(open(parameters, 'r'))
                self.update(parameters)
                # print('existing')
            except:
                # print('non-existing')
                parameters = os.path.split(parameters)[-1]
                # print('non-existing', parameters)
                self.settings_from_save_name(parameters)
        else:
            self.update(parameters)

       
        

    def settings_from_save_name(self, name):
        settings = name.split('_')
        # print(settings)
        for s in settings[1:-1]:
            # print(s)
            if 'tdp' == s[:3]:
                continue
            if 'e' == s[0]:
                 
                self['estimators'] = int(s[1:])
                continue
            for s_len in range(1,5)[::-1]:
                if (s[:s_len]) in self.name_substitutions:
                    setting, value = s[:s_len], s[s_len:]
                    setting = self.name_substitutions[setting]
                    try:
                        value = int(value)
                    except ValueError:
                        value = value.lower()
                        if value == 'false':
                            value = False
                        elif value == 'true':
                            value = True
                        elif value == 'none':
                            value = None
                    # print(setting, value)
                    self[setting] = value
                    break
                

    def build_save_name(self):
        name = 'rfm_e' + str(self['estimators'])

        for key in sorted(self.name_substitutions.keys()):
            if key not in self.defaults:
                # print(key)
                continue
            if str(self[key]) != str(self.defaults[key]):
                name += '_' + \
                    self.name_substitutions[key] + str(self[key]).upper()


        if self['training_data_percent'] != 'x':
            name += '_tdp' + str(self['training_data_percent']).upper()

        name +=  '_v' + str(self['version']) + '.yml' 
        return name

    def save(self, path = ''):
        """
        """
        name = self.build_save_name()
        yaml.dump(dict(self), open(os.path.join(path, name), 'w'))

        return os.path.join(path, name)


def create_model(features, labels, parameters, verbose=1, n_jobs=4):

    parameters['verbose'] = verbose
    parameters['n_jobs'] = n_jobs
    
    rf = RandomForestRegressor(
        n_estimators = parameters['estimators'], 
        random_state = parameters['random_state'],
        criterion = parameters['criterion'],   
        max_depth = parameters['max_depth'],
        min_samples_split = parameters['min_samples_split'],
        min_samples_leaf = parameters['min_samples_leaf'], 
        min_weight_fraction_leaf = parameters['min_weight_fraction_leaf'],
        max_features = parameters['max_features'],
        max_leaf_nodes = parameters['max_leaf_nodes'],
        min_impurity_decrease = parameters['min_impurity_decrease'],
        bootstrap= parameters['bootstrap'],
        oob_score= parameters['oob_score'],
        # max_samples = parameters['max_samples'], ## look this up
        warm_start = parameters['warm_start'],
        # ccp_alpha = parameters['ccp_alpha'],

        verbose=verbose, 
        n_jobs=n_jobs, 
        )

    # print('max features', rf.max_features)
    # print('max leaf nodes', rf.max_leaf_nodes)

    rf.fit(features.T, labels)
    return rf

def test_model(model):
    pass


def format_data(features, labels):
    # training_features_tmg = TemporalMultiGrid(features)
    start = features.config['start_timestep']
    end = start + features.config['num_timesteps']
    y_range = range(start+1, end)

    mask = features.config['mask']
    tf_array = features.get_as_ml_features(train_range=y_range)

    tl_array = labels.get_as_ml_features(None, mask, y_range)    
    
    return tf_array, tl_array



def setup(feature_grid, label_grid, ss_percent=.5):
    # feature_grid = TemporalMultiGrid(feature_file)
    # label_grid = TemporalGrid(label_file)

    features, labels = format_data(feature_grid, label_grid)

    ss_idx = rft.create_subsample_idx(features.shape, ss_percent)

    ss_features, ss_labels = rft.get_data_subsample(
        features, labels, idx = ss_idx
    )

    # feature_grid.config['subsample_index'] = ss_idx
    # label_grid.config['subsample_index'] = ss_idx


    return ss_features, ss_labels, ss_idx
    
def brute_force_task(ss_features, ss_labels, name):
    current_parameters = RFParams(name)

    model = create_model(ss_features, ss_labels, current_parameters,
                verbose=2, n_jobs=4)

    # model.fit(ss_features.T, ss_labels)

    return model

def brute_force_git_check_in(update, progress_file, computer):
    
    print ('Git check-in')

    rsp = os.popen('git pull').read()

    progress_frame = read_csv(progress_file, index_col=0)
    if not update is None:
        # computer,train time,predict time,diff mean,abs diff mean,diff var,median,mode
        progress_frame['computer'][update['name']] = update['computer']
        progress_frame['train time'][update['name']] = update['train time']
        progress_frame['predict time'][update['name']] = update['predict time']
        progress_frame['diff mean'][update['name']] = update['diff mean']
        progress_frame['abs diff mean'][update['name']] = update['abs diff mean']
        progress_frame['diff var'][update['name']] = update['diff var']
        progress_frame['abs diff var'][update['name']] = update['abs diff var']
        progress_frame['median'][update['name']] = update['median']
        progress_frame['mode'][update['name']] = update['mode']
        # progress_frame['mode'][update['name']] = update['mode']
        progress_frame['status'][update['name']] = 'complete'
        progress_frame['r^2'] = update['r^2']

    try:
        # print(progress_frame)
        _next = list(progress_frame[\
        progress_frame['status'] == 'not run'
        ]['status'].index)[0]
        print (_next)

        progress_frame['status'][_next] = 'in progress ' +  computer
    except  IndexError:
        _next = None
    progress_frame.to_csv(progress_file)

    log_cleanup.cleanup_no_git(progress_file)

    rsp = os.popen('git add ' + progress_file).read()

    if not update is None:
        rsp = os.popen(
            'git commit -m "Results for ' + update['name'] + str(datetime.now()) +'"'
        ).read()
    else:
        rsp = os.popen(
            'git commit -m "Starting ' + _next + ' ' + str(datetime.now()) +'"'
        ).read()

    rsp = os.popen('git pull').read()

    rsp = os.popen('git push').read()

    return _next

def evaluate_model(model, full_inputs, original_results):

    print ('generating predictions')
    start = datetime.now()
    model_predict = model.predict(full_inputs.T)
    total = datetime.now() - start
    print ('Evaluating predictions')

    
    diff = rank.find_diff(original_results, model_predict)
    
    ev = {}
    ev['prediction diff'] = diff
    ev['predict time'] = str(total)
    ev['diff mean'] = np.nanmean(diff)
    ev['diff var'] = np.nanvar(diff)
    ev['abs diff var'] = np.nanvar(np.abs(diff))
    ev['abs diff mean'] = np.abs(diff).mean()
    ev['mode'] = ''
    ev['median'] = np.nanmedian(diff)
    ev['r^2'] = model.score(full_inputs.T, model_predict)
     
    return ev

def setup_brute_force(feature_file, label_file):
    """Function includes some ugly hard coding
    """
    print('Loading data.')
    feature_grid = TemporalMultiGrid(feature_file)
    label_grid = TemporalGrid(label_file)

    ss_data_sets = {}
    print('Creating 25% subsample.')
    ss_features, ss_labels, ss_idx = setup(feature_grid,label_grid, .25)
    ss_data_sets['tdp25'] = {
        'features' : ss_features,
        'labels': ss_labels,
        'idx': ss_idx
    }
    print('Creating 50% subsample.')
    ss_features, ss_labels, ss_idx = setup(feature_grid,label_grid, .5)
    ss_data_sets['tdp50'] = {
        'features' : ss_features,
        'labels': ss_labels,
        'idx': ss_idx
    }
    print('Creating 75% subsample.')
    ss_features, ss_labels, ss_idx = setup(feature_grid,label_grid, .75)
    ss_data_sets['tdp75'] = {
        'features' : ss_features,
        'labels': ss_labels,
        'idx': ss_idx
    }
    print('Formating full data.')
    tf_array, tl_array = format_data(feature_grid, label_grid)
    ss_data_sets['full'] = {
        'features': tf_array,
        'labels': tl_array,
        'idx': 'N/A'
    }
    return ss_data_sets



def run_brute_force(computer, progress_file, ss_data_sets):
    update = None
    while True:
        print ('\n\n')
        print ('-' * 80)
        _next = brute_force_git_check_in(update, progress_file, computer)
        if _next == None:
            print("All current runs completed sleeping for 1 hour.")
            time.sleep(60 * 60)
            continue
        print("Generating random forest for ", _next + '.')

        tpd = _next[-9:-4]

        start = datetime.now()
        model = brute_force_task(
            ss_data_sets[tpd]['features'], 
            ss_data_sets[tpd]['labels'], 
            _next
        )
        total = datetime.now() - start

        print("Testing random forest for ", _next + '.')
        update = evaluate_model(
            model, 
            ss_data_sets['full']['features'], 
            ss_data_sets['full']['labels']
        )
        update['computer'] = computer
        update['train time'] = str(total)
        update['name'] = _next
        del(model)
        ## update is at top
 

def run(
        ss_features, ss_labels, base_parameters, vary_parameters, 
        max_train_time = timedelta(hours=4), 
        min_improvement = .01, verbose = 1, n_jobs=4
    ):
    """Not ready for use
    """

    model_ranking = []


    for parameter in vary_parameters:
        p_min, p_max, p_step = vary_parameters[parameter]
        current = p_min 
        
        while current < p_max:

            current_parameters = RFParams(dict(base_parameters))

            current_parameters[parameter] = current

            print("DEBUG forest.py(run):", current_parameters.build_save_name())
            
            start = datetime.now()
            
            model = create_model(ss_features, ss_labels, current_parameters,
                verbose=verbose, n_jobs=n_jobs)

            training_time = datetime.now() - start
            
            current_parameters['training time'] = str(training_time)
            print("DEBUG forest.py(run):", 'training time', str(training_time))


            if model_ranking == []:

                current_parameters['difference mean'] = 10
                current_parameters['difference variance'] = 1
                model_ranking.append(current_parameters)
            else:
                model_predict = model.predict(ss_features.T)
                rank.rank_model(
                    ss_labels, model_predict, current_parameters, model_ranking
                )

            print('------\n\n\n')
            current += p_step
            # if training_time > max_train_time:
            #     break

    return model_ranking
