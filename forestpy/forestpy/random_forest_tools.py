"""
Random Forest Tools
-------------------

random_forest_tools.py

tools for building, saving, loading, and viewing random scikit learn 
forest models

"""
import numpy as np
import matplotlib.pyplot as plt

def get_as_ml_features(self, mask = None, train_range = None):
    """TemporalMultiGrid version of get_as_ml_features,

    will be in next update to multigrids.TemporalMultiGrid

    Parameters
    ----------
    mask: np.array
        2d array of boolean values where true indicates data to get from
        the 2d array mask is applied to 
    train_range: list like, or None(default)
        [int, int], min an max years to get the features from the temporal 
        multigrid

    """
    features = [ [] for g in range(self.config['num_grids']) ]
    if mask is None:
        mask = self.config['mask']
    
    if train_range is None:
        train_range = self.get_range()
    
    for ts in train_range:
        for grid, gnum in self.config['grid_name_map'].items():
            features[gnum] += list(self[grid, ts][mask])

        
    return np.array(features)

def create_subsample_idx(shape, percent):
    """Function Docs 
    Parameters
    ----------
    Returns
    -------
    """
    if percent > 1:
        percent = 1
    _max = shape[1]
    keep = int(_max * percent)
    idx = np.random.randint(_max, size=keep)
    return idx

def get_data_subsample(features, labels, percent=.50, idx = None):
    """return a random subsample of data set
    
    using a “discrete uniform” distribution
    
    Parameters
    ----------
    features: np.array [n_features, n_samples]
        the features array
    labels: np.array [n_samples]
    percent: float [0, 1]
        percent of data to keep
        
    Returns 
    -------
    np.array[n_features, n_samples * percent]
        randomly subsampled feature array 
    np.array[n_features, n_samples * percent]
        randomly subsampled lable array using same 
        index as sampling feature array
    """
    if idx is None:
        idx = create_subsample_idx(features.shape, percent)
    
    return features.T[idx].T, labels[idx]

def apply_model_to_year(model, test_data, year, mask):
    """Apply the model to a years worth of data

    Parameters
    ----------
    model: scikit learn model
    test_data: TemporalMultigrid
    year: int
    mask: np.array
        shape is the same as each grid in test_data

    Returns
    -------
    np.array 
        the resulting map for a year
    """
    pf = test_data[year]
    test_f = []
    for f in pf:
        test_f.append(list(f[mask]))
    test_f = np.array(test_f)
    test_r = model.predict(test_f.T)

    test_map = np.zeros(test_data.config['grid_shape']) - np.nan
    test_map[mask]= test_r
    return test_map

def compare_year(original, new):
    """compute comparsion stats for a year

    Parameters
    ----------
    original: np.array 
        2d. 
    new: np.array 
        2d

    Returns
    -------
    """
    # new - original = 0 ... same
    # new - original > 0 ... new model is over predicting
    # new - original > 0 ...           ""      underpredicting

    diff = new - original
    ori_mod = np.abs(original)
    ori_mod[ori_mod < 1] = 1 

    m_diff = diff[np.logical_not(np.isnan(diff))]

    diff_stats = {
        'mean': m_diff.mean(),
        'max': m_diff.max(),
        'min': m_diff.min(),
        'var': m_diff.var(),
        'std': m_diff.std(),
        'median': np.median(m_diff),
        'diff_map': diff,
        '%_diff_map': diff/np.abs(original),
        'modified_%_diff_map': diff/ori_mod,
    }
    return diff_stats

def save_rf_model(model, filename):
    """saves model to pickle file

    Parameters
    ----------
    model: scikit learn model 
    file_path: path
        path to pickle file

    """
    with open(filename, 'wb') as f:
        pickle.dump(model,f)

def load_rf_model (file_path):
    """Loads a model from a pickle file
    
    Parameters
    ----------
    file_path: path
        path to pickle file
        
    Returns
    -------
    the sklearn model
    """
    with open(file_path, 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    return model

def generate_tree_svg(tree, feature_list, out_file = 'tree.svg'):
    """generate a image of the tree as svg
    
    Parameters
    ----------
    tree: pydot graph
    feature_list: list
        list of feature names (str)
    out_file: path, default 'tree.png'
        name/location of saved file
    """
    export_graphviz(
        tree, 
        out_file = out_file + '.dot', 
        feature_names = feature_list, 
        rounded = True, 
        precision = 1)
    (graph, ) = pydot.graph_from_dot_file(out_file + '.dot')
    # Write graph to a png file
    graph.write_svg(out_file)
       
def generate_tree_png(tree, feature_list, out_file = 'tree.png'):
    """generate a image of the tree as a png
    
    Parameters
    ----------
    tree: pydot graph
    feature_list: list
        list of feature names (str)
    out_file: path, default 'tree.png'
        name/location of saved file
    """
    export_graphviz(
        tree, 
        out_file = out_file + '.dot', 
        feature_names = feature_list, 
        rounded = True, 
        precision = 1
    )
    (graph, ) = pydot.graph_from_dot_file(out_file + '.dot')
    # Write graph to a png file
    graph.write_png(out_file)


def to_figure(raster_name, figure_name, title = "", cmap = 'viridis', 
        ticks = None, tick_labels=None, vmin=None,vmax=None, save=True
    ):
    """Converts a raster file, or 2d array, to a figure with colorbar and title

    Parameters
    ----------
    raster_name: path, or np.array
        raster file containing data, or 2D np.array of data
    figure_name: path
        output figure file
    title: str, default ""
    cmap: str or matplotlib colormap, default 'viridis'
    ticks: list, defaults None
        where the colorbat ticks are placed
    tick_labes: list, defaults None
        optional labels for the colorbar ticks
    vmin: Float or Int
    vmax: Float or Int
        min and max values to plot
    save: bool, defaults True
        shows figure if false
    """
    if type(raster_name) is str:
        data, md = load_raster(raster_name)
    else:
        data = raster_name
    imgplot = plt.matshow(data, cmap = cmap, vmin=vmin, vmax=vmax) 
    # imgplot.axes.get_xaxis().set_visible(False)
    # imgplot.axes.get_yaxis().set_visible(False)
    imgplot.axes.axis('off')
    cbar = plt.colorbar(shrink = .9, drawedges=False,  ticks=ticks) #[-1, 0, 1]
    # fig.colorbar(cax)
    if tick_labels:
        cbar.set_ticklabels(tick_labels)
        plt.clim(-0.5, len(tick_labels) - .5)
    plt.title(title, y=1.2)
    if save:
        plt.savefig(figure_name, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def quick_coolwarm (data, title, save =False):
    """generate a quick coolwarm plot

    cmap limits are [-50,50]

    Parameters
    ----------
    data: or np.array
        2D np.array of data
    title: str
    save: bool, defaults False
        saves figure as [title].png if true
    """
    to_figure(
        data, title + '.png', title, 'coolwarm', 
        vmin=-50, vmax=50, save=save
    )
