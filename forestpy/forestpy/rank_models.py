"""
Rank Models
-----------
rank_models.py

tools to rank the models

"""
import numpy as np 


def find_diff (original, new, absolute=False):
    """

    Parameters
    ----------
    original: np.array (shape N)
    new: np.array (shape N)
    absolute: bool, defaults False
        if true return absolute difference

    Returns
    -------
    np.array 
        new - original
    """
    if absolute:
        return np.abs(new-original)
    return new - original

def rank_model (base_predict, new_predict, new_model_meta, rank_list, 
        absolute=True):
    """ add a new model to the ranked results , based on difference mean
    and variance compared to new model. lower mean and variance values 
    rank higher

    Parameters
    ----------
    base_predict: np.array (shape N)
        predictions from the base model
    new_predict:  np.array (shape N)
        predictions from new model
    new_model_meta: dict:
        metadata for the new model, will be updated to contain 
        'difference mean' and 'difference variance' keys
    rank_list: list
        ranked list of models metadata, that is updated with new model metadata
    absolute: defaults True:
        use the absolute difference for raking purposes

    """
    diff = find_diff(base_predict, new_predict, False) # alaways want the real difference
    
    mean = np.nanmean(diff)
    var = np.nanvar(diff)
    if absolute:
        mean = np.abs(mean)

    new_model_meta['difference mean'] = mean
    new_model_meta['difference variance'] = var

    if len(rank_list) != 0:
        broke_out = False
        for i in range(len(rank_list)):
            current = rank_list[i]
            if mean < current['difference mean']: 
                rank_list.insert(i, new_model_meta)
                broke_out = True
                break
            elif mean == current['difference mean'] and \
                 var < current['difference variance']:
                rank_list.insert(i, new_model_meta)
                broke_out = True
                break

        if not broke_out:
           rank_list.append(new_model_meta) 

    else:
        rank_list.append(new_model_meta)
    
    
def test (absolute = True, min = 0, max = 10):
    """test function
    """
    TEST_BASE = np.zeros(10)
    TEST_DATA = [ np.random.randint(min, max ,size=10) for i in range(8)]
    rank = []
    for i in range(len(TEST_DATA)):
        model_meta = {"name": str(i)}
        rank_model(TEST_BASE, TEST_DATA[i], model_meta, rank, absolute)
    return rank, TEST_DATA
