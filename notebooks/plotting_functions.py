import matplotlib.pyplot as plt

from scipy import optimize
import numpy as np

from matplotlib import rcParams
rcParams["font.family"] = "serif"
rcParams['font.serif'] = ['Times']
rcParams['font.size'] = 14

colors = [  'slateblue', 'darkblue', 'cornflowerblue','skyblue']
symbols = ['o','o','o','o']

linear = lambda  x, m, b: m*x + b
second = lambda x, a,b,c: a*x*x + b*x +c
third = lambda x, a,b,c, d: a*x*x*x + b*x*x + c*x +d 
fourth = lambda x, a,b,c, d, e: a*x*x*x*x + b*x*x*x + c*x*x +d*x +e

from datetime import datetime, timedelta
to_td = lambda x: timedelta(hours = int(x.split(':')[0]),minutes = int(x.split(':')[1]), seconds = float(x.split(':')[2]))
to_seconds = lambda x: int(x.split(':')[0]) *60 *60 + int(x.split(':')[1]) * 60 + float(x.split(':')[2])
to_min = lambda x: to_seconds(x) / 60
to_hour = lambda x: to_min(x) / 60

def scatter(
        data, x_col, y_col, 
        title = '',
        x_buffer = 1, y_buffer = 1,
        color = 'slateblue', symbol = 'o',
        figsize = (20,12),
        alpha = 1,
        x_label = None, y_label = None,
        fig=None, ax =None,
        label = None,
        markersize=10,
        lg_loc = 'upper left',
#         auto_axis = True
    ):
    """creates color coded scatter plot by param for abs diff mean
    """
    
    
#     vals = data[param].unique()
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize = figsize)
    
#     print( x_col, y_col)
    
#     if auto_axis:
    x_min = data[x_col].min()
    x_max = data[x_col].max()

    y_min = data[y_col].min()
    y_max = data[y_col].max()

#         print([x_min-x_buffer, x_max+x_buffer, y_min-y_buffer, y_max+y_buffer])

    ax.axis([x_min-x_buffer, x_max+x_buffer, y_min-y_buffer, y_max+y_buffer])
    
    data.plot.scatter(
        x=x_col, y=y_col, 
        color = color, marker=symbol, ax =ax, alpha = alpha, label=label,
        s =markersize
    )
    
  
    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)
    if not label is None:
        ax.legend(loc=lg_loc)
    ax.set_title(title)
    
    return fig, ax
    
def fit(data,  x_col, y_col,
        fit_min=None, fit_max = None,
        curve = linear,
        title = '',
        x_buffer = 1, y_buffer = 1,
        color = 'slateblue',
        figsize = (20,12),
        alpha = 1,
        x_label = None, y_label = None,
        fig=None, ax =None,
        label = None,
        linestyle='--'
    ):

    x_data = data[x_col]
    y_data = data[y_col]

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize = figsize)

    popt, pcov = optimize.curve_fit(curve, x_data, y_data)

    if fit_min is None and fit_max is None:
        x_data = np.linspace(x_data.min(), x_data.max(), 100)
    else:
        x_data = np.linspace(fit_min, fit_max, 100)
    ax.plot(x_data, curve(x_data, *popt), color=color, linestyle=linestyle)
    return fig, ax
    # plt.show()



def scatter_by_param(
        data, x_col, y_col, param,
        title = '',
        x_buffer = 1, y_buffer = 1,
        colors = colors, symbol = 'o',
        figsize = (20,12),
        alpha = 1,
        x_label = None, y_label = None,
        fig=None, ax =None,
        label = None,
    ):
    """creates color coded scatter plot by param for abs diff mean
    """
    
    vals = data[param].unique()
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize = figsize)
    #
    
    c_idx = 0
    for val in vals:
        print (val)
        if val is None:
            scatter(
                data[data[param].apply(str) == 'None'],
                x_col, y_col,
                fig = fig, ax = ax,
                color = colors[c_idx]
            )
    
        else:
            scatter(
                data[data[param] == val], 
                x_col, y_col,
                fig = fig, ax = ax,
                color = colors[c_idx]
            )
        c_idx +=1
         
    
    
    x_min = data[x_col].min()
    x_max = data[x_col].max()

    y_min = data[y_col].min()
    y_max = data[y_col].max()

    
    ax.axis([x_min-x_buffer, x_max+x_buffer, y_min-y_buffer, y_max+y_buffer])
  
    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)
    if not label is None:
        ax.legend()
    ax.set_title(title)
    
    return fig, ax
