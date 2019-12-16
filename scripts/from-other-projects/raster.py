"""
raster
------
Input Output operations for rasters
"""
import gdal
from collections import namedtuple
import numpy as np

ROW, COL = 0,1
import math

import matplotlib.pyplot as plt

# raster metadata named tuple
#
# See Also
# --------
#   find_canon_name
RASTER_METADATA = namedtuple('RASTER_METADATA', 
    ['transform', 'projection', 
        'nX', 'nY', 'deltaX', 'deltaY', 'originX', 'originY'
    ]
)

def load_raster (filename):
    """Load a raster file and it's metadata
    
    Parameters
    ----------
    filename: str
        path to raster file to read
        
    Returns 
    -------
    np.array
        2d raster data
    RASTER_METADATA
        metadata on raster file read
    """
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    (X, deltaX, rotation, Y, rotation, deltaY) = dataset.GetGeoTransform()

    metadata = RASTER_METADATA(
        transform = (X, deltaX, rotation, Y, rotation, deltaY),
        projection = dataset.GetProjection(),  
        nX = dataset.RasterXSize,
        nY = dataset.RasterYSize,
        deltaX = deltaX,
        deltaY = deltaY,
        originX = X,
        originY = Y
    )
    ## assumes one band, also gdal uses one based indexing here 
    data = dataset.GetRasterBand(1).ReadAsArray()
    return data, metadata

def save_raster(filename, data, transform, projection, 
    datatype = gdal.GDT_Float32):
    """Function Docs 
    Parameters
    ----------
    filename: path
        path to file to save
    data: np.array like
        2D array to save
    transform: tuple
        (origin X, X resolution, 0, origin Y, 0, Y resolution) 
    projection: string
        SRS projection in WTK format
    datatype:
        Gdal data type
    
    """
    write_driver = gdal.GetDriverByName('GTiff') 
    raster = write_driver.Create(
        filename, data.shape[1], data.shape[0], 1, datatype
    )
    raster.SetGeoTransform(transform)  
    outband = raster.GetRasterBand(1)  
    outband.WriteArray(data) 
    raster.SetProjection(projection) 
    outband.FlushCache()  
    raster.FlushCache()      

   
def scale_layer_down (layer, current_resolution, target_resolution):
    """Increases the size of grid elements by combining adjacent elements
    
    Parameters
    ----------
    layer: np.array
        layer to resize
    current_resolution: tuple
        (row, col) resolution in m of current layer
    target_resolution: tuple
        (row, col) resolution in m of desired layer
    
    Returns
    -------
    np.array
        a re-scaled layer
    """
    if target_resolution == current_resolution:
        layer[layer<=0] = 0
        layer[layer>0] = 1
        return layer.flatten()
    
    resize_num = (
        abs(int(target_resolution[ROW]/current_resolution[ROW])),
        abs(int(target_resolution[COL]/current_resolution[COL]))
    )
    resized_layer = []
    
    shape = layer.shape
    dimensions = (
        int(math.ceil(abs(shape[ROW] * current_resolution[ROW] /target_resolution[ROW]))),
        int(math.ceil(abs(shape[COL] * current_resolution[COL] /target_resolution[COL])))
    )
    ## regroup at new resolution
    for row in range(0, int(shape[ROW]), resize_num[ROW]):
        for col in range(0, int(shape[COL]), resize_num[COL]):
            A = layer[row : row+resize_num [ROW], col:col + resize_num[COL]]
            b = A > 0
            resized_layer.append(len(A[b]))
    
    return np.array(resized_layer).reshape(dimensions)

def normalize_layers(layers, current_resolution, target_resolution):
    """Normalize Layers. Ensures that the fractional cohort areas in each 
    grid element sums to one. 
    Parameters
    ----------
    layer: np.array
        layer to resize
    current_resolution: tuple
        (row, col) resolution in m of current layer
    target_resolution: tuple
        (row, col) resolution in m of desired layer
    
    Returns
    -------
    np.array
        a re-scaled layer
    """
    layers = np.array(layers)
    total = layers.sum(0) #sum fractional cohorts at each grid element
    cohorts_required = \
        (float(target_resolution[ROW])/(current_resolution[ROW])) * \
        (float(target_resolution[COL])/(current_resolution[COL]))

    cohort_check = total / cohorts_required
    ## the total is zero in non study cells. Fix warning there
    adjustment = float(cohorts_required)/total

    check_mask = cohort_check > .5
    new_layers = []
    for layer in layers:
        
        layer_mask = np.logical_and(check_mask,(layer > 0))
        layer[layer_mask] = np.multiply(
            layer,adjustment, where=layer_mask)[layer_mask]
        
        layer = np.round((layer) / cohorts_required, decimals = 6)
        new_layers.append(layer)
    new_layers = np.array(new_layers)
    return new_layers

def mask_layer(layer, mask, mask_value = np.nan):
    """apply a mask to a layer 
    layer[mask == True] = mask_value
    """
    layer[mask] = mask_value
    return layer

def clip_raster (in_raster, out_raster, extent, datatpye=gdal.GDT_Float32):
    """Clip a raster to extent
    
    Parameters:
    in_raster: path
        input raster file
    out_raster: path
        output raster file
    extent: tuple
        (minX, maxY, maxX, minY)
    """

    tiff = gdal.Translate(
        out_raster, in_raster, projWin = extent, 
        format='GTiff', outputType=datatpye
    ) 
    # printtiff
    tiff.GetRasterBand(1).FlushCache()
    tiff.FlushCache()
    return True


def convert_to_figure(raster_name, figure_name, title = "", cmap = 'viridis', 
        ticks = None, tick_labels=None, vmin=None,vmax=None, save=True
    ):
    """Converts a raster file to a figure with colorbar and title

    Parameters
    ----------
    raster_name: path
        raster file
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


## examples
# --- ## numerical 
# raster.convert_to_figure( 
#     'atm_ns_potential_initialization_areas_full_winter_precipitation/1901-1950/2003_precip_gtavg.tif', 
#     'winter-2003-2004-full-winter-precip.jpg', 
#     'Winter (Oct-Mar) 2003-2004 Departure From Average Precipitation', 
#     cmap="Greens",  
# ) 

# --- ## categorical         
# raster.convert_to_figure( 
#     'atm_ns_potential_initialization_areas_full_winter_precipitation/1901-1950/2003_precip_gtavg.tif', 
#     'winter-2003-2004-full-winter-precip.jpg', 
#     'Winter (Oct-Mar) 2003-2004 Departure From Average Precipitation', 
#     cmap=plt.cm.get_cmap("Greens", 4),  
#     ticks=[0,1,2,3],  
#     vmin=0, 
#     vmax=3,  
#     tick_labels=['<= Average', '> Average', '> 1 Std. Dev.', '> 2 Std. Dev.'] 
# )                
