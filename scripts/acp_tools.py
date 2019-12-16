"""
Alaska Arctic Costal Plain Tools
--------------------------------
acp_tools.oy

tools for creating and managing Alaska Arctic Costal Plain 
raster datasets

ACP bounds 
    upper right: -530818.761, 2373989.086
    lower left:  565441.239, 1959449.086]  

"""
import glob
import os
from atm.images import raster
from multigrids.temporal_grid import TemporalGrid
import re
import numpy as np

ACP_EXTENT = [-530818.761, 2373989.086, 565441.239, 1959449.086]
def clip_out_acp (in_dir, out_dir, base_out_name="ACP_data"):
    """Clips the ACP out of all tif files in a directory

    Parameters
    ----------
    in_dir: path
    out_dir: path
    base_out_name: str
        how to name output files
    """
    in_files = glob.glob(os.path.join(in_dir, "*.tif"))
    # print(in_files)
    for file_path in sorted(in_files): 
        print(file_path)
        file = os.path.split(file_path)[1] 
        month, year = file[:-4].split('_')[-2:] 
        out_file = os.path.join(
            out_dir, base_out_name + '_' + year + "_" + month + ".tif"
        )  
        raster.clip_raster(file_path, out_file, ACP_EXTENT) 
 
def rasters_to_temporal_grid (in_dir, name):
    
    files = sorted(glob.glob(os.path.join(in_dir,'*.tif')))

    num_ts = len(files)
    num_years = num_ts//12
    data, metadata = raster.load_raster(files[0])
    start_year = int(files[0][:-4].split('_')[-2])
    end_year = start_year + num_years # bounds a are [start_year, end_year)

    grid = TemporalGrid(data.shape[0], data.shape[1], num_ts)

    grid_names={}                                       
    c = 0
    for year in range(start_year,end_year):
        for mon in range(1,13):
            fm = '000'+str(mon)
            tsn = str(year) +'-' + fm[-2:]
            grid_names[tsn] = c
            c += 1

    # # print grid_names
    grid.config['grid_name_map'] = grid_names

    mm_yyyy = re.compile(r'\d\d_\d\d\d\d')
    
    for path in files:
        year, month = os.path.split(path)[1][:-4].split('_')[-2:]
        # print(year, month)
        data, md = raster.load_raster(path)
        data[data==-9999] = np.nan
        grid[ year + "-" + month] = data
        

    grid.config['raster_metadata'] = metadata
    grid.config['dataset_name'] = \
        "Monthly-" + name +"-" + str(start_year) + "-" + str(end_year-1)
    grid.config['description'] = \
        "Monthly " + name +" from" + str(start_year) + " to " +\
        str(end_year-1) + ". Grid names are in the format 'YYYY-MM'."
    return grid
