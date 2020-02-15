import os
import glob
import re

from pyproj import Proj, transform
import matplotlib.pyplot as plt
import numpy as np

from atm.images import raster
from atm.tools import calc_degree_days, initiation_areas
from multigrids import TemporalGrid


# Setup directories ------------------------------------------------------------
data_interface_version = 'V1'

drive_data = os.path.join('/Volumes/toshi-stati/data/',data_interface_version)
local_data = os.path.join(
    '/Users/rwspicer/Desktop/data/',data_interface_version
)

snap_temp     = 'snap/tas_mean_C_iem_cru_TS31_1901_2015_ALL/'
snap_precip   = 'snap/pr_total_mm_iem_cru_TS31_1901_2015/'

sp_precip_tag = 'precipitation/monthly/SP/v1/'
sp_temp_tag = 'temperature/monthly/SP/v1/'

sp_precip = os.path.join(local_data, sp_precip_tag)
sp_temp = os.path.join(local_data, sp_temp_tag)

# ------------------------------------------------------------------------------

# sort snap files --------------------------------------------------------------
snap_precip_files = glob.glob(os.path.join(drive_data,snap_precip) +'/*.tif'  )
snap_temp_files = glob.glob(os.path.join(drive_data,snap_temp) +'/*.tif'  )

def sort_by_year_month (unordered):
    """ sort files by year then month, instead of month year """

    ordered = []
    years  = sorted(list(set([fn.split('.')[0][-4:] for fn in unordered])))

    for yr in years:
        ordered += sorted(
            [fn for fn in unordered if fn.split('.')[0][-4:] == yr ] 
        )

    return ordered

snap_precip_files = sort_by_year_month(snap_precip_files)
snap_temp_files = sort_by_year_month(snap_temp_files)
# ------------------------------------------------------------------------------

### Set up extent --------------------------------------------------------------
extent_wgs84 = [-168.8, 66.3, -160.5, 64.4]
inProj = Proj(init='epsg:4326') #GPS
outProj = Proj(init='epsg:3338')#Alaska albers
extent_alaska_albers = list(
    transform(inProj,outProj,extent_wgs84[0],extent_wgs84[1])
) 
extent_alaska_albers += list(
    transform(inProj,outProj,extent_wgs84[2],extent_wgs84[3])
) 
extent = extent_alaska_albers
### ----------------------------------------------------------------------------

### clip rasters to extent -----------------------------------------------------
def clip_rasters_to_directory(in_files, out_dir, extent, file_tag):
    """
    """
    for fn in in_files:
        # print(fn)
        mn, yr = re.search('_\d{2}_\d{4}', fn).group(0).split('_')[1:] 
        out_file = os.path.join(
            out_dir, file_tag + '-' + yr + '-' + mn + '.tif'
        ) 
        raster.clip_raster(fn,out_file, extent)

    return True

# clip_rasters_to_directory(
#     snap_temp_files, os.path.join(sp_temp,'tiff/'), extent, 'SP-temp-c'
# )
# clip_rasters_to_directory(
#     snap_precip_files, os.path.join(sp_precip,'tiff/'), extent, 'SP-precip-mm'
# )
# ------------------------------------------------------------------------------

### Multigrids setup -----------------------------------------------------------

def set_up_mgs ():
    sp_monthly_precip_files = sorted(
        glob.glob(os.path.join(sp_precip, 'tiff', '*.tif'))
    )
    sp_monthly_temp_files = sorted(
        glob.glob(os.path.join(sp_temp, 'tiff', '*.tif'))
    )

    example_tiff = sp_monthly_temp_files[0]
    ex_raster, ex_md = raster.load_raster(example_tiff)

    rows, cols = ex_raster.shape
    n_months = len(sp_monthly_temp_files)

    mask = ex_raster > -9999

    sp_monthly_precip_mg = TemporalGrid(rows, cols, n_months, mask = mask)
    sp_monthly_precip_mg.config['raster_metadata'] = ex_md
    sp_monthly_precip_mg.config['dataset_name'] =\
        'Seward Peninsula Monthly Precipitation(mm)'
    sp_monthly_precip_mg.config['dataset-version'] = '1.0.0'
    sp_monthly_precip_mg.config['description'] = """
        Seward Peninsula Monthly Precipitation(mm) for 1901 to 2015
        Clipped from SNAP pr_total_mm_iem_cru_TS31_1901_2015 data
    """
    sp_monthly_precip_mg.config['units'] = 'mm'
    grid_name_map = {}
    for idx, fn in  enumerate(sp_monthly_precip_files):
        grid_id = fn.split('.')[0][-7:]
        grid_name_map[grid_id] = idx
        data, md = raster.load_raster(fn)
        sp_monthly_precip_mg[idx] = data

    sp_monthly_precip_mg.config['grid_name_map'] = grid_name_map




    sp_monthly_temp_mg = TemporalGrid(rows, cols, n_months, mask = mask)
    sp_monthly_temp_mg.config['raster_metadata'] = ex_md
    sp_monthly_temp_mg.config['dataset_name'] =\
        'Seward Peninsula Monthly Temperature(C)'
    sp_monthly_temp_mg.config['dataset-version'] = '1.0.0'
    sp_monthly_temp_mg.config['description'] = """
        Seward Peninsula Monthly Temperature(C) for 1901 to 2015
        Clipped from SNAP tas_mean_C_iem_cru_TS31_1901_2015
    """
    sp_monthly_temp_mg.config['units'] = 'deg C'

    ## don't need to re do gridname map its same as precip version
    for idx, fn in  enumerate(sp_monthly_temp_files):
        data, md = raster.load_raster(fn)
        sp_monthly_temp_mg[idx] = data

    sp_monthly_temp_mg.config['grid_name_map'] = grid_name_map

    sp_monthly_temp_mg.show_figure('1920-01',figure_args={'mask':sp_monthly_temp_mg.config['mask']})
    plt.show()


    sp_monthly_temp_mg.save(
        os.path.join(sp_temp, 'multigrid', 'SP-monthly-temp-c.yml')
    )
    sp_monthly_precip_mg.save(
        os.path.join(sp_precip, 'multigrid', 'SP-monthly-precip-mm.yml')
    )

    return sp_monthly_precip_mg,sp_monthly_temp_mg
# sp_monthly_precip_mg, sp_monthly_temp_mg = set_up_mgs()

sp_monthly_precip_mg = TemporalGrid(
     os.path.join(sp_precip, 'multigrid', 'SP-monthly-precip-mm.yml')
)
sp_monthly_temp_mg = TemporalGrid(
    os.path.join(sp_temp, 'multigrid', 'SP-monthly-temp-c.yml')
)

# ------------------------------------------------------------------------------

## Setup summed prcip data------------------------------------------------------

# def get_monthly_keys(months):
#     keys = []
#     for yr in range(1901,2016):
#         for mn in months:  
#             keys.append(str(yr)+'-'+str(mn))
#     return keys

# ewp_keys = get_monthly_keys([10,11])

# # print (ewp_keys)

# ewp_grids = sp_monthly_precip_mg.get_grids_at_keys(ewp_keys)

# print(type(ewp_grids),ewp_grids)

time_periods = [
    ([10,11], 'Early Winter', 'ewp'),
    ([10,11,12,13,14,15], 'Full Winter', 'fwp'),
    ([4,5,6,7,8,9], 'Summer', 'sp'),
    ([8,9], 'Late Summer', 'lsp'),
]
for tp in time_periods:
    sp_sum_mg = initiation_areas.create_precip_sum_multigrid(
        sp_monthly_precip_mg, 
        tp[0],
        1901, 2015,
        title = 'Seward Peninsula ' + tp[1] +' Precipitation(mm)',
        units = 'mm',
        description= """
            Seward Peninsula __TAG__ Precipitation(mm) for 1901 to 2015
            Summed from data in: V1\precipitation\monthly\SP\multigrid
        """.replace('__TAG__',tp[1]),
        raster_metadata=sp_monthly_precip_mg.config['raster_metadata'],
        other = {
            "dataset-version": "1.0.0",
            'mask':sp_monthly_precip_mg['1901-01'] > -9999
            
        }
    )

    sp_sum_mg.save(
        os.path.join(
            local_data,'precipitation',tp[1].lower().replace(' ','-'),
            'SP','v1','multigrid',
            'SP-'+tp[2].upper()+'-precip-mm.yml'
        )
    )
    # sp_sum_mg.show_figure(1950, figure_args={'mask':sp_sum_mg.config['mask']})
    # plt.show()
# ------------------------------------------------------------------------------

## Calc Degree days ------------------------------------------------------------
#use the command line utility
##
sp_monthly_temp_files = sorted(
        glob.glob(os.path.join(sp_temp, 'tiff', '*.tif'))
    )
# print(sp_monthly_temp_files)
with open('sp_temp_file_list.txt', 'w') as fd:
    fd.write('\n'.join(sp_monthly_temp_files))

tdd_dir = os.path.join(
    local_data,'degree-day', 'thawing', 'SP', 'v1', 'multigrid'
)

fdd_dir = os.path.join(
    local_data,'degree-day', 'freezing', 'SP', 'v1', 'multigrid'
)

tdd_mm = 'tdd.memmap'
fdd_mm = 'fdd.memmap'


## get shape
sp_monthly_temp_files = sorted(
    glob.glob(os.path.join(sp_temp, 'tiff', '*.tif'))
)
example_tiff = sp_monthly_temp_files[0]
ex_raster, ex_md = raster.load_raster(example_tiff)

rows, cols = ex_raster.shape
n_years = len(sp_monthly_temp_files) // 12## get years
mask = ex_raster > -9999


degree_day_info = [
    ('fdd', 'Freezing Degree-day', fdd_mm, fdd_dir ),
    ('tdd', 'Thawing Degree-day',  tdd_mm, tdd_dir  )
]

for items in degree_day_info:
    dd_mg = TemporalGrid(rows, cols, n_years, mask = mask)
    dd_mg.config['raster_metadata'] = ex_md
    dd_mg.config['dataset_name'] =\
        'Seward Peninsula ' + items[1]
    dd_mg.config['dataset-version'] = '1.0.0'
    dd_mg.config['description'] = """
        Seward Peninsula __TAG__ for 1901 to 2015
        Calculated via spline method using SP monthly temperatue data
    """.replace('__TAG__', items[1])
    dd_mg.config['units'] = 'Degree-days'
    dd_mg.config['start_timestep'] = 1901
    dd_mm = np.memmap(
        items[2], dtype='float32', mode='r', shape=(n_years, rows, cols)
    )

    ## don't need to re do gridname map its same as precip version
    for idx, yr in  enumerate(range(1901,1901+n_years)):
        dd_mg[yr][:] = dd_mm[idx][:]

    

    dd_mg.save(
        os.path.join(items[3], 'SP-'+items[0].upper()+'.yml'
        )
    )
    dd_mg.show_figure(1950)
    dd_mg.show_figure(2015)


# ------------------------------------------------------------------------------


## clip elev,slop,aspect files -------------------------------------------------

info = [
    ('slope', 'AK-SLOPE1000m.tif', 'SP-SLOPE1000m.tif'),
    ('elevation', 'AK-DEM-1000m.tif', 'SP-DEM1000m.tif'),
    ('aspect', 'AK-ASPECT1000m.tif', 'SP-ASPECT1000m.tif')
]

for item in info:
    save_path = os.path.join(local_data,'geolocation',item[0],'SP','v1',item[2])
    load_path = os.path.join(
        local_data,'geolocation',item[0],'alaska','v1',item[1]
    )
    raster.clip_raster(load_path, save_path, extent)



# ------------------------------------------------------------------------------


## make lat long files ---------------------------------------------------------

## DONE VIA the classy_gdal Stuff


# ------------------------------------------------------------------------------


## Group as Temporal Multigrid -------------------------------------------------


# ------------------------------------------------------------------------------
