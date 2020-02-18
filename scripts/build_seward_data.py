import os
import glob
import re

from pyproj import Proj, transform
import matplotlib.pyplot as plt
import numpy as np

from atm.images import raster
from atm.tools import calc_degree_days, initiation_areas
from multigrids import TemporalGrid, TemporalMultiGrid



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

    # sp_monthly_temp_mg.show_figure('1920-01',figure_args={'mask':sp_monthly_temp_mg.config['mask']})
    # plt.show()


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

    ## correct off by one years in fdd data
    if items[0] == 'fdd':
        dd_mg.grids[:-1] = dd_mg.grids[1:]

    # print (
    #     items[0], 
    #     np.allclose(dd_mg[1901], dd_mg[1902], equal_nan=True), 
    #     np.allclose(dd_mg[2014], dd_mg[2015], equal_nan=True) 
    # )
    dd_mg.save(
        os.path.join(items[3], 'SP-'+items[0].upper()+'.yml'
        )
    )
    # dd_mg.show_figure(1950)
    # dd_mg.show_figure(2015)


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



#   aspect: 10
#   elev: 12
#   ewp: 3
#   fdd: 0
#   fwp: 4
#   lat: 8
#   long: 9
#   lsp: 6
#   slope: 11
#   sp: 5
#   sp+1: 7
#   tdd: 1
#   tdd+1: 2

rows, cols = ex_raster.shape
n_years = len(sp_monthly_temp_files) // 12 ## get years
list_grids = [
    'fdd', 'tdd', 'tdd+1', 
    'ewp', 'fwp', 'sp', 'lsp', 'sp+1',
    'lat', 'long', 'aspect', 'slope', 'elev',
]
# mask = ex_raster > -9999


SP_training_mg = TemporalMultiGrid(
    rows, cols, len(list_grids), n_years,
    dataset_name = "SP Training Data Base",
    grid_names = list_grids,
)

SP_training_mg.config['start_timestep'] = 1901 
SP_training_mg.config['description'] = """
    Seward Peninsula Training Data with base features used in intital tests
"""

tdd_dir = os.path.join(
    local_data,'degree-day', 'thawing', 'SP', 'v1', 'multigrid'
)

fdd_dir = os.path.join(
    local_data,'degree-day', 'freezing', 'SP', 'v1', 'multigrid'
)

time_periods = [
    ([10,11], 'Early Winter', 'ewp'),
    ([10,11,12,13,14,15], 'Full Winter', 'fwp'),
    ([4,5,6,7,8,9], 'Summer', 'sp'),
    ([8,9], 'Late Summer', 'lsp'),
]

base = os.path.join(local_data,'precipitation')
set_descriptor = os.path.join('SP','v1','multigrid')
precip_paths = {
    tp[2]: os.path.join( 
        base, 
        tp[1].lower().replace(' ','-'), 
        set_descriptor,
        'SP-'+tp[2].upper()+'-precip-mm.yml'
    ) for tp in time_periods
}


info = [
    ('fdd', 'fdd',   fdd_dir, 0 ),
    ('tdd', 'tdd',    tdd_dir, 0 ), 
    ('tdd+1','tdd', tdd_dir, 1 ),
    ('sp','sp', precip_paths['sp'], 0),
    ('lsp', 'lsp', precip_paths['lsp'],0),
    ('ewp','ewp', precip_paths['ewp'],0),
    ('fwp','fwp', precip_paths['fwp'],0),
    ('sp+1','sp', precip_paths['sp'],1),
]
for item in info:
    # print (item)
    try:
        item_data = TemporalGrid(
            os.path.join(item[2], 'SP-'+item[1].upper()+'.yml')
        )
    except:
        item_data = TemporalGrid(item[2])

    if item[3] == 0:
        try:
            SP_training_mg[item[0]][:] = item_data.grids[:].reshape(
                (n_years,rows, cols)
            )
        except ValueError:
            SP_training_mg[item[0]][:-1] = item_data.grids[:].reshape(
                (n_years-1,rows, cols)
            )
    else: #item[3] == 1
        SP_training_mg[item[0]][:-1] = item_data.grids[1:].reshape(
            (n_years - 1,rows, cols)
        )


info = [
    ('slope', 'slope', 'SP-SLOPE1000m.tif'),
    ('elev','elevation',  'SP-DEM1000m.tif'),
    ('aspect','aspect', 'SP-ASPECT1000m.tif'),
    ('lat','geolocation', 'SP-geolocation_lat.tif'),
    ('long','geolocation', 'SP-geolocation_long.tif')
]

for item in info:
    load_path = os.path.join(local_data,'geolocation',item[1],'SP','v1',item[2])
    data, md = raster.load_raster(load_path)

    for year in range(1901, 1901+n_years):

        SP_training_mg[item[0], year][:] = data[:]

    

SP_training_mg.save(os.path.join(
    local_data, 
    'master-project', 'training', 'SP', 'v1', 'baseline', 'multigrid',
    'SP-rf-training-set.yml'
))

SPTD = SP_training_mg
print ('verification report')
print ('fdd: 2014 == 2015', np.allclose(SPTD['fdd', 2014], SPTD['fdd', 2015],
    equal_nan=True))
print ('fdd: 1901 != 1902', not np.allclose(SPTD['fdd', 1901], 
    SPTD['fdd', 1902], equal_nan=True))
print ('tdd[n+1] == tdd+1[n]', np.allclose(SPTD['tdd', 1951], 
    SPTD['tdd+1', 1950], equal_nan=True))
print ('sp[n+1] == sp+1[n]', np.allclose(SPTD['sp', 1951], SPTD['sp+1', 1950],
     equal_nan=True))
# ------------------------------------------------------------------------------


## SP om run -------------------------------------------------------------------

fdd = TemporalGrid(os.path.join(local_data, 'degree-day/freezing/SP/v1/multigrid/SP-FDD.yml'))
fdd.config['ts_offset'] = 0

tdd = TemporalGrid(os.path.join(local_data, 'degree-day/thawing/SP/v1/multigrid/SP-TDD.yml'))
tdd.config['ts_offset'] = 0


ewp = TemporalGrid(
    os.path.join(local_data, 'precipitation/early-winter/SP/v1/multigrid/SP-EWP-precip-mm.yml')
)
ewp.config['ts_offset'] = 0

fwp = TemporalGrid(
    os.path.join(local_data, 'precipitation/full-winter/SP/v1/multigrid/SP-FWP-precip-mm.yml')
)
fwp.config['ts_offset'] = 0


tdd_p1 = tdd.clone()
tdd_p1.grids[:-1] = tdd.grids[1:] # tdd_p1[0] <- tdd[1], tdd_p1[1] <- tdd[2], and so on
tdd_p1.config['ts_offset'] = 1


grid_dict = {
    "tdd": tdd,
    "ewp": ewp,
    "fwp": fwp,
    "fdd": fdd,
    "tdd+1": tdd_p1
}

bounds = [1901,1950]

ia_grid, stats_grid = initiation_areas.find_initiation_areas_vpdm(grid_dict, bounds)

ia_grid.config['dataset_version'] = '1.0.0'
ia_grid.config['dataset_name'] = 'SP 5 var Thermokarst Initiation areas'
ia_grid.config['description'] = """Seward Peninsula 5 var TKI areas version 4. 
    Calculated from tdd(v1), ewp(v1), fwp(v1), fdd(v1), tdd+1 (tdd v1 offset by 1 year).
"""

# need to clip out seward first
# tkpd, md = raster.load_raster(
#     os.path.join(local_data,'thermokarst/predisposition-model/SP/v1/SP-tk-predisp-model.tif')
# )
# tkpd[ tkpd < -5] = np.nan


# apply_tkpd = lambda tki: tki * ( tkpd.flatten() / 100)
# ia_grid_with_predisp = ia_grid.apply_function(apply_tkpd)

# ia_grid_with_predisp.config['dataset_version'] = '1.0.0'
# ia_grid_with_predisp.config['dataset_name'] = 'SP 5 var TKI  areas'
# ia_grid_with_predisp.config['description'] = """Seward Peninsula 5 var TKI areas version 4. 
#     Calculated from tdd(v1), ewp(v1), fwp(v1), fdd(v1), tdd+1 (tdd v1 offset by 1 year).
#     With predisposition model applied
# """

ia_grid.config['raster_metadata'] = tdd.config['raster_metadata']
# ia_grid_with_predisp.config['raster_metadata']= tdd.config['raster_metadata']
stats_grid.config['raster_metadata']= tdd.config['raster_metadata']

stats_grid.config['dataset-name'] = "SP-IA-Stats"

ia_grid.save(
    '/Users/rwspicer/Desktop/data/V1/thermokarst/initiation-regions/SP/v1/PDM-5var/without_predisp/multigrid/SP-TKI-PDM5.yml'
)
# ia_grid_with_predisp.save('/Users/rwspicer/Desktop/data/V1/thermokarst/initiation-regions/SP/v1/PDM-5var/with_predisp/multigrid/SP-TKI-PDM5-with-predisp.yml')

stats_grid.save('/Users/rwspicer/Desktop/data/V1/thermokarst/initiation-regions/SP/v1/PDM-stats/multigrid/SP-TKI-stats.yml')

