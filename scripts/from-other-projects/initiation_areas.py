"""
initiation Areas tools
----------------------

initiation_areas.py

Tools to help find areas were thermokarst initiation is likely 

"""
from atm.images import raster
from multigrids import temporal_grid, multigrid
import glob
import numpy as np
import os 
import re
         
def show(data, title='fig'):  
    """Unility Function to show a figure
    """
    import matplotlib.pyplot as plt

    plt.imshow(data, cmap='viridis', vmin=0,vmax=4)
    plt.title(title, wrap = True)
    cb = plt.colorbar(ticks =range( 4), orientation = 'vertical')
    cb.set_ticklabels(['below', 'above', '>1 std', '>2 std'])
    #plt.imshow(data, cmap='seismic')
    #plt.colorbar()
    plt.show()


def load_precip_data(precip_dir, start_year, end_year):
    """Loads monthly precipitation data from raster files in a given directory,
    for a given range of years(inclusive range).

    Parameters
    ----------
    precip_dir: path
        directory contains the raster files in the format "**MM_YYYY.tif"
        The "SNAP naming convention"
    start_year: int
        start year of rasters to load
    end_year: int
        end year of rasters to load

    Returns
    -------
    multigrids.temporal_gird.TemporalGrid
        The monthly precipitation data for the range of years. Grid names are 
        in the format "YYYY-MM."
    """
    files = glob.glob(precip_dir+'/*.tif')

    num_ts = (end_year+1-start_year)*12
    data, metadata = raster.load_raster(files[0])

    precip = temporal_grid.TemporalGrid(data.shape[0], data.shape[1], num_ts)

    grid_names={}                                       
    c = 0
    for year in range(start_year,end_year+1):
        for mon in range(1,13):
            fm = '000'+str(mon)
            tsn = str(year) +'-' + fm[-2:]
            grid_names[tsn] = c
            c += 1

    # # print grid_names
    precip.config['grid_name_map'] = grid_names

    mm_yyyy = re.compile(r'\d\d_\d\d\d\d')
    
    for file_with_path in sorted(files):
        file = os.path.split(file_with_path)[1]
        mon, year = mm_yyyy.findall(file)[0].split('_')
        if not (start_year <= int(year) <= end_year):
            continue


        data, metadata = raster.load_raster(file_with_path)
        data[data==-9999] = np.nan
        idx = year +'-'+ mon
        
        precip[idx] = data

    precip.config['grid_name_map'] = grid_names  
    precip.config['raster_metadata'] = metadata
    precip.config['dataset_name'] = \
        "Monthly-Precipitation-" + str(start_year) + "-" + str(end_year)
    precip.config['description'] = \
        "Monthly Precipitation from" + str(start_year) + " to " +\
        str(end_year) + ". Grid names are in the format 'YYYY-MM'."

    return precip


def calc_early_winter_precip_avg (precip, years='all'):
    """Calculate the average and standard diveation for early winter 
    precipitation. Early winter consists of October and November

    Parameters
    ----------
    precip: multigrids.temporal_gird.TemporalGrid
    years: str
        Range of years to calculate average over. 
        'all' or 'start-end' ie '1901-1950'.
    
    Returns
    ------- 
    winter_avg: np.array [M x N]
        map early winter precipitation averages 
    winter_std: np.array [M x N]
        map early winter precipitation standard diveations 
    winter_precip: np.array [M x N x Num_Years]
        maps of early winter precipitation for each year
    """
    # precip.grids = np.array(precip.grids)
    keys = precip.config['grid_name_map'].keys()
    oct_keys = sorted([k for k in keys if k[-2:]=='10'])
    nov_keys = sorted([k for k in keys if k[-2:]=='11'])
    # # print nov_keys
    if years != 'all':
        start, end = years.split('-')
        oct_filtered = []
        nov_filtered = []
        for year in range(int(start),int(end)+1):
            oct_filtered += [ok for ok in oct_keys if ok[:4] == str(year)]
            nov_filtered += [nk for nk in nov_keys if nk[:4] == str(year)]

        oct_keys = sorted(oct_filtered)
        nov_keys = sorted(nov_filtered)
            
    # # print oct_keys
    # # print nov_keys

    oct_precip = precip.get_grids_at_keys(oct_keys)
    nov_precip = precip.get_grids_at_keys(nov_keys)

    winter_precip = oct_precip + nov_precip
    # print winter_precip.shape
    winter_avg =  winter_precip.mean(0)
    winter_std =  winter_precip.std(0)
    return winter_avg, winter_std, winter_precip


def calc_winter_precip_avg (precip, years='all'):
    """Calculate the average and standard diveation for winter 
    precipitation. Winter consists of October - March

    Parameters
    ----------
    precip: multigrids.temporal_gird.TemporalGrid
    years: str
        Range of years to calculate average over. 
        'all' or 'start-end' ie '1901-1950'.
    
    Returns
    ------- 
    winter_avg: np.array [M x N]
        map winter precipitation averages 
    winter_std: np.array [M x N]
        map winter precipitation standard diveations 
    winter_precip: np.array [M x N x Num_Years]
        maps of winter precipitation for each year
    """
    # precip.grids = np.array(precip.grids)
    keys = precip.config['grid_name_map'].keys()
    oct_keys = sorted([k for k in keys if k[-2:]=='10'])
    nov_keys = sorted([k for k in keys if k[-2:]=='11'])
    dec_keys = sorted([k for k in keys if k[-2:]=='12'])
    jan_keys = sorted([k for k in keys if k[-2:]=='01'])
    feb_keys = sorted([k for k in keys if k[-2:]=='02'])
    mar_keys = sorted([k for k in keys if k[-2:]=='03'])

    if years != 'all':
        start, end = years.split('-')
        oct_filtered = []
        nov_filtered = []
        dec_filtered = []
        jan_filtered = []
        feb_filtered = []
        mar_filtered = []
        for year in range(int(start),int(end)):
            oct_filtered += [ok for ok in oct_keys if ok[:4] == str(year)]
            nov_filtered += [nk for nk in nov_keys if nk[:4] == str(year)]
            dec_filtered += [dk for dk in dec_keys if dk[:4] == str(year)]
            jan_filtered += [jk for jk in jan_keys if jk[:4] == str(year + 1)]
            feb_filtered += [fk for fk in feb_keys if fk[:4] == str(year + 1)]
            mar_filtered += [mk for mk in mar_keys if mk[:4] == str(year + 1)]

        oct_keys = sorted(oct_filtered)
        nov_keys = sorted(nov_filtered)
        dec_keys = sorted(dec_filtered)
        jan_keys = sorted(jan_filtered)
        feb_keys = sorted(feb_filtered)
        mar_keys = sorted(mar_filtered)
            

    oct_precip = precip.get_grids_at_keys(oct_keys)
    nov_precip = precip.get_grids_at_keys(nov_keys)
    dec_precip = precip.get_grids_at_keys(dec_keys)
    jan_precip = precip.get_grids_at_keys(jan_keys)
    feb_precip = precip.get_grids_at_keys(feb_keys)
    mar_precip = precip.get_grids_at_keys(mar_keys)


    winter_precip = oct_precip + nov_precip + dec_precip +\
                    jan_precip + feb_precip + mar_precip
    winter_avg =  winter_precip.mean(0)
    winter_std =  winter_precip.std(0)
    return winter_avg, winter_std, winter_precip

def sum_precip(monthly, months, years='all'):
    """Calculate the average and standard diveation for winter 
    precipitation. Winter consists of October - March

    Parameters
    ----------
    monthly: multigrids.temporal_gird.TemporalGrid
        monthly precip data
    years: str
        Range of years to calculate average over. 
        'all' or 'start-end' ie '1901-1950'.
    
    Returns
    ------- 
    winter_precip: np.array [M x N x Num_Years]
        maps of winter precipitation for each year
    """
    # precip.grids = np.array(precip.grids)
    keys = monthly.config['grid_name_map'].keys()

    find = lambda x: sorted( [k for k in keys if k[-2:]=='{:0>2}'.format(x)])
    monthly_keys = {
        x: find(x) for x in range(1, 13)
        }     

    pad = 1 - max([m//12 for m in months])

    if years != 'all':
        try:
            start, end = years.split('-')
        except AttributeError:
            start, end = years[0], years[1]
        start, end = int(start), int(end)
    else:
        start = min([int(x.split('-')[0])for x in keys]) 
        end = max([int(x.split('-')[0])for x in keys]) 
    

    month_filter = lambda m, y, ks: [k for k in ks[m] if k[:4] == str(y)]
    months_filtered = {}
    for year in range(start, end+pad):
        for mon in months:
            if mon < 1:
                raise IndexError ("Months cannot be less then 1")
            adj_mon = mon
            adj_year = year
            while adj_mon > 12:
                adj_mon -= 12
                adj_year += 1

            try:
                months_filtered[mon] +=  month_filter(
                    adj_mon, adj_year, monthly_keys
                ) 
            except KeyError:
                months_filtered[mon] = month_filter(
                    adj_mon, adj_year, monthly_keys
                )
    
    precip_sum = None
    for mon in months_filtered:
        try:
            precip_sum += monthly.get_grids_at_keys(months_filtered[mon])
        except TypeError:
            precip_sum = monthly.get_grids_at_keys(months_filtered[mon])

    return precip_sum

def create_precip_sum_multigrid(monthly, months, start, end, 
    title='summed precip', units='mm', description=None, raster_metadata=None,
    other={}):
    """
    """
    precip = sum_precip(
        monthly, months, str(start) + '-' + str(end)
    )
    precip_grid = temporal_grid.TemporalGrid(
        precip.shape[1], precip.shape[2], precip.shape[0]
    )
    precip_grid.grids[:] = precip.reshape(
        precip.shape[0], precip.shape[1] * precip.shape[2]
    ) 

    precip_grid.config['units'] = units
    precip_grid.config['dataset_name'] = title
    precip_grid.config['description'] = description
    if description is None:
        precip_grid.config['description'] = \
            title + '. For months =' + str(months) 
    
    precip_grid.config['mean'] = precip.mean(0)
    precip_grid.config['std. dev.'] = precip.std(0)
    precip_grid.config['start_timestep'] = start 
    precip_grid.config['raster_metadata'] = raster_metadata
    precip_grid.config.update(other)

    return precip_grid



def find_initiation_areas (precip, tdd, fdd, directory, years = 'all', 
        winter_precip_type = 'full'
    ):
    """Creates raster outputs of the Initialization Areas.

    These initiation areas rasters are calculated based on a given winter
    and the following summer. Basically if the winter is warmer than average
    with high precipitation (early or full) is higher than average, followed 
    by a warmer than average summer then the likelihood for initiation is
    higher.

    Parameters
    ----------
    precip: multigrids.temporal_gird.TemporalGrid 
        Full monthly precipitation dataset
    tdd: multigrids.temporal_gird.TemporalGrid
        Thawing Degree Days dataset 
    fdd: multigrids.temporal_gird.TemporalGrid
        Freezing Degree Days dataset 
    directory: path
        directory to save raster files(.tif) to.
    years: str
        Range of years to calculate averages over. 
        'all' or 'start-end' ie '1901-1950'.
    winter_precip_type: str
        'early', 'full' or 'both'

        

    Returns
    ------
    winter_precip_avg, winter_precip_std, 
    tdd_avg, tdd_std, fdd_avg, fdd_std: np.array [M x N]
        maps of the average and standard diveation. 
    """

    # Fist line gets the average and standard diveation for the desired range
    # of years. The second and third lines gets the full winter 
    # precipitation data.
    full_winter_precip_avg, full_winter_precip_std, full_winter_precip = \
        calc_winter_precip_avg(precip, years) 
    full_winter_precip  = calc_winter_precip_avg(precip, 'all') 
    full_winter_precip = full_winter_precip[2] 

    early_winter_precip_avg, early_winter_precip_std, early_winter_precip = \
        calc_early_winter_precip_avg(precip, years) 
    early_winter_precip  = calc_early_winter_precip_avg(precip, 'all') 
    early_winter_precip = early_winter_precip[2] 
    

    ## get the degree day averages and standard dilations. 
    if years == 'all':
        start = 1901
        fdd_avg = fdd.grids.reshape(fdd.config['real_shape'])
        tdd_avg = tdd.grids.reshape(tdd.config['real_shape'])
    else:
        start, end = years.split('-')
        fdd_avg = fdd.get_grids_at_keys(range(int(start),int(end)+1))
        tdd_avg = tdd.get_grids_at_keys(range(int(start),int(end)+1))
    fdd_std = fdd_avg.std(0)
    tdd_std = tdd_avg.std(0)
    fdd_avg = fdd_avg.mean(0)
    tdd_avg = tdd_avg.mean(0)

    # raster metadata 
    transform = precip.config['raster_metadata'].transform
    projection = precip.config['raster_metadata'].projection

    os.makedirs(os.path.join(directory, years))
    raster.save_raster(
        os.path.join(directory, years + '_precip_full_winter_avg.tif'), 
        full_winter_precip_avg, transform, projection)
    raster.save_raster(
        os.path.join(directory, years + '_precip_early_winter_avg.tif'), 
        early_winter_precip_avg, transform, projection)
    raster.save_raster(
        os.path.join(directory, years + '_fdd_avg.tif'),
        fdd_avg, transform, projection
    )
    raster.save_raster(
        os.path.join(directory, years + '_tdd_avg.tif'), 
        tdd_avg, transform, projection
    )

    ## find the areas
    shape = tdd.config['grid_shape']
    fdd_grid = np.zeros(shape)  
    fdd_grid[::] = np.nan
    # show(fdd_grid)

    tdd_grid = np.zeros(shape)  
    tdd_grid[::] = np.nan

    full_precip_grid = np.zeros(shape)  
    full_precip_grid[::] = np.nan

    early_precip_grid = np.zeros(shape)  
    early_precip_grid[::] = np.nan

    warm_winter = np.zeros(shape)  
    warm_winter[::] = np.nan

    

    _max = -10
    _min = 20
    
    for idx in range(fdd.grids.shape[0]):

        # current year values
        c_fdd = fdd.grids[idx].reshape(shape)
        c_tdd = tdd.grids[idx].reshape(shape)
        c_full_precip = full_winter_precip[idx].reshape(shape)
        c_early_precip = early_winter_precip[idx].reshape(shape)

        # grids for mapping deviation 
        fdd_grid[::] = c_fdd - c_fdd
        tdd_grid[::] = c_tdd - c_tdd
        full_precip_grid[::] = c_full_precip  - c_full_precip 
        early_precip_grid[::] = c_early_precip - c_early_precip 

        # for  deviation grids:
        #   0 means <= average
        #   1 means > average
        #   2 means > 1 std. deviation. 
        #   2 means > 2 std. deviations. 
        for s in [0, 1, 2]:
            fdd_grid[c_fdd > (fdd_avg + s * fdd_std)] = s + 1  
            tdd_grid[c_tdd > (tdd_avg + s * tdd_std)] = s + 1 
            full_precip_grid[
                c_full_precip  > (full_winter_precip_avg + s * full_winter_precip_std)
            ] = s + 1 
            early_precip_grid[
                c_early_precip  > (early_winter_precip_avg + s * early_winter_precip_std)
            ] = s + 1 


        # calculate the initiation map
        # 'warm winters' (high precip + high temps) + hot summers = higher initiation 
        initiation = warm_winter + (tdd_grid)
        _min = min(_min, np.nanmin(initiation))
        _max = max(_max, np.nanmax(initiation))
        

        # save rastes
        yr = str(int(start)+idx)
        raster.save_raster(os.path.join(directory, years, yr + '_initialization_areas.tif'), initiation, transform, projection)
        raster.save_raster(os.path.join(directory, years, yr + '_full_precip_gtavg.tif'), full_precip_grid, transform, projection)
        raster.save_raster(os.path.join(directory, years, yr + '_early_precip_gtavg.tif'), early_precip_grid, transform, projection)
        raster.save_raster(os.path.join(directory, years, yr + '_fdd_gtavg.tif'), fdd_grid, transform, projection)
        raster.save_raster(os.path.join(directory, years, yr + '_tdd_gtavg.tif'), tdd_grid, transform, projection)

        # high precip + high temps 
        if winter_precip_type == 'full':
            print('using full winter')
            warm_winter = (fdd_grid) + full_precip_grid
        elif winter_precip_type == 'early':
            print('using early winter')
            warm_winter = (fdd_grid) + early_precip_grid
        elif winter_precip_type == 'both':
            print('using full and early winter')
            warm_winter = (fdd_grid) + full_precip_grid + early_precip_grid
        else:
            m = "Argument winter_precip_type must be 'early', 'full', or 'both'"
            raise TypeError(m)

    print(_min,_max)
    return full_winter_precip_avg, full_winter_precip_std, early_winter_precip_avg, early_winter_precip_std, tdd_avg, tdd_std, fdd_avg, fdd_std

   



# precip = temporal_grid.TemporalGrid('/Users/rwspicer/Desktop/ns_precip_monthly/cliped-precip-loaded-data-1901-2015.yml')
# fdd = temporal_grid.TemporalGrid('/Users/rwspicer/Desktop/ns_fdd/fdd.yml')
# tdd = temporal_grid.TemporalGrid('/Users/rwspicer/Desktop/ns_tdd/tdd.yml')


## 0 = precip <= avg, fdd <= avg, fdd <= avg
## 1 = precip > avg, fdd <= avg, fdd <= avg
## 2 = precip > 1 std, fdd <= avg, fdd <= avg
## 3 = precip > 2 std, fdd <= avg, fdd <= avg
## 4 = precip <= avg std, fdd > avg, fdd <= avg
## 5 = precip <= avg std, fdd > avg, fdd <= avg
## 6 = precip <= avg std, fdd > avg, fdd <= avg
## 7 = precip <= avg std, fdd > avg, fdd <= avg



def find_initiation_areas_2 (precip, tdd, fdd, directory, years = 'all', 
        winter_precip_type = 'full'
    ):
    """Creates raster outputs of the Initialization Areas.

    These initiation areas rasters are calculated based on a given winter
    and the following summer. Basically if the winter is warmer than average
    with high precipitation (early or full) is higher than average, followed 
    by a warmer than average summer then the likelihood for initiation is
    higher.

    Parameters
    ----------
    precip: multigrids.temporal_gird.TemporalGrid 
        Full monthly precipitation dataset
    tdd: multigrids.temporal_gird.TemporalGrid
        Thawing Degree Days dataset 
    fdd: multigrids.temporal_gird.TemporalGrid
        Freezing Degree Days dataset 
    directory: path
        directory to save raster files(.tif) to.
    years: str
        Range of years to calculate averages over. 
        'all' or 'start-end' ie '1901-1950'.
    winter_precip_type: str
        'early', 'full' or 'both'

        

    Returns
    ------
    winter_precip_avg, winter_precip_std, 
    tdd_avg, tdd_std, fdd_avg, fdd_std: np.array [M x N]
        maps of the average and standard diveation. 
    """

    # Fist line gets the average and standard diveation for the desired range
    # of years. The second and third lines gets the full winter 
    # precipitation data.
    full_winter_precip_avg, full_winter_precip_std, full_winter_precip = \
        calc_winter_precip_avg(precip, years) 
    full_winter_precip  = calc_winter_precip_avg(precip, 'all') 
    full_winter_precip = full_winter_precip[2] 

    early_winter_precip_avg, early_winter_precip_std, early_winter_precip = \
        calc_early_winter_precip_avg(precip, years) 
    early_winter_precip  = calc_early_winter_precip_avg(precip, 'all') 
    early_winter_precip = early_winter_precip[2] 
    

    ## get the degree day averages and standard dilations. 
    if years == 'all':
        start = 1901
        fdd_avg = fdd.grids.reshape(fdd.config['real_shape'])
        tdd_avg = tdd.grids.reshape(tdd.config['real_shape'])
    else:
        start, end = years.split('-')
        fdd_avg = fdd.get_grids_at_keys(range(int(start),int(end)+1))
        tdd_avg = tdd.get_grids_at_keys(range(int(start),int(end)+1))
    fdd_std = fdd_avg.std(0)
    tdd_std = tdd_avg.std(0)
    fdd_avg = fdd_avg.mean(0)
    tdd_avg = tdd_avg.mean(0)

    input_count = 3
    if winter_precip_type == 'both':
        input_count = 4

    # raster metadata 
    transform = precip.config['raster_metadata'].transform
    projection = precip.config['raster_metadata'].projection

    os.makedirs(os.path.join(directory, years))
    raster.save_raster(
        os.path.join(directory, years + '_precip_full_winter_avg.tif'), 
        full_winter_precip_avg, transform, projection)
    raster.save_raster(
        os.path.join(directory, years + '_precip_early_winter_avg.tif'), 
        early_winter_precip_avg, transform, projection)
    raster.save_raster(
        os.path.join(directory, years + '_fdd_avg.tif'),
        fdd_avg, transform, projection
    )
    raster.save_raster(
        os.path.join(directory, years + '_tdd_avg.tif'), 
        tdd_avg, transform, projection
    )

    ## find the areas
    shape = tdd.config['grid_shape']
    fdd_grid = np.zeros(shape)  
    fdd_grid[::] = np.nan
    # show(fdd_grid)

    tdd_grid = np.zeros(shape)  
    tdd_grid[::] = np.nan

    full_precip_grid = np.zeros(shape)  
    full_precip_grid[::] = np.nan

    early_precip_grid = np.zeros(shape)  
    early_precip_grid[::] = np.nan

    warm_winter = np.zeros(shape)  
    warm_winter[::] = np.nan

    

    _max = -10
    _min = 20
    
    for idx in range(fdd.grids.shape[0]):

        # current year values
        c_fdd = fdd.grids[idx].reshape(shape)
        c_tdd = tdd.grids[idx].reshape(shape)
        c_full_precip = full_winter_precip[idx].reshape(shape)
        c_early_precip = early_winter_precip[idx].reshape(shape)

        # grids for mapping deviation 
        fdd_grid[::] = c_fdd - c_fdd
        tdd_grid[::] = c_tdd - c_tdd
        full_precip_grid[::] = c_full_precip  - c_full_precip 
        early_precip_grid[::] = c_early_precip  - c_early_precip 

        # for  deviation grids:
        #   0 means <= average
        #   1 means > average
        #   2 means > 1 std. deviation. 
        #   2 means > 2 std. deviations. 
        # for s in [0, 1, 2]:
            # fdd_grid[c_fdd > (fdd_avg + s * fdd_std)] = s + 1  
            # tdd_grid[c_tdd > (tdd_avg + s * tdd_std)] = s + 1 
            # full_precip_grid[
            #     c_full_precip  > (full_winter_precip_avg + s * full_winter_precip_std)
            # ] = s + 1 
            # early_precip_grid[
            #     c_full_precip  > (early_winter_precip_avg + s * early_winter_precip_std)
            # ] = s + 1 
        fdd_grid = ((c_fdd - fdd_avg) / np.abs(fdd_avg)) * 100
        tdd_grid = ((c_tdd - tdd_avg) / np.abs(tdd_avg)) * 100

        full_precip_grid = ((c_full_precip - full_winter_precip_avg) / np.abs(full_winter_precip_avg)) * 100
        early_precip_grid = ((c_early_precip - early_winter_precip_avg) / np.abs(early_winter_precip_avg)) * 100



        # calculate the initiation map
        # 'warm winters' (high precip + high temps) + hot summers = higher initiation 
        initiation = (warm_winter + (tdd_grid)) / input_count
        _min = min(_min, np.nanmin(initiation))
        _max = max(_max, np.nanmax(initiation))
        

        # save rastes
        yr = str(int(start)+idx)
        raster.save_raster(os.path.join(directory, years, yr + '_initialization_areas.tif'), initiation, transform, projection)
        raster.save_raster(os.path.join(directory, years, yr + '_full_precip_gtavg.tif'), full_precip_grid, transform, projection)
        raster.save_raster(os.path.join(directory, years, yr + '_early_precip_gtavg.tif'), early_precip_grid, transform, projection)
        raster.save_raster(os.path.join(directory, years, yr + '_fdd_gtavg.tif'), fdd_grid, transform, projection)
        raster.save_raster(os.path.join(directory, years, yr + '_tdd_gtavg.tif'), tdd_grid, transform, projection)

        # high precip + high temps 
        if winter_precip_type == 'full':
            warm_winter = (fdd_grid) + full_precip_grid
        elif winter_precip_type == 'early':
            warm_winter = (fdd_grid) + early_precip_grid
        elif winter_precip_type == 'both':
            warm_winter = (fdd_grid) + full_precip_grid + early_precip_grid
        else:
            m = "Argument winter_precip_type must be 'early', 'full', or 'both'"
            raise TypeError(m)

    print(_min,_max)
    return full_winter_precip_avg, full_winter_precip_std, early_winter_precip_avg, early_winter_precip_std, tdd_avg, tdd_std, fdd_avg, fdd_std


def find_initiation_areas_vpdm(grid_dict, mean_bounds):
    """variable percent difference method for finding potential
    thermokarst initiation areas. Allows differnet variables to 
    be specified. 

    Parameters
    ----------
    grid_dict: dict of TemporalGrids
        keys should be variable names. all temporal grids need ts_offset in
    their configuration, this is used to determine which years data is used
    relative to the current timestamp being assemble.

    Returns
    -------
    ia_grid: 
        TemporalGrid of initiation areas.
    means: dict
        dict of means
    deviations: dict
        dict of standard deviations
    """
    grid_list = list(grid_dict.values()) 
    rows, cols = grid_list[0].config['grid_shape']
    time_steps = grid_list[0].config['num_timesteps']

    ia_grid = temporal_grid.TemporalGrid(rows, cols, time_steps)
    start = grid_list[0].config['start_timestep']
    end = start + time_steps

    ia_grid.config['start_timestep'] = start



    means = {}
    deviations = {}

    # calculate means and std. deviations
    years = range(mean_bounds[0], mean_bounds[1]+1)
    # print (grid_dict)
    for name, var in grid_dict.items():
        # print(var, name)
        means[name] = var.calc_statistics_for(years)
        deviations[name] = var.calc_statistics_for(years, np.std)

   
    for year in range(start, end):
        temp = np.zeros([rows, cols])
        ia_year = year
        for name, var in grid_dict.items():
            offset_year = year + var.config['ts_offset']
            ia_year = max(ia_year, offset_year)
            # print (name, year, offset_year)
            try:
                pd = (
                    (var[offset_year] - means[name])/ np.abs(means[name])
                ) * 100
                temp += pd
            except IndexError:
                # temp[:] = -9999 * len(grid_dict)
                break
            
        ia_grid[ia_year][:] = temp / len(grid_dict)

    stats_names = []
    for name in grid_dict:
        stats_names += [name+'-mean', name+'-std-dev']


    stats_grid = multigrid.MultiGrid(
        rows, cols, len(stats_names), grid_names=stats_names
    )

    for name in grid_dict:
        stats_grid[name + '-mean'][:] = means[name]
        stats_grid[name + '-std-dev'][:] = deviations[name]
    

    return ia_grid, stats_grid
