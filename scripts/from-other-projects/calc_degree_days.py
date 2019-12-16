"""
Calc Degree Days
----------------

Tools for caclualting and storing spatial degree days values from temperature
"""
import numpy as np
from scipy import interpolate
from multiprocessing import Process, Lock, active_children, cpu_count
from copy import deepcopy

from datetime import datetime

from atm.grids.constants import ROW, COL

from .stack_rasters import load_and_stack

from multigrids.temporal_grid import TemporalGrid

def calc_degree_days(day_array, temp_array, expected_roots = None):
    """Calc degree days (thawing, and freezing)
    
    Parameters
    ----------
    day_array: list like
        day number for each temperature value. 
        len(days_array) == len(temp_array).
    temp_array: list like
        Temperature values. len(days_array) == len(temp_array).
    expected_roots: int, None
        # of roots expected to find
        
    Returns
    -------
    tdd, fdd: lists
        thawing and freezing degree day lists
    """
    spline = interpolate.UnivariateSpline(day_array, temp_array)
    if not expected_roots is None and len(spline.roots()) != expected_roots:
        print (len(spline.roots()))
        i = 1
        while len(spline.roots()) != expected_roots:
            spline.set_smoothing_factor(i)
            i+= 1
            #print len(spline.roots())
            if i >50:
                print ('expected roots is not the same as spline.roots()')
                return np.zeros(115) - np.inf,np.zeros(115) - np.inf

    tdd = []
    fdd = []
    for rdx in range(len(spline.roots())-1):
        val = spline.integral(spline.roots()[rdx], spline.roots()[rdx+1])
        if val > 0:
            tdd.append(val)
        else:
            fdd.append(val)

    return tdd, fdd


def calc_and_store  (
        index, day_array, temp_array, tdd_grid, fdd_grid, lock = Lock()
        ):
    """Caclulate degree days (thawing, and freezing) and store in to 
    a grid.
    
    Parameters
    ----------
    index: int
        Flattened grid cell index.
    day_array: list like
        day number for each temperature value. 
        len(days_array) == len(temp_array).
    temp_array: list like
        Temperature values. len(days_array) == len(temp_array).
    tdd_grid: np.array
        2d grid of # years by flattend grid size. TDD values are stored here.
    fdd_grid: np.array
        2d grid of # years by flattend grid size. FDD values are stored here.
    lock: multiprocessing.Lock, Optional.
        lock object, If not passed a new lock is created. 
        
    """
    expected_roots = 2 * len(tdd_grid)
    tdd, fdd  = calc_degree_days(day_array, temp_array, expected_roots)
    lock.acquire()
    tdd_grid[:,index] = tdd
    fdd_grid[:,index] = [fdd[0]] + fdd ## add frist value equal to second value
    lock.release()




def calc_gird_degree_days (
        day_array, temp_grid, tdd_grid, fdd_grid, shape, 
        start = 0, num_process = 1
        ):
    """Calculate degree days (Thawing, and Freezing) for an area. 
    
    Parameters
    ----------
    day_array: list like
        day number for each temperature value. 
        len(days_array) == temp_grid.shape[0].
    temp_grid: np.array
        2d numpy array temperature values of timestep by flattened grid size.
        len(days_array) == temp_grid.shape[0].
    tdd_grid: np.array
        Empty thawing degree day grid. Size should be # of years to model by 
        flattened grid size. Values are set based on integration of a curve
        caluclated from days_array, and a timeseries from temp_grid. If
        the timeseries is all no data values np.nan is set as all valus 
        for element. If the curve has too many roots, or too few 
        (# roots != 2*# years) -inf is set as all values.
    fdd_grid: np.array
        Empty freezing degree day grid. Size should be # of years to model by 
        flattened grid size. Values are set based on integration of a curve
        caluclated from days_array, and a timeseries from temp_grid. If
        the timeseries is all no data values np.nan is set as all valus 
        for element. If the curve has too many roots, or too few 
        (# roots != 2*# years) -inf is set as all values.
    shape: tuple
        shape of model domain
    start: int, defaults to 0 , or list
        Calculate values starting at flattened grid index equal to start.
        If it is a list, list values should be grid indices, and only 
        values for those indices will be caclulated.
    num_process: int, Defaults to 1.
        number of processes to use to do calcualtion. If set to None,
        cpu_count() value is used. If greater than 1 ttd_grid, and fdd_grid 
        should be memory mapped numpy arrays.
    """
    p_lock = Lock()
    w_lock = Lock()
    
    if num_process is None:
       num_process = cpu_count()
    
    if type(start) is int:   
        indices = range(start, temp_grid.shape[1])
    else:
        indices = start
    
    for idx in  indices: # flatted area grid index
        while len(active_children()) >= num_process:
            continue
        print ('calculating degree days for element ', idx)
        if (temp_grid[:,idx] == -9999).all():
            w_lock.acquire()
            tdd_grid[:,idx] = np.nan
            fdd_grid[:,idx] = np.nan
            w_lock.release()
            continue
        Process(target=calc_and_store,
            args=(idx,day_array,temp_grid[:,idx], tdd_grid, fdd_grid, w_lock)
        ).start()
    
    while len(active_children()) > 0 :
        continue
    
    ## fix missing cells
    m_rows, m_cols = np.where(tdd_grid[0].reshape(shape) == -np.inf)   
    #~ print  m_rows, m_cols
    cells = []
    for cell in range(len(m_rows)):
        f_index = m_rows[cell] * shape[1] + m_cols[cell]
        
        
        g_tdd = np.array(
            tdd_grid.reshape((tdd_grid.shape[0],shape[0],shape[1]))\
                [:,m_rows[cell]-1:m_rows[cell]+2,m_cols[cell]-1:m_cols[cell]+2]
        )
        
        g_fdd = np.array(
            fdd_grid.reshape((fdd_grid.shape[0],shape[0],shape[1]))\
                [:,m_rows[cell]-1:m_rows[cell]+2,m_cols[cell]-1:m_cols[cell]+2]
        )
        
    
        g_tdd[g_tdd == -np.inf] = np.nan
        g_fdd[g_fdd == -np.inf] = np.nan
        tdd_mean = np.nanmean(g_tdd.reshape(tdd_grid.shape[0],9),axis = 1)
        fdd_mean = np.nanmean(g_fdd.reshape(fdd_grid.shape[0],9),axis = 1)
        
        tdd_grid[:,f_index] = tdd_mean
        fdd_grid[:,f_index] = fdd_mean
        cells.append(f_index)
            
    return cells
        
    
        
        
def create_day_array (dates):
    """Calulates number of days after start day for each date in dates array
    
    Parameters
    ----------
    dates: datetime.datetime
        sorted list of dates
        
    Returns
    -------
    days: list
        list of interger days since first date. The first value will be 0.
        len(days) == len(dates)
    """
    init = dates[0]
    days = []
    for date in dates:
        days.append((date - init).days)
        
    return days

def npmm_to_mg (npmm, rows, cols, ts, cfg={}):
    """convert a numpy memory map file (npmm) to a Temporal grid (mg)
    """
    grid = TemporalGrid(rows, cols, ts)
    grid.config.update(cfg)
    grid.grids[:] = npmm.reshape(ts, rows*cols)[:]
    return grid


    
    
def utility ():
    """Utility for caclulating the freezing and thawing degree days into a 
    fromat that can be read as a memory mapped numpy array.
    
    Flags
    -----
    --monthly_temperature_list_file: path
        a file containg an ordered list of paths to monthly temperature rasters
        one per line
    --temperature_file: path
        path to store temp temperature memmap at
    --fdd_file: path
        name of Freezing Degree-days file to create
    --tdd_file: path
        name of Thawing Degree-days file to create
    --start_date: 
        'YYYY-MM'
        
    --num_process: int, optinal
        number process to use, will default to max processors on system
        
    Note this utility will not work if there are missing months

    """
    from . import clite 
    
    try:
        arguments = clite.CLIte([
            '--monthly_temperature_list_file',
            '--temperature_file',
            '--fdd_file', 
            '--tdd_file', 
            '--start_date'
            ],
            ['--num_process',]
        
        )
    except (clite.CLIteHelpRequestedError, clite.CLIteMandatoryError) as E:
        print (E)
        print(utility.__doc__)
        return


    print('Seting up input...')
    init_date = datetime.strptime(arguments['--start_date'], '%Y-%m')
    
    current_year = init_date.year
    current_month = init_date.month
    
    NP = arguments['--num_process']
    with open(arguments['--monthly_temperature_list_file']) as temps:
        text = temps.read().rstrip().replace('\r\n','\n')
        files = text.split('\n')
        
    dates = []
    for f in files:
        current_date = datetime.strptime(
            '{}-{:02d}'.format(current_year, current_month), '%Y-%m'
        )
        dates.append(current_date)
        current_month += 1
        if current_month > 12:
            current_year +=1
            current_month = 1
    days = create_day_array( dates )
    
    
    print('Stacking temprerature data...')
    temperatures, shape = load_and_stack(files, arguments['--temperature_file'])
    
    n_months = len(temperatures)
    
    t_shape = (n_months, shape[0] * shape[1])
    
    # close and reopen as read only
    del temperatures
    temperatures = np.memmap(
        arguments['--temperature_file'], dtype='float32', 
        mode='r', shape=t_shape
    )
    
    
    dd_shape = ((n_months // 12) , shape[0] * shape[1])
    fdd = np.memmap(
        arguments['--fdd_file'], dtype='float32', mode='w+', shape= dd_shape
    )
    tdd = np.memmap(
        arguments['--tdd_file'], dtype='float32', mode='w+', shape= dd_shape
    )
    
    print('Calculating Degree-days... this will take some time')
    calc_gird_degree_days(days, temperatures, tdd, fdd, shape, num_process = NP)
    



