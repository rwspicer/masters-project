## commands used to minipulate raster
from atm.images import raster
raster.clip_raster(
    "/Volumes/toshi-stati/data/snap/pr_total_mm_iem_cru_TS31_1901_2015/pr_total_mm_CRU_TS40_historical_01_1901.tif",
    "test.tif"
    [-530818.761, 2373989.086, 565441.239, 1959449.086]
)
raster.clip_raster(
    "/Volumes/toshi-stati/data/snap/pr_total_mm_iem_cru_TS31_1901_2015/pr_total_mm_CRU_TS40_historical_01_1901.tif",
    "test.tif",
    [-530818.761, 2373989.086, 565441.239, 1959449.086]
)
in_dir = "/Volumes/toshi-stati/data/snap/pr_total_mm_iem_cru_TS31_1901_2015/"
import os
os.listdir(in_dir)
import glob
glob.glob(in_dir)
glob.glob(in_dir + "*.tif")
glob.glob(in_dir + "*.tif")[0]
len(glob.glob(in_dir + "*.tif"))
len(glob.glob(in_dir + "*.tif")) /12
glob.glob(in_dir + "*.tif")
infiles = glob.glob(in_dir + "*.tif")
in_files = glob.glob(in_dir + "*.tif")
for file_path in in_files:
    file = os.path.split(file_path[1])
    print(file)
for file_path in in_files:
    file = os.path.split(file_path[1])
    print(file_path)
for file_path in in_files:
    file = os.path.split(file_path)[1]
    print(file_path)
for file_path in in_files:
    file = os.path.split(file_path)[1]
    print(file)
out_dir = "ACP_pricp_tiff_monthly/"
for file_path in in_files:
    file = os.path.split(file_path)[1]
    print(file.split(file))
    #raster.clip_raster(file_path, os.path.join(out_dir, )
out_dir = "ACP_pricp_tiff_monthly/"
for file_path in in_files:
    file = os.path.split(file_path)[1]
    print(file.split('_'))
    #raster.clip_raster(file_path, os.path.join(out_dir, )
out_dir = "ACP_pricp_tiff_monthly/"
for file_path in in_files:
    file = os.path.split(file_path)[1]
    print(file[:-4].split('_'))
    #raster.clip_raster(file_path, os.path.join(out_dir, )
out_dir = "ACP_pricp_tiff_monthly/"
for file_path in in_files:
    file = os.path.split(file_path)[1]
    month, year = file[:-4].split('_')[-2:]
    #raster.clip_raster(file_path, os.path.join(out_dir, )
    print (month, year)
out_dir = "ACP_pricp_tiff_monthly/"
for file_path in sorted(in_files):
    file = os.path.split(file_path)[1]
    month, year = file[:-4].split('_')[-2:]
    #raster.clip_raster(file_path, os.path.join(out_dir, )
    print (month, year)
out_dir = "ACP_pricp_tiff_monthly/"
for file_path in sorted(in_files):
    file = os.path.split(file_path)[1]
    month, year = file[:-4].split('_')[-2:]
    out_file = os.path.join(out_dir, "ACP_precip_mm_" + year + "_" + month + ".tif") 
    #raster.clip_raster(file_path, os.path.join(out_dir, )
    print (out_file)
out_dir = "ACP_pricp_tiff_monthly/"
for file_path in sorted(in_files):
    file = os.path.split(file_path)[1]
    month, year = file[:-4].split('_')[-2:]
    out_file = os.path.join(out_dir, "ACP_precip_mm_" + year + "_" + month + ".tif") 
    raster.clip_raster(
        file_path
        out_file 
        [-530818.761, 2373989.086, 565441.239, 1959449.086] 
    )
out_dir = "ACP_pricp_tiff_monthly/"
for file_path in sorted(in_files):
    file = os.path.split(file_path)[1]
    month, year = file[:-4].split('_')[-2:]
    out_file = os.path.join(out_dir, "ACP_precip_mm_" + year + "_" + month + ".tif") 
    raster.clip_raster(
        file_path,
        out_file, 
        [-530818.761, 2373989.086, 565441.239, 1959449.086] 
    )
from multigrids.temporal_grid import TemporalGrid
monthly_precip = TemporalGrid(415, 1096, 12 * 115)
monthly_preip =
monthly_preip
monthly_precip
monthly_precip[0].shape
monthly_precip.config['start_timestep'] = 1901
monthly_precip.config['dataset_name'] = "ACP monthly precip"
monthly_precip.config['data_type']
monthly_precip.config['description'] = "ACP monthly precipitation data derived from snap pr_total_mm_iem_cru_TS31_1901_2015 data set"
monthly_precip.config['units'] = mm
monthly_precip.config['units'] = 'mm'
monthly_precip[1901]
monthly_precip[1901].shape
mp_old = TemporalGrid("data/precipitation/ACP/V1/monthly/TemporalGrid/precip-1901-2015-monthly.yml")
mp_old.config
mp_old.config["grid_name_map"]
monthly_precip.config['grid_name_map'] = mp_old.config["grid_name_map"]
mp_old.config['start_timestep']
monthly_precip.config['start_timestep'] = 0
monthly_precip['1901-01']
out_dir
glob.golb(out_dir + "*.tif")
glob.glob(out_dir + "*.tif")
tiff_data = "data/precipitation/ACP/V2/monthly/tiff/"
glob.glob(tiff_data + "*.tif")
sorted(glob.glob(tiff_data + "*.tif"))
sorted(glob.glob(tiff_data + "*.tif"))[0]
sorted(glob.glob(tiff_data + "*.tif"))[-1]
tiff_files = sorted(glob.glob(tiff_data + "*.tif"))
for path in tiff_files:
    year, month = os.path.split(path)[1][:-4].split('_')[-2:]
    print(year, month)
for path in tiff_files:
    year, month = os.path.split(path)[1][:-4].split('_')[-2:]
    print(year, month)
    data, md = raster.load_raster(path)
    monthly_precip.config['raster_metadata'] = md
    monthly_precip[ year + "-" + month] = data
monthly_precip['1901-01']
monthly_precip.show_figure('1901-01')
import matplotlib.pyplot as plt
plt.imshow(monthly_precip['1901-01'])
monthly_precip.grids
monthly_precip.grids == -9999
import numpy as np
idx = monthly_precip.grids == -9999
monthly_precip.grids[idx] = np.nan
plt.imshow(monthly_precip['1901-01'])
monthly_precip.save("data/precipitation/ACP/V2/monthly/temporal_grid/ACP_precip_mm.yml")
%sace
%save
%save stuff.py
%save stuff
%history stuff.py
%history -f stuff.py
