## notes on some GDAL commands

Upper Left  ( -530818.761, 2373989.086) (168d16'56.97"W, 70d50'21.70"N)
Lower Left  ( -530818.761, 1959449.086) (166d15' 6.65"W, 67d 9' 3.54"N)
Upper Right (  565441.239, 2373989.086) (138d49' 1.66"W, 70d46'10.54"N)
Lower Right (  565441.239, 1959449.086) (140d58' 7.62"W, 67d 5'31.01"N)
Center      (   17311.239, 2166719.086) (153d33'50.71"W, 69d28'13.56"N)


With Gdal / qgis

    <!-- gdalwarp -r near -of GTiff -tr 1000 1000 /docker-qgis-shared/ned_60m_dtm_hs.tif /docker-qgis-shared/AK-DEM1000m.tif -->
    gdalwarp -t_srs EPSG:3338 -tr 1000.0 1000.0 -r near -of GTiff /docker-qgis-shared/alaska/ned_60m_dtm_hs.tif /docker-qgis-shared/alaska-3338/AK-DEM-1000m.tif

    <!-- gdaldem aspect /docker-qgis-shared/ACP-DEM1000m.tif /docker-qgis-shared/AK-ASPECT1000m.tif -of GTiff -b 1 -->
    gdaldem aspect /docker-qgis-shared/alaska-3338/AK-DEM-1000m.tif /docker-qgis-shared/alaska-3338/AK-ASPECT1000m.tif -of GTiff -b 1

    <!-- gdaldem slope /docker-qgis-shared/AK-DEM1000m.tif /docker-qgis-shared/AK-SLOPE1000m.tif -of GTiff -b 1 -s 1.0 -->
    gdaldem slope /docker-qgis-shared/alaska-3338/AK-DEM-1000m.tif /docker-qgis-shared/alaska-3338/AK-SLOPE1000m.tif -of GTiff -b 1 -s 1.0


using my python code/gdal translate
    raster.clip_raster(  
        "Desktop/docker-qgis-shared/AK-DEM1000m.tif", 
        "Desktop/docker-qgis-shared/ACP-DEM1000m.tif", 
        [-530818.761, 2373989.086, 565441.239, 1959449.086] 
    )          

    raster.clip_raster(  
            "Desktop/docker-qgis-shared/AK-ASPECT1000m.tif", 
            "Desktop/docker-qgis-shared/ACP-ASPECT1000m.tif", 
            [-530818.761, 2373989.086, 565441.239, 1959449.086] 
        )                                                                      

    raster.clip_raster(  
            "Desktop/docker-qgis-shared/AK-SLOPE1000m.tif", 
            "Desktop/docker-qgis-shared/ACP-SLOPE1000m.tif", 
            [-530818.761, 2373989.086, 565441.239, 1959449.086] 
        )     
