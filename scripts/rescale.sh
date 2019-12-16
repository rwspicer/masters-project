#!/bin/bash

### script to bulk rescale raster pixels

for filename in tiff/*.tif; do
    echo $(basename $filename)
    gdalwarp -r near -of GTiff -tr 1000 1000 tiff/$(basename $filename) tiff2/$(basename $filename)

done
