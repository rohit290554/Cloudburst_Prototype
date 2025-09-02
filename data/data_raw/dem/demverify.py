import rasterio

with rasterio.open("n28_e076_1arc_v3.tif") as dem:
    print("DEM bounds:", dem.bounds)
    print("Resolution:", dem.res)
    print("Shape:", dem.height, "x", dem.width)