import richdem as rd

dem_data = rd.LoadGDAL("n28_e076_1arc_v3.tif")
slope = rd.TerrainAttribute(dem_data, attrib='slope_degrees')