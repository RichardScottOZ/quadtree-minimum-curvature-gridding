# Quadtree minimum-curvature gridding v2

Complete installable package with adaptive quadtree fitting and rectilinear,
xarray/rioxarray, NetCDF, Zarr and GeoTIFF outputs.

```bash
pip install -e ".[raster]"
```

```python
from quadtree_minimum_curvature import QuadtreeMinCurvatureGridder
g=QuadtreeMinCurvatureGridder(x,y,z).fit()
gx,gy,Z=g.grid_spacing(dx=50,dy=50)
da=g.to_xarray(dx=50,crs="EPSG:32719",name="surface")
g.to_netcdf("surface.nc",dx=50,crs="EPSG:32719")
g.to_zarr("surface.zarr",dx=50,crs="EPSG:32719")
g.to_geotiff("surface.tif",dx=50,crs="EPSG:32719")
```

The core dependency set remains NumPy only, matching the original implementation.
Install the `raster` extra for xarray/rioxarray and file exports.
