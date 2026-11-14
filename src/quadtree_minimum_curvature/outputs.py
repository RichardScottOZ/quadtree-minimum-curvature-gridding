from pathlib import Path
import numpy as np

def _bounds(g,b):
    if b is None:return float(g.x.min()),float(g.x.max()),float(g.y.min()),float(g.y.max())
    if len(b)!=4:raise ValueError("bounds must be (xmin, xmax, ymin, ymax)")
    b=tuple(map(float,b))
    if not np.isfinite(b).all() or b[0]>=b[1] or b[2]>=b[3]:raise ValueError("invalid bounds")
    return b

def grid_vectors(g,nx,ny,bounds=None):
    nx,ny=int(nx),int(ny)
    if nx<1 or ny<1:raise ValueError("nx and ny must be positive")
    xmin,xmax,ymin,ymax=_bounds(g,bounds); x=np.linspace(xmin,xmax,nx); y=np.linspace(ymin,ymax,ny); xx,yy=np.meshgrid(x,y)
    return x,y,np.asarray(g.predict(xx,yy))

def grid_spacing(g,dx,dy=None,bounds=None):
    dx=float(dx);dy=dx if dy is None else float(dy)
    if not np.isfinite([dx,dy]).all() or dx<=0 or dy<=0:raise ValueError("dx and dy must be finite and positive")
    xmin,xmax,ymin,ymax=_bounds(g,bounds); x=xmin+np.arange(int(np.floor((xmax-xmin)/dx))+1)*dx; y=ymin+np.arange(int(np.floor((ymax-ymin)/dy))+1)*dy; xx,yy=np.meshgrid(x,y)
    return x,y,np.asarray(g.predict(xx,yy))

def to_xarray(g,*,nx=None,ny=None,dx=None,dy=None,bounds=None,crs=None,name="value",attrs=None,nodata=np.nan):
    try: import xarray as xr
    except ImportError as e: raise ImportError("install with: pip install '.[raster]'") from e
    if dx is not None:
        if nx is not None or ny is not None:raise ValueError("choose spacing or shape")
        x,y,z=grid_spacing(g,dx,dy,bounds)
    else:
        if nx is None or ny is None:raise ValueError("provide dx, or nx and ny")
        x,y,z=grid_vectors(g,nx,ny,bounds)
    da=xr.DataArray(z,coords={"y":y,"x":x},dims=("y","x"),name=name,attrs=dict(attrs or {}));da.attrs.setdefault("reconstruction",getattr(g,"reconstruction","cell"))
    if crs is not None:
        try: import rioxarray  # noqa
        except ImportError as e: raise ImportError("CRS output requires rioxarray") from e
        da=da.rio.set_spatial_dims(x_dim="x",y_dim="y",inplace=False).rio.write_crs(crs,inplace=False).rio.write_nodata(nodata,inplace=False)
    return da

def to_netcdf(g,path,**kwargs):
    da=to_xarray(g,**kwargs);da.to_dataset(name=da.name or "value").to_netcdf(path);return Path(path)
def to_zarr(g,path,*,mode="w",consolidated=True,**kwargs):
    da=to_xarray(g,**kwargs);da.to_dataset(name=da.name or "value").to_zarr(path,mode=mode,consolidated=consolidated);return Path(path)
def to_geotiff(g,path,*,crs,dtype="float32",compress="deflate",tiled=True,**kwargs):
    da=to_xarray(g,crs=crs,**kwargs)
    if da.y.size>1 and da.y.values[0]<da.y.values[-1]:da=da.sortby("y",ascending=False)
    da.rio.to_raster(path,dtype=dtype,compress=compress,tiled=tiled);return Path(path)
def install(cls):
    for f in (grid_vectors,grid_spacing,to_xarray,to_netcdf,to_zarr,to_geotiff):setattr(cls,f.__name__,f)
    return cls
