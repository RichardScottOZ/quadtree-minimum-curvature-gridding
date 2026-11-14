import numpy as np
from quadtree_minimum_curvature import QuadtreeMinCurvatureGridder

def test_rectilinear_and_xarray():
 x=np.array([0,.1,.3,.6,.8,1.]);y=np.array([0,.05,.7,.4,.85,1.]);z=x+2*y
 g=QuadtreeMinCurvatureGridder(x,y,z,max_level=4,points_per_leaf=1).fit()
 gx,gy,Z=g.grid_spacing(.1)
 assert gx.ndim==gy.ndim==1 and Z.shape==(gy.size,gx.size)
 da=g.to_xarray(dx=.1,name="surface")
 assert da.dims==("y","x") and da.shape==Z.shape
