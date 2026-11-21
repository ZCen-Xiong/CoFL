"""Conservative raster collision queries and opposing-side bottleneck filtering."""
from functools import lru_cache
import numpy as np
from scipy.ndimage import distance_transform_edt, minimum_filter


class GroundRaster:
    def __init__(self, free, resolution, y_min):
        self.free=np.asarray(free,bool); self.resolution=float(resolution); self.y_min=float(y_min)
        if self.free.ndim!=2 or not self.free.size or not np.isfinite(resolution) or resolution<=0 or not np.isfinite(y_min):
            raise ValueError('Expected 2D occupancy and positive resolution')
        # Occupied cells represent full squares, not point obstacles.
        padded=np.pad(self.free,1,constant_values=False)
        self.distance=distance_transform_edt(padded,sampling=resolution)[1:-1,1:-1]-resolution/2**.5

    @lru_cache(maxsize=16)
    def _minimum(self, rx, ry):
        return minimum_filter(self.distance,size=(2*ry+1,2*rx+1),mode='constant',cval=-1e6)

    def rectangle_clearance(self,left,right,bottom,top):
        shape=np.shape(left);r=self.resolution
        cx=(left+right)/2;cy=(bottom+top)/2
        ix=np.rint(cx/r).astype(int);iy=np.rint((cy-self.y_min)/r).astype(int)
        rx=np.ceil((right-left)/2/r+np.abs(cx/r-ix)).astype(int)
        ry=np.ceil((top-bottom)/2/r+np.abs((cy-self.y_min)/r-iy)).astype(int)
        out=np.full(shape,-1e6,dtype=float)
        for ax,ay in set(zip(rx.ravel(),ry.ravel())):
            mask=(rx==ax)&(ry==ay)&(ix>=0)&(ix<self.free.shape[1])&(iy>=0)&(iy<self.free.shape[0])
            if np.any(mask):out[mask]=self._minimum(int(ax),int(ay))[iy[mask],ix[mask]]
        return out

    def is_free(self, points):
        p=np.asarray(points);ix=np.rint(p[...,0]/self.resolution).astype(int)
        iy=np.rint((p[...,1]-self.y_min)/self.resolution).astype(int)
        ok=(ix>=0)&(iy>=0)&(ix<self.free.shape[1])&(iy<self.free.shape[0])
        return ok & self.free[np.clip(iy,0,self.free.shape[0]-1),np.clip(ix,0,self.free.shape[1]-1)]

    def point_clearance(self, x, y):
        """Lipschitz lower bound without inflating a point to a grid rectangle."""
        r=self.resolution
        gx=np.asarray(x)/r; gy=(np.asarray(y)-self.y_min)/r
        result=np.full(np.broadcast_shapes(gx.shape,gy.shape),-1e6)
        for ix in (np.floor(gx).astype(int),np.ceil(gx).astype(int)):
            for iy in (np.floor(gy).astype(int),np.ceil(gy).astype(int)):
                valid=(ix>=0)&(iy>=0)&(ix<self.free.shape[1])&(iy<self.free.shape[0])
                sample=self.distance[np.clip(iy,0,self.free.shape[0]-1),np.clip(ix,0,self.free.shape[1]-1)]
                bound=sample-np.hypot(gx-ix,gy-iy)*r
                result=np.maximum(result,np.where(valid,bound,-1e6))
        return np.where(self.is_free(np.stack(np.broadcast_arrays(x,y),-1)),result,-1e6)


def narrow_profile(path, raster, *, max_width=1.1, min_length=.3, context_m=.5):
    """Width is the occupied-to-occupied ray cross-section normal to path motion.

    Requires both opposing surfaces; does not use 2*nearest-obstacle distance.
    This is a raster cross-section estimate, not a certified physical doorway width.
    """
    path=np.asarray(path); widths=np.full(len(path),np.inf);lengths=np.zeros(len(path))
    if len(path)<2:return dict(accepted=False,selected_indices=[],narrow_length_m=0,widths_m=[])
    d=np.diff(path[:,:2],axis=0);lengths[1:]=np.linalg.norm(d,axis=1)
    rays=np.arange(raster.resolution,max_width+2*raster.resolution,raster.resolution)
    for i,delta in enumerate(d,1):
        norm=np.linalg.norm(delta)
        if norm<1e-8:
            # Shape-only transitions do not split a spatial bottleneck run.
            widths[i]=widths[i-1]
            continue
        normal=np.array([-delta[1],delta[0]])/norm
        side=[]
        for sign in (-1,1):
            hits=np.flatnonzero(~raster.is_free(path[i,:2]+sign*rays[:,None]*normal))
            side.append(rays[hits[0]] if len(hits) else np.inf)
        widths[i]=sum(side)
    narrow=widths<=max_width
    # Require a contiguous spatial run; isolated close samples are insufficient.
    runs=[];start=None; total_length=0.
    for i,flag in enumerate(np.r_[narrow,False]):
        if flag and start is None:start=i
        if not flag and start is not None:
            run_length=lengths[start+1:i].sum()
            if run_length>=min_length-1e-9:
                runs.extend(range(start,i))
                total_length+=run_length
            start=None
    arc=np.cumsum(lengths);selected=np.zeros(len(path),bool)
    for i in runs:selected |= np.abs(arc-arc[i])<=context_m
    return dict(accepted=bool(runs), selected_indices=np.flatnonzero(selected).tolist(),
                narrow_length_m=float(total_length),
                widths_m=[float(w) if np.isfinite(w) else None for w in widths],
                max_width_m=max_width,min_contiguous_length_m=min_length,context_m=context_m)
