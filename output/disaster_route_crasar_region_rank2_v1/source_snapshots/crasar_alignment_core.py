"""Published polygon-to-image pixel translations, without label use."""
import numpy as np
from scipy.spatial import cKDTree

def nearest_shift_indices(origins,x,y):
    origins=np.asarray(origins,float);x=np.asarray(x,float);y=np.asarray(y,float)
    if origins.ndim!=2 or origins.shape[1]!=2 or x.shape!=y.shape or len(origins)==0:raise ValueError('2D origins and matching coordinates required')
    if not all(np.isfinite(a).all() for a in [origins,x,y]):raise ValueError('Finite coordinates required')
    return cKDTree(origins).query(np.column_stack((x.ravel(),y.ravel())))[1].reshape(x.shape)

def corrected_image_transform(native_transform,source_transform,shift):
    # Published adjustment moves footprint pixel p to image pixel p+shift.
    # Shifting image georeferencing by -shift maps image back onto canonical footprint p.
    from affine import Affine
    scale=np.array([native_transform.a/source_transform.a,native_transform.e/source_transform.e])
    return source_transform*Affine.translation(*(-np.asarray(shift)*scale))
