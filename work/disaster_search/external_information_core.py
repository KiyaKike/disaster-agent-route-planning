"""Information-bounded RGB transforms and canonical paid-cell masks."""
import numpy as np
from PIL import Image

from external_area_core import CELLS,box

PATCH_MODES=("patch128_to512","blank512_plus_native_cell","free_coarse512_plus_native_cell")
COARSE_MODE="free_coarse512_only"
BLANK_RGB=(124,116,104)


def require_pair(pair,size):
    if not isinstance(pair,(tuple,list)) or len(pair)!=2:raise ValueError("Exactly two RGB images required")
    if any(not isinstance(im,Image.Image) or im.mode!="RGB" or im.size!=size for im in pair):raise ValueError("Wrong RGB information boundary")


def coarse_canvas(coarse_pair):
    require_pair(coarse_pair,(64,64))
    return tuple(im.resize((512,512),Image.Resampling.BILINEAR) for im in coarse_pair)


def compose_inputs(mode,patch_pair,cell,coarse_pair=None):
    """Never accepts the high-resolution scene; only64px coarse and128px ROI."""
    bounds=box(cell);require_pair(patch_pair,(128,128))
    if mode==PATCH_MODES[0]:
        if coarse_pair is not None:raise ValueError("Scale-only input must not receive coarse context")
        return tuple(im.resize((512,512),Image.Resampling.BILINEAR) for im in patch_pair)
    if mode==PATCH_MODES[1]:
        if coarse_pair is not None:raise ValueError("Blank control must not receive coarse context")
        canvases=tuple(Image.new("RGB",(512,512),BLANK_RGB) for _ in range(2))
    elif mode==PATCH_MODES[2]:canvases=coarse_canvas(coarse_pair)
    else:raise ValueError("Unplanned input condition")
    for canvas,patch in zip(canvases,patch_pair):canvas.paste(patch,bounds[:2])
    return canvases


def canonical_mask(full_mask,mode,cell):
    bounds=box(cell)
    if full_mask.dtype!=np.bool_ or full_mask.shape!=(512,512):raise ValueError("Boolean512 prediction required")
    if mode==PATCH_MODES[0]:
        return np.asarray(Image.fromarray(full_mask).resize((128,128),Image.Resampling.NEAREST),dtype=np.bool_).copy()
    if mode in PATCH_MODES[1:] or mode==COARSE_MODE:
        x0,y0,x1,y1=bounds;return full_mask[y0:y1,x0:x1].copy()
    raise ValueError("Unplanned output condition")


def local_summary(mask):
    if mask.dtype!=np.bool_ or mask.shape!=(128,128):raise ValueError("Canonical128 prediction required")
    count=int(mask.sum())
    return {"predicted_severe_pixels":count,"declared_target":count>=4}
