import torch
import numpy as np

from scipy import ndimage
from scipy.ndimage import morphology

from skimage.morphology import disk
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def semantic2instance(semantic_mask, connectivity=4, denoise=False):
    if denoise:
        semantic_mask = ndimage.binary_opening(semantic_mask, structure=disk(2)).astype(np.uint8)
        semantic_mask = ndimage.binary_closing(semantic_mask, structure=disk(2)).astype(np.uint8)
        semantic_mask = ndimage.binary_opening(semantic_mask, structure=disk(1)).astype(np.uint8)
        semantic_mask = ndimage.binary_closing(semantic_mask, structure=disk(1)).astype(np.uint8)

    dist_map = morphology.distance_transform_edt(semantic_mask)
    dist_map = ndimage.gaussian_filter(dist_map, sigma=1)
    
    peak_points = peak_local_max(
        dist_map, 
        footprint=disk(12), 
        labels=semantic_mask,
    )
    peak_mask = np.zeros_like(semantic_mask)
    for y, x in peak_points: peak_mask[y, x] = 1

    ccl_mask = ndimage.label(peak_mask, None if connectivity == 4 else np.ones((3, 3), dtype=np.uint8))[0]
    instance_mask = watershed(-dist_map, ccl_mask, mask=semantic_mask)
    
    return instance_mask
