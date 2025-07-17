import numpy as np
import sanghyunjo as shjo
import cv2
from scipy import ndimage
import skimage
from skimage.segmentation import watershed

def instance2gradient(instance_mask):
    semantic_mask = (instance_mask > 0).astype(np.uint8)
    origin_mask = instance_mask.copy()

    x_map = np.zeros(origin_mask.shape[:2], dtype=np.float32)
    y_map = np.zeros(origin_mask.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(origin_mask))
    inst_list.remove(0)
    for inst_id in inst_list:
        inst_map = np.array(origin_mask == inst_id, np.uint8)
        inst_box = get_bounding_box(inst_map)

        if inst_box[0] >= 2:
            inst_box[0] -= 2
        if inst_box[2] >= 2:
            inst_box[2] -= 2
        if inst_box[1] <= origin_mask.shape[0] - 2:
            inst_box[1] += 2
        if inst_box[3] <= origin_mask.shape[0] - 2:
            inst_box[3] += 2

        inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

        if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
            continue

        inst_com = list(ndimage.center_of_mass(inst_map))

        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)

        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

    hv_map = np.stack([x_map, y_map])

    return semantic_mask, hv_map

def get_bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

def remove_small_objects(pred, min_size=64):
    ccs = pred

    component_sizes = np.bincount(ccs.ravel())
    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    pred[too_small_mask] = 0

    return pred

def gradient2instance(semantic_mask, gradients):
    h_map = gradients[0]
    v_map = gradients[1]

    blb = np.array(semantic_mask > 0.5)
    blb = ndimage.measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1

    norm_params = {
        "alpha": 0,
        "beta": 1,
        "norm_type": cv2.NORM_MINMAX,
        "dtype": cv2.CV_32F,
    }
    h_dir = cv2.normalize(h_map, None, **norm_params)
    v_dir = cv2.normalize(v_map, None, **norm_params)

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (cv2.normalize(sobelh, None, **norm_params))
    sobelv = 1 - (cv2.normalize(sobelv, None, **norm_params))

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = np.array(overall >= 0.4)
    marker = blb - overall
    marker[marker < 0] = 0
    marker = ndimage.morphology.binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = ndimage.measurements.label(marker)[0]
    marker = remove_small_objects(marker, 10)
    instance_mask = watershed(dist, markers=marker, mask=blb)

    return instance_mask