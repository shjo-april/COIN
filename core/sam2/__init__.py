# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import torch
import numpy as np
import sanghyunjo as shjo

from hydra import initialize_config_module
initialize_config_module("core.sam2.configs", version_base="1.2")

from .build_sam import build_sam2
from .sam2_image_predictor import SAM2ImagePredictor
from .automatic_mask_generator import SAM2AutomaticMaskGenerator, SamEdgeGenerator

class EdgeSAM2:
    def __init__(self, pt_path, batch=64, details=False, device=torch.device('cuda:0')):
        sam = build_sam2(shjo.basename(pt_path.replace('.pt', '.yaml')), pt_path, device, apply_postprocessing=False)

        params = {
            'points_per_batch': batch
        }
        if details:
            params['points_per_side'] = 64  # 32
            params['pred_iou_thresh'] = 0.7 # 0.8
            params['stability_score_thresh'] = 0.92 # 0.95
            params['stability_score_offset'] = 1.00 # 0.7
            params['crop_n_layers'] = 1 # 0
            params['crop_n_points_downscale_factor'] = 2 # 1
            params['min_mask_region_area'] = 25.0 # 0
            params['use_m2m'] = True # False

        self.edge_predictor = SamEdgeGenerator(
            sam, 
            nms_threshold=0.7, 
            pred_iou_thresh_filtering=True,
            stability_score_thresh_filtering=False,
            **params
        )
        self.edge_detection = cv2.ximgproc.createStructuredEdgeDetection('./weights/edge.yml.gz')
    
    def predict(self, cv_image):
        cv_image = shjo.convert(cv_image, 'bgr2rgb')
        
        masks = self.edge_predictor.generate(cv_image)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)

        p_max = masks[0]["prob"]
        for mask in masks[1:]:
            p_max = np.maximum(p_max, mask['prob'])
        
        edges = (p_max - p_max.min()) / (p_max.max() - p_max.min())
        edges = self.edge_detection.edgesNms(edges, self.edge_detection.computeOrientation(edges))

        return edges
    
class SuperpixelSAM2:
    def __init__(self, pt_path, batch=64, details=False, device=torch.device('cuda:0')):
        sam = build_sam2(shjo.get_name(pt_path.replace('.pt', '.yaml')), pt_path, device, apply_postprocessing=False)
        
        # refer to: https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/automatic_mask_generator_example.ipynb
        params = {
            'points_per_batch': batch
        }
        if details:
            params['points_per_side'] = 64  # 32
            params['pred_iou_thresh'] = 0.7 # 0.8
            params['stability_score_thresh'] = 0.92 # 0.95
            params['stability_score_offset'] = 1.00 # 0.7
            params['crop_n_layers'] = 1 # 0
            params['crop_n_points_downscale_factor'] = 2 # 1
            # params['min_mask_region_area'] = 25.0 # 0
            params['use_m2m'] = True # False

        self.predictor = SAM2AutomaticMaskGenerator(sam, **params)
    
    @torch.no_grad()
    def predict(self, cv_image):
        cv_image = shjo.convert(cv_image, 'bgr2rgb')
        
        masks = self.predictor.generate(cv_image)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        return masks
    
class SAM2:
    def __init__(self, pt_path, device=torch.device('cuda:0')):
        sam = build_sam2(shjo.basename(pt_path.replace('.pt', '.yaml')), pt_path, device)
        self.mask_predictor = SAM2ImagePredictor(sam)

    def set_image(self, image):
        image = shjo.convert(image, 'bgr2rgb')
        self.mask_predictor.set_image(image.copy())
    
    """
    box = [115, 268, 1566, 2851]
    point_coords = [(100, 200), (300, 400)]
    point_labels = [1, 0] # 1: foreground, 0: background
    """
    def predict(self, box=[], point_coords=[], point_labels=[], mask_input=None):
        input_dict = {}

        if len(box) > 0: 
            input_dict['box'] = np.asarray([box])
        
        if len(point_coords):
            input_dict['point_coords'] = np.asarray(point_coords, dtype=np.int32)
            input_dict['point_labels'] = np.asarray(point_labels, dtype=np.int32)
        
        if mask_input is not None:
            input_dict['mask_input'] = np.asarray([mask_input])
        
        masks, scores, logits = self.mask_predictor.predict(
            multimask_output=True,
            **input_dict
        )

        sorted_indices = np.argsort(scores)[::-1]
        masks = masks[sorted_indices]
        scores = scores[sorted_indices]
        logits = logits[sorted_indices]

        return masks[0].astype(np.uint8)
    
    def get_features(self):
        return self.mask_predictor._features['image_embed'], self.mask_predictor._features['high_res_feats']
    
    # def get_prompt(self, binary):
    #     xmin, ymin, xmax, ymax = (0, 0, binary.shape[1]-1, binary.shape[0]-1)
    #     while np.sum(binary[:, xmin]) == 0: xmin += 1
    #     while np.sum(binary[ymin, :]) == 0: ymin += 1
    #     while np.sum(binary[:, xmax]) == 0: xmax -= 1
    #     while np.sum(binary[ymax, :]) == 0: ymax -= 1
        
    #     points, labels = binary2points(binary)
        
    #     return {
    #         'box': [xmin, ymin, xmax, ymax],
    #         'point_coords': points, 'point_labels': labels,
    #         'mask_input': _compute_logits_from_mask(binary)
    #     }

def binary2points(binary: np.ndarray, downsampling: int=4, center: float=0.0, k: int=10):
    h, w = binary.shape

    # down sample to smooth results
    dh, dw = h // downsampling, w // downsampling
    sh, sw = h / dh, w / dw

    down_binary = cv2.resize(binary, (dw, dh), interpolation=cv2.INTER_NEAREST)
    down_binary = down_binary.reshape(-1)

    fg_points = []
    fg_labels = []

    bg_points = []
    bg_labels = []

    # """ foreground points
    for idx in np.arange(dh*dw)[down_binary == 1]:
        x = min(int(((idx % dw + center) * sw).item()), w - 1)
        y = min(int(((idx // dw + center) * sh).item()), h - 1)
        fg_points.append((x, y)); fg_labels.append(1)
    # """

    # """ background points
    for idx in np.arange(dh*dw)[down_binary == 0]:
        x = min(int(((idx % dw + center) * sw).item()), w - 1)
        y = min(int(((idx // dw + center) * sh).item()), h - 1)
        bg_points.append((x, y)); bg_labels.append(0)
    # """
        
    fg_points, fg_labels = map(np.asarray, [fg_points, fg_labels])
    bg_points, bg_labels = map(np.asarray, [bg_points, bg_labels])
    # k = min(len(fg_points), len(bg_points))
    
    fg_indices = np.arange(len(fg_points))
    bg_indices = np.arange(len(bg_points))

    np.random.shuffle(fg_indices)
    np.random.shuffle(bg_indices)

    fg_points = fg_points[fg_indices[:k]]
    fg_labels = fg_labels[fg_indices[:k]]

    bg_points = bg_points[bg_indices[:k]]
    bg_labels = bg_labels[bg_indices[:k]]

    return list(fg_points)+list(bg_points), list(fg_labels)+list(bg_labels)

def _compute_logits_from_mask(mask, downsampling: int=4, eps=1e-3):
    inv_sigmoid = lambda x: np.log(x / (1 - x))
    mask = cv2.resize(mask, None, fx=1/downsampling, fy=1/downsampling, interpolation=cv2.INTER_NEAREST)

    logits = np.zeros(mask.shape, dtype="float32")
    logits[mask == 1] = 1 - eps
    logits[mask == 0] = eps
    logits = inv_sigmoid(logits)
    
    return logits