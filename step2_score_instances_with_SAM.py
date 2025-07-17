# Copyright (C) 2024 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import cmapy
import numpy as np
import sanghyunjo as shjo

from torch.nn import functional as F

from core import evaluators
from core.sam2 import SAM2

if __name__ == '__main__':
    args = shjo.Parser(
        {
            'image': './data/MoNuSeg/train/image/',

            'pred': './submissions/Ours+PSM@CRF(G=2)/train_instance/',
            'mask': './submissions/Ours+PSM@CRF(G=2)/train_sam/',

            'sam': './weights/sam2.1_hiera_l.pt',
            'threshold': -1.,
            'strategy': 'mean',
        }
    )

    model_sam = SAM2(args.sam)

    mask_dir = shjo.makedir(args.mask)
    semantic_dir = shjo.makedir(args.mask.replace('_sam', '_sam2semantic'))
    score_dir = shjo.makedir(args.mask.replace('_sam', '_sam2score'))

    colors = shjo.get_colors() # RGB color
    colors[0] = [0, 0, 0]      # background
    colors[1] = [32, 167, 132] # foreground (e.g., cell)
    # colors = colors[:, ::-1]

    for pred_path in shjo.progress(sorted(shjo.listdir(args.pred + '*.png'))):
        pred_path = pred_path.replace('\\', '/')
        mask_name = shjo.basename(pred_path)
        image_id = mask_name.replace('.png', '')

        pred_mask = shjo.imread(args.pred + image_id + '.png')
        pred_mask = pred_mask.astype(np.int64)
        pred_mask = pred_mask[:, :, 0] * 256 + pred_mask[:, :, 1]

        cv_image = shjo.imread(shjo.listdir(args.image + image_id + '.*')[0])

        model_sam.set_image(cv_image)

        scores = []
        masks = []

        for index in shjo.progress(np.unique(pred_mask), image_id):
            if index == 0: continue

            pred_ins_mask = (pred_mask == index).astype(np.uint8)

            cy, cx = map(lambda x: int(np.average(x)), np.where(pred_ins_mask > 0))
            sam_ins_mask = model_sam.predict(point_coords=[(cx, cy)], point_labels=[1])

            IoU = evaluators.get_IoU(pred_ins_mask, sam_ins_mask)

            scores.append(IoU)
            masks.append(sam_ins_mask > 0)

        scores = np.asarray(scores)
        masks = np.asarray(masks)
        
        sorted_indices = np.argsort(scores) # [::-1]

        sorted_masks = masks[sorted_indices]
        sorted_scores = scores[sorted_indices]

        if args.threshold == -1:
            if args.strategy == 'mean':
                threshold = scores.mean()
            else:
                threshold = min(scores.mean() + scores.std(), 0.9)
        else:
            threshold = args.threshold
        
        sorted_masks = sorted_masks[sorted_scores > threshold]
        
        instance_mask = np.zeros_like(pred_mask).astype(np.int64)
        instance_score = np.zeros_like(pred_mask).astype(np.float32)

        for instance_id, (mask, score) in enumerate(zip(sorted_masks, sorted_scores)):
            instance_mask[mask > 0] = instance_id + 1
            instance_score[mask > 0] = score

        # shjo.write_image(score_dir + image_id + '.png', (instance_score * 255).astype(np.uint8), cmapy.cmap('seismic', rgb_order=True)[:, 0, :])
        shjo.imwrite(score_dir + image_id + '.png', shjo.colorize((instance_score * 255).astype(np.uint8)))
        
        h, w = instance_mask.shape
        reliable_mask = np.zeros((h, w, 3), dtype=np.uint8)
        reliable_mask[pred_mask > 0, :] = 255

        for index in np.unique(instance_mask):
            if index == 0: continue # ignore background
            reliable_mask[instance_mask == index, 0] = index // 256 # B channel
            reliable_mask[instance_mask == index, 1] = index  % 256 # G channel
            reliable_mask[instance_mask == index, 2] = 0            # R channel
        
        shjo.imwrite(args.mask + image_id + '.png', reliable_mask)

        # instance to semantic
        pred_mask = reliable_mask.astype(np.int64)

        ignore_mask = pred_mask.sum(axis=2) == (255*3)
        pred_mask[ignore_mask, :] = 0 

        pred_mask = pred_mask[:, :, 0] * 256 + pred_mask[:, :, 1]
        pred_mask = (pred_mask > 0).astype(np.uint8)
        
        pred_mask[ignore_mask] = 255

        shjo.imwrite(semantic_dir + image_id + '.png', pred_mask, colors)