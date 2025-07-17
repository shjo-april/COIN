# Copyright (C) 2024 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import torch
import numpy as np
import sanghyunjo as shjo

from torch.nn import functional as F

from core import refinements

def mask_normalize(_heatmaps):
    min_v = torch.min(_heatmaps.view(_heatmaps.shape[0], -1), dim=1)[0][:, None, None]
    max_v = torch.max(_heatmaps.view(_heatmaps.shape[0], -1), dim=1)[0][:, None, None]
    _heatmaps = (_heatmaps-min_v)/(max_v-min_v)
    return _heatmaps.clip(min=0, max=1)

if __name__ == '__main__':
    args = shjo.Parser(
        {
            'scales': [4],
            'image': './data/MoNuSeg/train/image/',

            'uss': './temp/MoNuSeg_MAE@ViT-B/train_{}@hflip/',
            'pred': './submissions/SSA@MoNuSeg@best/train_instance/',
            'mask': './submissions/Ours+SSA@mean+std/train/',

            'average_per_sample': False,
            'accumulate_centroids': False,
            'ot': False,
            'gamma': 1,
            'threshold': 0.4, # ablation study related to not using Optimal Transport
        }
    )

    if args.ot:
        ot_fn = refinements.OptimalTransport()
    
    mask_dir = shjo.makedir(args.mask)

    crf_fn = refinements.DenseCRF()

    IGNORE = 255
    colors = shjo.get_colors(ignore_index=IGNORE) # RGB color
    colors[0] = [0, 0, 0]                         # background
    colors[1] = [32, 167, 132]                    # foreground (e.g., cell)

    if args.accumulate_centroids:
        centroids = []
        
        for pred_path in shjo.progress(shjo.listdir(args.pred + '*.png'), 'Accumulation'):
            pred_path = pred_path.replace('\\', '/')
            mask_name = shjo.get_name(pred_path)
            image_id = mask_name.replace('.png', '')

            pred_mask = shjo.read_image(args.pred + image_id + '.png')
            pred_mask = pred_mask.astype(np.int64)
            pred_mask = pred_mask[:, :, 0] * 256 + pred_mask[:, :, 1]

            for scale in args.scales:
                with torch.no_grad():
                    f_uss = F.normalize(torch.load(args.uss.format(scale) + image_id + '.pt').cuda().float(), dim=0)
                    centroids_per_sample = f_uss[:, pred_mask > 0]
                    if args.average_per_sample:
                        centroids_per_sample = centroids_per_sample.mean(dim=1, keepdim=True)
                    centroids.append(centroids_per_sample)
        
        accumulated_centroid = torch.cat(centroids, dim=1).mean(dim=1)
    else:
        accumulated_centroid = None

    for pred_path in shjo.progress(shjo.listdir(args.pred + '*.png')):
        pred_path = pred_path.replace('\\', '/')
        mask_name = shjo.get_name(pred_path)
        image_id = mask_name.replace('.png', '')

        pred_mask = shjo.read_image(args.pred + image_id + '.png')
        pred_mask = pred_mask.astype(np.int64)
        pred_mask = pred_mask[:, :, 0] * 256 + pred_mask[:, :, 1]

        # USS propagation
        heatmaps = []

        for scale in args.scales:
            with torch.no_grad():
                f_uss = F.normalize(torch.load(args.uss.format(scale) + image_id + '.pt').cuda().float(), dim=0)

                centroids = torch.stack(
                    [
                        f_uss[:, pred_mask == 0].mean(dim=1),
                        f_uss[:, pred_mask > 0].mean(dim=1) if accumulated_centroid is None else accumulated_centroid
                    ]
                )
                heatmaps.append(F.cosine_similarity(centroids[:, :, None, None], f_uss[None, :, :, :], dim=1))

        heatmaps = torch.stack(heatmaps).mean(dim=0)

        # with Optimal Transport
        if args.ot:
            heatmaps = mask_normalize(heatmaps)
            heatmaps[0] = torch.maximum(heatmaps[0], 1. - heatmaps[1])

            c, ih, iw = heatmaps.shape
            T: torch.Tensor = ot_fn.apply(heatmaps.view(c, ih * iw).transpose(1, 0))

            heatmaps = heatmaps * T.transpose(1, 0).view(c, ih, iw)
            heatmaps = mask_normalize(heatmaps)
            heatmaps[0] = torch.maximum(heatmaps[0], 1. - heatmaps[1])
        
            # Refine CRF
            cv_image = shjo.read_image(shjo.listdir(args.image + image_id + '.*')[0])
            if cv_image.shape[-1] == 4:
                cv_image = cv_image[..., :3]

            heatmaps = heatmaps.cpu().numpy()
            if args.gamma > 1:
                heatmaps[1] = np.power(heatmaps[1], args.gamma)

            if args.gamma > 0:
                pseudo_mask = crf_fn(
                    cv_image[..., ::-1], 
                    heatmaps,
                )
            else:
                pseudo_mask = heatmaps
            
            pseudo_mask = np.argmax(pseudo_mask, axis=0)
        else:
            if args.threshold == -1:
                args.threshold = heatmaps[1].mean().item()
            # heatmaps[0] = 1. - heatmaps[1]
            heatmaps[0] = args.threshold

            if args.gamma > 0:
                # Refine CRF
                cv_image = shjo.read_image(shjo.listdir(args.image + image_id + '.*')[0])
                if cv_image.shape[-1] == 4:
                    cv_image = cv_image[..., :3]

                pseudo_mask = crf_fn(
                    cv_image[..., ::-1], 
                    heatmaps.cpu().numpy(),
                )
                pseudo_mask = np.argmax(pseudo_mask, axis=0)
            else:
                pseudo_mask = np.argmax(heatmaps.cpu().numpy(), axis=0)

        shjo.write_image(mask_dir + image_id + '.png', pseudo_mask, colors)

