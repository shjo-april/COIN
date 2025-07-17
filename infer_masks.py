# Copyright (C) 2024 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import os
import ray
import copy
import torch
import numpy as np
import sanghyunjo as shjo

from tqdm import tqdm
from scipy import ndimage
from joblib import Parallel, delayed

from core import networks, datasets, refinements
from tools import torch_utils, transforms

def mask_normalize(_heatmaps):
    min_v = torch.min(_heatmaps.view(_heatmaps.shape[0], -1), dim=1)[0][:, None, None]
    max_v = torch.max(_heatmaps.view(_heatmaps.shape[0], -1), dim=1)[0][:, None, None]
    _heatmaps = (_heatmaps-min_v)/(max_v-min_v)
    return _heatmaps.clip(min=0, max=1)

@delayed
def inference(model, test_dataset, indices, device, args, visualize):
    model = model.to(device)
    if visualize: pbar = tqdm(total=len(indices), desc=f'Inference ({device})')

    if args.ot:
        ot_fn = refinements.OptimalTransport()

    for index in indices:
        if visualize: pbar.update(1)

        image_id, image = test_dataset[index]

        if os.path.isfile(args.temp + image_id + '.pt') or os.path.isfile(args.temp + image_id + '.png'): continue
        
        # preprocessing
        image = torch.from_numpy(image).to(device).unsqueeze(0)
        
        # inference
        pred_mask, _ = model.apply_ms(image)
        if args.sliding:
            c, ih, iw = pred_mask.shape
            pred_masks = [pred_mask]
            
            for zoom in [2]:
                zwh = min(iw // zoom, ih // zoom)
                visit = np.zeros((ih, iw), dtype=np.uint8)

                pred_count = torch.zeros((1, ih, iw)).float().to(device)
                pred_mask_per_zoom = torch.zeros_like(pred_mask)

                ymin = 0
                while ymin < ih:
                    xmin = 0
                    while xmin < iw:
                        xmax = min(xmin + zwh, iw)
                        xmin = min(xmin, xmax - zwh)
                        ymax = min(ymin + zwh, ih)
                        ymin = min(ymin, ymax - zwh)
                        if visit[ymin:ymax, xmin:xmax].mean() >= 1.0: break
                        
                        visit[ymin:ymax, xmin:xmax] = 1.0
                        pred_count[:, ymin:ymax, xmin:xmax] += 1.0
                        pred_mask_per_zoom[:, ymin:ymax, xmin:xmax] += model.apply_ms(image[:, :, ymin:ymax, xmin:xmax])

                        xmin += int(zwh * args.stride)
                    ymin += int(zwh * args.stride)

                    # check the last patch
                    xmax = min(xmin + zwh, iw)
                    xmin = min(xmin, xmax - zwh)
                    ymax = min(ymin + zwh, ih)
                    ymin = min(ymin, ymax - zwh)

                    if visit[ymin:ymax, xmin:xmax].mean() >= 1.0: break

                pred_mask_per_zoom /= pred_count
                pred_masks.append(pred_mask_per_zoom)

            pred_mask = torch.stack(pred_masks).mean(dim=0)
        
        pred_mask = mask_normalize(pred_mask)

        if args.ot:
            c, ih, iw = pred_mask.shape
            T: torch.Tensor = ot_fn.apply(pred_mask.view(c, ih * iw).transpose(1, 0))

            pred_mask = pred_mask * T.transpose(1, 0).view(c, ih, iw)
            pred_mask = mask_normalize(pred_mask)
        
        pred_mask = torch_utils.get_numpy(pred_mask.half())

        # save a torch tensor
        torch.save(pred_mask, args.temp + image_id + '.pt')

@ray.remote
def apply_crf(image_id, image, temp, colors, denorm_fn, crf_fn, args):
    pseudo_path = temp + image_id + '.png'
    if os.path.isfile(pseudo_path):
        return
    
    if os.path.isfile(temp + f'{image_id}.pt'):
        pred_mask = crf_fn(
            denorm_fn(image).copy(), 
            torch.load(temp + f'{image_id}.pt').astype(np.float32)
        )

        pred_mask = np.argmax(pred_mask, axis=0)
        pred_mask = ndimage.binary_fill_holes(pred_mask>0).astype(np.uint8)

        shjo.write_image(pseudo_path, pred_mask, colors)
        os.remove(temp + f'{image_id}.pt')

def main(args):
    # set gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # build model
    if args.decoder == 'deeplabv3+': model = networks.DeepLabv3plus_Edge(args.backbone, 2).cpu().eval()
    else: raise NotImplementedError(f'ERROR: {args.decoder}')

    torch_utils.load_model(model, f'./experiments/models/{args.tag}/{args.checkpoint}.pth')
    
    # create datasets
    test_transform = transforms.Compose([transforms.Normalize()])
    test_dataset = datasets.EvalDataset(args.root, args.domain, args.data, test_transform)

    # inference
    args.temp = shjo.makedir(args.pred)

    length, num_gpus = len(test_dataset), len(args.gpus.split(','))
    length_per_gpu = length // num_gpus

    params = []
    indices = np.arange(length)
    
    for gpu_index in range(num_gpus-1):
        param = [copy.deepcopy(model), test_dataset, indices[:length_per_gpu], torch.device('cuda', gpu_index), args, False]
        params.append(param); indices = indices[length_per_gpu:]
    
    param = [copy.deepcopy(model), test_dataset, indices, torch.device('cuda', num_gpus-1), args, True]
    params.append(param)

    Parallel(n_jobs=len(params))([inference(*param) for param in params])
    torch.cuda.empty_cache()

    # CRF
    denorm_fn = transforms.Denormalize()
    crf_fn = refinements.DenseCRF()

    params = []
    colors = shjo.get_colors() # RGB color
    colors[0] = [0, 0, 0]      # background
    colors[1] = [32, 167, 132] # foreground (e.g., cell)

    ray.init(num_cpus=args.cpus)

    for image_id, image in tqdm(test_dataset):
        if os.path.isfile(args.temp + image_id + '.png'): continue
        params.append(apply_crf.remote(image_id, image, args.temp, colors, denorm_fn, crf_fn, args))
    
    ray.get(params)

if __name__ == '__main__':
    main(
        shjo.Parser(
            {
                'cpus': 32, 'gpus': '0', 
                'data': 'MoNuSeg', 'domain': 'train', 'root': './data/',
                'backbone': 'resnet101v2', 'decoder': 'deeplabv3+', 
                'tag': 'ResNet-101-v2@VOC2012@MARS', 'checkpoint': 'last',

                'pred': './submissions/SSA@MoNuSeg@best/train_instance/',
                'ot': False, 'sliding': False, 'stride': 2/3,
            }
        )
    )