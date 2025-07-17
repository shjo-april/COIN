# Copyright (C) 2024 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import torch
import numpy as np
import sanghyunjo as shjo

from torchvision import transforms as T
from torch.nn import functional as F

from core import uss

args = shjo.Parser(
    {
        'root': './data/', 'data': 'MoNuSeg', 'domain': 'train', 'zoom': [4], 
        'uss': 'DINOv2-reg', 'backbone': 'ViT-S', 
        'scales': [1.0], 'hflip': False, 'stride': 2/3, 'reduce_vram': False,
        'image_scale': -1, # for OOM or low-resolution input images
        'turn_off_pca': False,
    }
)

OVERLAP = 1.00

with torch.no_grad():
    if args.uss == 'DINOv1':
        model = uss.DINOv1(args.backbone, patch_size=8)
    else:
        model = uss.FiT3D(args.uss, args.backbone, fit3d=True)

    image_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(model.mean, model.std)
        ]
    )

    tag = f'{args.zoom[0]}'
    if len(args.scales) > 1 or args.scales[0] != 1.0: tag += f'@{args.scales}'
    if args.hflip: tag += '@hflip'
    
    output_dir = shjo.makedir(f'./temp/{args.data}_{args.uss}@{args.backbone}/{args.domain}_{tag}/')
    
    for image_path in shjo.progress(shjo.listdir(args.root + args.data + '/' + args.domain + '/image/*')):
        cv_image = shjo.read_image(image_path)
        ih, iw = cv_image.shape[:2]

        image_name, image_ext = shjo.get_name(image_path, ext=True)
        pt_path = output_dir + image_name.replace('.'+image_ext, f'.pt')
        pca_path = output_dir + image_name.replace('.'+image_ext, f'.png')

        if not shjo.isfile(pt_path):
            global_uss = torch.zeros((model.embed_dim, ih, iw)).cuda().float()
            global_cnt = torch.zeros((ih, iw)).cuda().float()

            for zoom in args.zoom:
                zwh = min(iw // zoom, ih // zoom)
                visit = np.zeros((ih, iw), dtype=np.uint8)

                ymin = 0
                while ymin < ih:
                    xmin = 0
                    while xmin < iw:
                        xmax = min(xmin + zwh, iw)
                        xmin = min(xmin, xmax - zwh)
                        ymax = min(ymin + zwh, ih)
                        ymin = min(ymin, ymax - zwh)
                        if visit[ymin:ymax, xmin:xmax].mean() >= OVERLAP: break
                        
                        visit[ymin:ymax, xmin:xmax] = 1.0

                        input_image = shjo.resize(cv_image[ymin:ymax, xmin:xmax], scale=zoom if args.image_scale == -1 else (args.image_scale * zoom))
                        input_image = image_transform(shjo.cv2pil(input_image).convert('RGB'))
                        
                        f_uss = uss.inference(
                            model, 
                            input_image, 
                            args.scales, 
                            args.hflip,
                            reduce_vram=args.reduce_vram
                        )
                        f_uss = F.interpolate(f_uss[None], (ymax-ymin, xmax-xmin), mode='bilinear')[0]

                        # print(global_uss.shape, f_uss.shape, (xmin, ymin, xmax, ymax))
                        
                        global_uss[:, ymin:ymax, xmin:xmax] += f_uss
                        global_cnt[ymin:ymax, xmin:xmax] += 1

                        xmin += int(zwh * args.stride)
                    ymin += int(zwh * args.stride)

                    # check the last patch
                    xmax = min(xmin + zwh, iw)
                    xmin = min(xmin, xmax - zwh)
                    ymax = min(ymin + zwh, ih)
                    ymin = min(ymin, ymax - zwh)

                    if visit[ymin:ymax, xmin:xmax].mean() >= OVERLAP: break
            
            f_uss = (global_uss / global_cnt)

            if not args.turn_off_pca:
                from sklearn.decomposition import PCA
                fd, fh, fw = f_uss.shape
                pca = PCA(n_components=3, random_state=0).fit_transform(f_uss.view(fd, fh * fw).permute(1, 0).cpu().numpy())
                
                pca = pca.reshape(fh, fw, 3)
                min_v, max_v = pca.min(axis=(0, 1)), pca.max(axis=(0, 1))
                pca = (pca - min_v) / (max_v - min_v)
                pca = (pca * 255).astype(np.uint8)
                pca = shjo.resize(pca, (ih, iw), mode='nearest')
                shjo.write_image(pca_path, pca)

            torch.save(f_uss.half(), pt_path)