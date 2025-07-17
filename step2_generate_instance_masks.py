import cv2
import numpy as np
import sanghyunjo as shjo

from core import refinements

args = shjo.Parser(
    {
        'root': './data/', 'data': 'MoNuSeg', 'domain': 'test', 
        'tag': 'PSM@MoNuSeg@best', 'sem_domain': 'test', 'denoise': False,
        'exp': './submissions/',
    }
)

image_dir = args.root + f'{args.data}/{args.domain}/image/'
gt_mask_dir = args.root + f'{args.data}/{args.domain}/mask/'
pred_sem_dir = args.exp + f'{args.tag}/{args.sem_domain}/'

print(args.domain)

ins_tag = 'instance'
if args.denoise: ins_tag += '@denoise'
pred_ins_dir = shjo.makedir(args.exp + f'{args.tag}/{args.sem_domain}_{ins_tag}/')

for image_name in shjo.progress(shjo.listdir(image_dir), args.tag):
    pred_sem_mask = shjo.imread(pred_sem_dir + image_name.replace('.tif', '.png'), backend='mask')

    # print(pred_sem_dir + image_name.replace('.tif', '.png'))
    # print(pred_sem_mask.shape, np.unique(pred_sem_mask)); input()

    pred_ins_mask = refinements.semantic2instance(pred_sem_mask, denoise=args.denoise)
    
    ih, iw = pred_ins_mask.shape
    pred_mask = np.zeros((ih, iw, 3), dtype=np.uint8)

    for index in np.unique(pred_ins_mask):
        if index == 0: continue # ignore background
        pred_mask[pred_ins_mask == index, 0] = index // 256 # B channel
        pred_mask[pred_ins_mask == index, 1] = index  % 256 # G channel
    
    shjo.write_image(pred_ins_dir + image_name.replace('.tif', '.png'), pred_mask)