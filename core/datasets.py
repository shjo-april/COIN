import cv2
import numpy as np
import sanghyunjo as shjo

from PIL import Image

class ConventionalDataset:
    def __init__(self, root_dir: str, domain: str, dataset: str, transform=None, return_formats=[]):
        self.domain = domain
        self.transform = transform

        self.return_formats = return_formats
        self.return_dict = {
            'id': self.get_id,
            'image': self.get_image,
            'mask': self.get_mask,
        }
        
        self.image_dir = root_dir + f'{dataset}/{domain}/image/'
        self.mask_dir = root_dir + f'{dataset}/{domain}/mask/'

        self.image_names = shjo.listdir(self.image_dir)

    def __len__(self): return len(self.image_names)
    def get_id(self, image_name: str): 
        ext = image_name.split('.')[-1]
        return image_name.replace('.'+ext, '')
    def get_image(self, image_name: str): return Image.open(self.image_dir + image_name).convert('RGB')
    def get_mask(self, image_name: str): 
        if '/mask/' in self.mask_dir:
            ext = image_name.split('.')[-1]
            gt_mask = shjo.imread(self.mask_dir + image_name.replace('.'+ext, '.png'))

            gt_mask = gt_mask.astype(np.int64)
            gt_mask = gt_mask[:, :, 0] * 256 + gt_mask[:, :, 1]
            binary_mask = (gt_mask > 0).astype(np.float32)

            return Image.fromarray(binary_mask)
        else:
            ext = image_name.split('.')[-1]
            return Image.open(self.mask_dir + image_name.replace('.'+ext, '.png'))
    
    def __getitem__(self, i):
        image_name = self.image_names[i]
        output_dict = {fmt: self.return_dict[fmt](image_name) for fmt in self.return_formats}
        if self.transform is not None: output_dict = self.transform(output_dict)
        return output_dict

class SegmentationDataset(ConventionalDataset):
    def __init__(self, root_dir: str, domain: str, dataset: str='MoNuSeg', transform=None, mask_dir='', scales=1):
        super().__init__(root_dir, domain, dataset, transform, ['image', 'mask'])
        if len(mask_dir) > 0: self.mask_dir = mask_dir

        if scales > 1:
            total_image_names = []
            for _ in range(scales):
                total_image_names += self.image_names
            self.image_names = total_image_names
    
    def __getitem__(self, i):
        output_dict = super().__getitem__(i)
        return output_dict['image'], output_dict['mask']
    
class FalseNegativeSegmentationDataset(ConventionalDataset):
    def __init__(self, root_dir: str, domain: str, dataset: str='MoNuSeg', transform=None, mask_dir='', scales=1, image_ids=[]):
        super().__init__(root_dir, domain, dataset, transform, ['image', 'mask'])
        if len(mask_dir) > 0: self.mask_dir = mask_dir

        assert len(image_ids) > 0, "Please set image_ids to reduce FN."
        self.image_names = [image_id + '.tif' for image_id in image_ids]

        if scales > 1:
            total_image_names = []
            for _ in range(scales):
                total_image_names += self.image_names
            self.image_names = total_image_names

        self.ignore = len(mask_dir) > 0 # True: training, False, testing 
    
    def __getitem__(self, i):
        image_name = self.image_names[i]
        output_dict = {fmt: self.return_dict[fmt](image_name) for fmt in self.return_formats}

        if self.ignore:
            mask = np.asarray(output_dict['mask']).copy()
            # print(np.unique(mask)); input()

            mask[mask == 0] = 255 # replace 0 with 255
            output_dict['mask'] = Image.fromarray(mask)

        if self.transform is not None: output_dict = self.transform(output_dict)
        return output_dict['image'], output_dict['mask']

from .utils import instance2gradient

class InstanceSegmentationDataset(ConventionalDataset):
    def __init__(self, root_dir: str, domain: str, dataset: str='MoNuSeg', transform=None, mask_dir='', scales=1):
        super().__init__(root_dir, domain, dataset, transform, ['image', 'mask']) # 'id', 
        if len(mask_dir) > 0: self.mask_dir = mask_dir

        if scales > 1:
            total_image_names = []
            for _ in range(scales):
                total_image_names += self.image_names
            self.image_names = total_image_names
    
    def get_mask(self, image_name: str): 
        return Image.open(self.mask_dir + image_name.replace('.tif', '.png'))
        # return Image.fromarray(shjo.read_image(self.mask_dir + image_name.replace('.tif', '.png')))

    def __getitem__(self, i):
        image_name = self.image_names[i]
        
        output_dict = {fmt: self.return_dict[fmt](image_name) for fmt in self.return_formats}
        output_dict['ins_mask'] = output_dict['mask']; del output_dict['mask']

        # image_id = output_dict['id']
        # shjo.write_image(f'{image_id}.jpg', np.asarray(output_dict['ins_mask']).astype(np.uint8)); input()
        
        output_dict = self.transform(output_dict)

        # print(output_dict['image'].shape, output_dict['ins_mask'].shape)

        pred_ins_mask = shjo.resize(output_dict['ins_mask'], scale=0.25, mode='nearest').astype(np.uint32)
        # pred_ins_mask = output_dict['ins_mask'].astype(np.uint32)
        
        ignore_mask = pred_ins_mask.sum(axis=2) == (255*3)
        pred_ins_mask = pred_ins_mask[:, :, 0] * 256 + pred_ins_mask[:, :, 1]

        pred_ins_mask[ignore_mask] = 0 
        semantic_mask, hv_mask = instance2gradient(pred_ins_mask)
        semantic_mask[ignore_mask] = 255

        # print(semantic_mask.shape, np.unique(semantic_mask), hv_map.shape, hv_map.min(), hv_map.max())
        # input()

        return output_dict['image'], semantic_mask, hv_mask
    
class EdgeSegmentationDataset(ConventionalDataset):
    def __init__(self, root_dir: str, domain: str, dataset: str='MoNuSeg', transform=None, mask_dir='', scales=1):
        super().__init__(root_dir, domain, dataset, transform, ['image', 'mask']) # 'id', 
        if len(mask_dir) > 0: self.mask_dir = mask_dir

        if scales > 1:
            total_image_names = []
            for _ in range(scales):
                total_image_names += self.image_names
            self.image_names = total_image_names
    
    def get_mask(self, image_name: str): 
        ext = image_name.split('.')[-1]
        return Image.open(self.mask_dir + image_name.replace('.'+ext, '.png'))

    def __getitem__(self, i):
        image_name = self.image_names[i]
        
        output_dict = {fmt: self.return_dict[fmt](image_name) for fmt in self.return_formats}
        output_dict['ins_mask'] = output_dict['mask']; del output_dict['mask']

        output_dict = self.transform(output_dict)

        pred_ins_mask = output_dict['ins_mask'].astype(np.uint32)
        
        ignore_mask = pred_ins_mask.sum(axis=2) == (255*3)
        pred_ins_mask = pred_ins_mask[:, :, 0] * 256 + pred_ins_mask[:, :, 1]

        pred_ins_mask[ignore_mask] = 0 

        edge = np.zeros_like(pred_ins_mask, dtype=np.uint8)
        for index in np.unique(pred_ins_mask):
            if index == 0: continue
            canny = cv2.Canny((pred_ins_mask == index).astype(np.uint8)*255, 100, 200)
            edge = np.maximum(edge, (canny > 0).astype(np.uint8))
        edge[ignore_mask] = 255
        
        semantic_mask = (pred_ins_mask > 0).astype(np.uint8)
        semantic_mask[ignore_mask] = 255

        # print(semantic_mask.shape, np.unique(semantic_mask), hv_map.shape, hv_map.min(), hv_map.max())
        # input()

        return output_dict['image'], semantic_mask, edge

class EvalDataset(ConventionalDataset):
    def __init__(self, root_dir: str, domain: str, dataset: str='MoNuSeg', transform=None):
        super().__init__(root_dir, domain, dataset, transform, ['id', 'image'])
    
    def __getitem__(self, i):
        output_dict = super().__getitem__(i)
        return output_dict['id'], output_dict['image']