import numpy as np

def get_IoU(pred_mask, gt_mask, eps=1e-5):
    mask1 = pred_mask > 0
    mask2 = gt_mask > 0

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(pred_mask, gt_mask).sum() + eps
    return intersection / union

def get_dice(pred_mask, gt_mask, eps=1e-5):
    mask1 = pred_mask > 0
    mask2 = gt_mask > 0
    intersection = np.logical_and(mask1, mask2)
    return (2. * intersection.sum() + eps) / (mask1.sum() + mask2.sum() + eps)

def get_AJI(pred_mask, gt_mask):
    def get_jaccard(_pred, _gt):
        return np.sum(np.logical_and(_pred, _gt)) / np.sum(np.logical_or(_pred, _gt))

    gt_indices = np.unique(gt_mask)[1:]      # remove a background class
    pred_indices = np.unique(pred_mask)[1:]  # remove a background class
    unused_mask = np.zeros(len(pred_indices)) # a false positive detection out of predicted instances

    intersection = 0
    union = 0

    for gt_index in gt_indices:
        gt = gt_mask == gt_index
        pred_indices_in_gt = pred_mask[gt]

        if np.sum(pred_indices_in_gt) == 0: # i.e., all background pixels
            union += np.sum(gt) # FN
        else:
            pred_indices_in_gt = np.unique(pred_indices_in_gt)

            matched_JI = 0
            matched_pred_index = -1

            for pred_index in pred_indices_in_gt:
                if pred_index == 0: continue # skip a background class

                JI_per_pred = get_jaccard(pred_mask == pred_index, gt)
                if JI_per_pred > matched_JI:
                    matched_JI = JI_per_pred
                    matched_pred_index = pred_index

            matched_pred_mask = pred_mask == matched_pred_index
            intersection += np.sum(np.logical_and(gt, matched_pred_mask)) # P ∩ T
            union += np.sum(np.logical_or(gt, matched_pred_mask))         # P ∪ T

            unused_mask[matched_pred_index-1] += 1
    
    FP = 0
    for pred_index in np.where(unused_mask==0)[0]: # FP
        FP += np.sum(pred_mask == pred_index)

    # refer to Eq. 2 in https://arxiv.org/pdf/2407.18673v1 
    return intersection / (union + FP)

class SemanticSegmentation:
    def __init__(self, class_names, ignore_index=255):
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.ignore_index = ignore_index
        self.meter_dict = self.set()

    def set(self):
        meter_dict = {}
        for tag in ['P', 'T', 'TP']:
            meter_dict[tag] = np.zeros(self.num_classes, dtype=np.float32)
        return meter_dict
    
    def add(self, pred_mask, gt_mask):
        obj_mask = gt_mask != self.ignore_index
        correct_mask = (pred_mask == gt_mask) * obj_mask

        for i in range(self.num_classes):
            self.meter_dict['P'][i] += np.sum((pred_mask==i)*obj_mask)
            self.meter_dict['T'][i] += np.sum((gt_mask==i)*obj_mask)
            self.meter_dict['TP'][i] += np.sum((gt_mask==i)*correct_mask)
    
    def get(self):
        IoU_list = []
        FPR_list = [] # over activation
        FNR_list = [] # under activation

        TP = self.meter_dict['TP']
        P = self.meter_dict['P']
        T = self.meter_dict['T']

        IoU_dict = {}
        
        for i in range(self.num_classes):
            union = (T[i] + P[i] - TP[i])
            if union <= 0:
                IoU_list.append(np.nan)
                FPR_list.append(np.nan)
                FNR_list.append(np.nan)
            else:
                IoU = float(TP[i] / union * 100)
                FPR = float((P[i] - TP[i]) / union)
                FNR = float((T[i] - TP[i]) / union)
                
                IoU_list.append(IoU)
                FPR_list.append(FPR)
                FNR_list.append(FNR)
                IoU_dict[self.class_names[i]] = {'IoU': IoU, 'FPR': FPR, 'FNR': FNR}
        
        mIoU = float(np.nanmean(IoU_list))
        mFPR = float(np.nanmean(FPR_list))
        mFNR = float(np.nanmean(FNR_list))
        IoU_dict['Average'] = {'mIoU': mIoU, 'mFPR': mFPR, 'mFNR': mFNR}
        
        self.meter_dict = self.set()
        
        return mIoU, mFPR, mFNR, IoU_dict
    
    def print(self, tag='', end='\n', ignore_bg=False):
        if len(tag) > 0: tag = f'# {tag:>30} | '
        
        mIoU, mFPR, mFNR, IoU_dict = self.get(int(ignore_bg))
        print(f'{tag}mIoU: {mIoU:.1f}%, mFPR: {mFPR:.3f}, mFNR: {mFNR:.3f}', end=end)
        
        return IoU_dict
