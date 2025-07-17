# Copyright (C) 2024 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import copy
import numpy as np
import sanghyunjo as shjo

from tqdm import tqdm

class Evaluator:
    def __init__(self, class_names, ignore_index=100_000_000_000_000):
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.ignore_index = ignore_index
        self.clear()
    
    def clear(self):
        self.meter_dict = {'AJI': [], 'PQ': [], 'Dice': []}
        for tag in [
                "IoU", "FP", "FN", 
                "F1", "Precision", "Recall"
            ]:
            self.meter_dict[tag] = np.zeros(self.num_classes, dtype=np.float32)
    
    def add(self, pred_mask, gt_mask):
        try:
            AJI = self.get_AJI(pred_mask, gt_mask)
            DQ, SQ, PQ = self.get_PQ(pred_mask, gt_mask)
        except IndexError: # empty instances in pred_mask
            AJI = 0. 
            PQ = 0. 

        self.meter_dict['AJI'].append(AJI)
        self.meter_dict['PQ'].append(PQ)

        Dice = self.get_dice(pred_mask > 0, gt_mask > 0)
        self.meter_dict['Dice'].append(Dice)
        
        matrix = self.calculate_confusion_matrix(pred_mask, gt_mask)
        for i in range(self.num_classes):
            union = max(matrix["T"][i] + matrix["P"][i] - matrix["TP"][i], 1e-5)
            self.meter_dict["IoU"][i] += (matrix["TP"][i] / union)
            self.meter_dict["FP"][i] += ((matrix["P"][i] - matrix["TP"][i]) / union)
            self.meter_dict["FN"][i] += ((matrix["T"][i] - matrix["TP"][i]) / union)

            precision = matrix["TP"][i] / max(matrix["P"][i], 1e-5)
            recall = matrix["TP"][i] / max(matrix["T"][i], 1e-5)
            self.meter_dict["F1"][i] += ((2 * precision * recall) / max(precision + recall, 1e-5))
            self.meter_dict["Precision"][i] += precision
            self.meter_dict["Recall"][i] += recall
        
        return float(AJI), float(Dice)
            
    def get(self):
        length = len(self.meter_dict['AJI'])

        AJI = np.mean(self.meter_dict['AJI'])
        PQ = np.mean(self.meter_dict['PQ'])
        Dice = np.mean(self.meter_dict['Dice'])

        mIoU = (self.meter_dict['IoU'] / length).mean()
        mFP = (self.meter_dict['FP'] / length).mean()
        mFN = (self.meter_dict['FN'] / length).mean()

        IoU = float((self.meter_dict['IoU'][1] / length))
        FP = float((self.meter_dict['FP'][1] / length))
        FN = float((self.meter_dict['FN'][1] / length))

        precision = float(self.meter_dict['Precision'][1] / length)
        recall = float(self.meter_dict['Recall'][1] / length)
        F1 = float(self.meter_dict['F1'][1] / length)

        return float(AJI), float(PQ), float(Dice), float(mIoU), float(mFP), float(mFN), IoU, FP, FN, precision, recall, F1

    def calculate_confusion_matrix(self, pred_mask, gt_mask):
        target_mask = gt_mask != self.ignore_index

        pred_mask = (pred_mask > 0).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)

        correct_mask = (pred_mask == gt_mask) * target_mask

        matrix = {tag: np.zeros(self.num_classes, dtype=np.float32) for tag in ["P", "T", "TP"]}
        for i in range(self.num_classes):
            matrix["P"][i] += np.sum((pred_mask == i) * target_mask)
            matrix["T"][i] += np.sum((gt_mask == i) * target_mask)
            matrix["TP"][i] += np.sum((gt_mask == i) * correct_mask)

        return matrix
    
    def get_dice(self, pred, gt, smooth=1.):
        intersection = np.sum(pred * gt)
        union = np.sum(pred) + np.sum(gt)
        return 2.0 * (intersection + smooth) / (union + smooth)

    def get_jaccard(self, _pred, _gt):
        return np.sum(np.logical_and(_pred, _gt)) / np.sum(np.logical_or(_pred, _gt))

    def get_AJI(self, pred_mask, gt_mask):
        gt_indices = sorted(np.unique(gt_mask))[1:]      # remove a background class
        pred_indices = sorted(np.unique(pred_mask))[1:]  # remove a background class

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
                    if pred_index == 0: continue

                    JI_per_pred = self.get_jaccard(pred_mask == pred_index, gt)
                    if JI_per_pred > matched_JI:
                        matched_JI = JI_per_pred
                        matched_pred_index = pred_index

                matched_pred_mask = pred_mask == matched_pred_index
                intersection += np.sum(np.logical_and(gt, matched_pred_mask)) # P ∩ T
                union += np.sum(np.logical_or(gt, matched_pred_mask))         # P ∪ T

                unused_mask[int(matched_pred_index)-1] += 1
        
        FP = 0
        for pred_index in np.where(unused_mask==0)[0]: # FP
            FP += np.sum(pred_mask == (pred_index + 1))

        # refer to Eq. 2 in https://arxiv.org/pdf/2407.18673v1 
        return intersection / (union + FP)

    def get_PQ(self, pred_mask, gt_mask, iou_th=0.5):
        true_id_list = np.unique(gt_mask).tolist()
        pred_id_list = np.unique(pred_mask).tolist()

        true_masks = [None]
        refined_true_id_list = []
        for i, t in enumerate(true_id_list[1:]):
            true_masks.append((gt_mask == t).astype(np.uint8))
            refined_true_id_list.append(i)

        pred_masks = [None]
        refined_pred_id_list = []
        for i, p in enumerate(pred_id_list[1:]):
            pred_masks.append((pred_mask == p).astype(np.uint8))
            refined_pred_id_list.append(i)

        # address a mismatched example between the maximum of indices and length, e.g., [0, 1, 3, 4] to [0, 1, 2, 3]
        true_id_list = refined_true_id_list
        refined_pred_id_list = refined_pred_id_list

        pairwise_iou = np.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)
        
        ### Hungarian Algorithm
        for true_id in true_id_list[1:]: # 0-th is background
            t_mask = true_masks[true_id]
            pred_true_overlap = pred_mask[t_mask > 0]
            pred_true_overlap_id = list(np.unique(pred_true_overlap))
            for pred_id in pred_true_overlap_id:
                if pred_id == 0: continue # background
                p_mask = pred_masks[pred_id]
                total = (t_mask + p_mask).sum()
                inter = (t_mask * p_mask).sum()
                iou = inter / (total - inter)
                pairwise_iou[true_id - 1, pred_id - 1] = iou
        
        paired_iou = pairwise_iou[pairwise_iou > iou_th]
        pairwise_iou[pairwise_iou <= iou_th] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original

        unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
        unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]

        tp = len(paired_true)
        fp = len(unpaired_pred)
        fn = len(unpaired_true)

        # get the F1-score i.e DQ
        dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1.0e-6)
        sq = paired_iou.sum() / (tp + 1.0e-6)

        return dq, sq, dq * sq 

def main(args):
    evaluator = Evaluator(['background', 'foreground'])

    gt_dir = args.root + args.data + '/' + args.domain + '/mask/'
    pred_dir = './submissions/' + args.tag + f'/{args.domain}_instance/'

    data_dict = {
        'Average': {},
        'Samples': {}
    }

    for image_name in tqdm(shjo.listdir(gt_dir)):
        gt_mask = shjo.imread(gt_dir + image_name).astype(np.uint32)
        gt_mask = gt_mask[:, :, 0] * 256 + gt_mask[:, :, 1]
        
        if not args.UIS:
            pred_mask = shjo.imread(pred_dir + image_name).astype(np.uint32)
            pred_mask = pred_mask[:, :, 0] * 256 + pred_mask[:, :, 1]
        else:
            if shjo.isfile(pred_dir + image_name):
                pred_mask = shjo.imread(pred_dir + image_name, backend='mask')
                pred_mask = pred_mask.copy().astype(np.uint32)
            else:
                pred_mask = np.zeros_like(gt_mask).astype(np.uint32)

        if args.ordering:
            refined_gt_mask = np.zeros_like(gt_mask)
            for instance_id, gt_id in enumerate(np.unique(gt_mask)):
                refined_gt_mask[gt_mask == gt_id] = instance_id

            refined_pred_mask = np.zeros_like(pred_mask)
            for instance_id, pred_id in enumerate(np.unique(pred_mask)):
                refined_pred_mask[pred_mask == pred_id] = instance_id

            gt_mask, pred_mask = refined_gt_mask, refined_pred_mask
        
        AJI, Dice = evaluator.add(pred_mask, gt_mask)

        data_dict['Samples'][image_name] = {'AJI': AJI, 'Dice': Dice}
    
    AJI, PQ, Dice, mIoU, mFP, mFN, IoU, FP, FN, precision, recall, F1 = evaluator.get()
    
    data_dict['Average']['AJI'] = AJI
    data_dict['Average']['PQ'] = PQ
    data_dict['Average']['Dice'] = Dice

    data_dict['Average']['mIoU'] = mIoU
    data_dict['Average']['mFN'] = mFN
    data_dict['Average']['mFP'] = mFP

    data_dict['Average']['Precision'] = precision
    data_dict['Average']['Recall'] = recall
    data_dict['Average']['F1'] = F1

    shjo.jswrite('./submissions/' + args.tag + '@' + args.domain + '.json', data_dict)

    print(f'# [{args.tag:30s}] AJI: {AJI:.3f}, PQ: {PQ:.3f}, Dice: {Dice:.3f}, mIoU: {mIoU:.3f}, mFN: {mFN:.3f}, mFP: {mFP:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {F1:.3f}')
    if args.detail: 
        print(f'IoU: {IoU:.3f}, FN: {FN:.3f}, FP: {FP:.3f}')

if __name__ == '__main__':
    args = shjo.Parser(
        {
            'root': './data/', 'data': 'MoNuSeg', 'domain': 'test', 
            'tag': 'Ours+SSA@R2', 'ordering': False, 'detail': False,
            'UIS': False,
        }
    )
    main(args)