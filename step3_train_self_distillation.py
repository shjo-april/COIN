# Copyright (C) 2024 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import os
import torch
import shutil

import numpy as np
import sanghyunjo as shjo

from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

from core import networks, datasets, losses
from tools import torch_utils, evaluators, trainers, optimizers, transforms as T

def collate(batch):
    images = []
    masks = []
    edges = []
    
    for image, mask, edge in batch:
        images.append(torch.from_numpy(image))
        masks.append(torch.from_numpy(mask))
        edges.append(torch.from_numpy(edge))
    
    return {
        'images': torch.stack(images),
        'masks': torch.stack(masks),
        'edges': torch.stack(edges),
    }

def main(args):
    # set gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    gpus = args.gpus.split(',')
    device = torch.device('cuda')

    # set directories
    model_dir = f'./experiments/models/{args.tag}/'
    tensorboard_dir = f'./experiments/tensorboards/{args.tag}/'

    txt_path = model_dir + f'{args.tag}.txt'
    if os.path.isfile(txt_path):
        if input('Found existing logs. yes=remove, no=keyboardinterept') == 'no': raise KeyboardInterrupt
        else:
            if os.path.isdir(model_dir): shutil.rmtree(model_dir)
            if os.path.isdir(tensorboard_dir): shutil.rmtree(tensorboard_dir)
    
    def log(string, path=None):
        print(string)
        if path is not None: 
            open(path, 'a+').write(string+'\n')
        
    log_fn = lambda string='': log(string, txt_path)
    model_dir = shjo.makedir(model_dir)
    tensorboard_dir = shjo.makedir(tensorboard_dir)
    
    # create model
    if args.decoder == 'deeplabv3+': model = networks.DeepLabv3plus_Edge(args.backbone, 2).to(device)
    else: raise NotImplementedError(f'ERROR: {args.decoder}')

    num_params = torch_utils.calculate_parameters(model.parameters())
    log_fn(f'[i] Backbone: {args.backbone} ({num_params:.1f}MB)\n')

    if len(gpus) > 1:
        model = nn.DataParallel(model)

    # define loss functions
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=255).to(device)
    # dice_loss_fn = losses.DiceLoss(ignore_index=255).to(device)
    dice_loss_fn = losses.ForegroundDiceLoss(ignore_index=255).to(device)
    
    # define trainer 
    class Trainer(trainers.BaseTrainer):
        def __init__(self):
            param = trainers.Parameter(
                args.seed, True, args.ema,
                args.epochs, tensorboard_dir, -1
            )
            super().__init__(model, device, param)

            self.best_mIoU_val = 0
            self.best_mIoU_train = 0
        
        def prepare_dataset(self):
            train_transform = T.get_transform(args.train_transform, args)
            test_transform = T.get_transform(args.test_transform, args)

            log_fn(f'Training augmentation: {train_transform}')
            log_fn(f'Testing augmentation: {test_transform}')
            
            if ',' in args.train:
                self.train_dataset = ConcatDataset(
                    [
                        datasets.EdgeSegmentationDataset(args.root, domain, args.data, train_transform, args.mask, args.scales)
                        for domain in args.train.split(',')
                    ]
                )
            else:
                self.train_dataset = datasets.EdgeSegmentationDataset(args.root, args.train, args.data, train_transform, args.mask, args.scales)
            
            self.valid_dataset = datasets.SegmentationDataset(args.root, args.valid, args.data, test_transform, scales=1)
            self.train_dataset_for_eval = datasets.SegmentationDataset(args.root, args.train_eval, args.data, test_transform, scales=1)

        def prepare_loader(self, is_print=True):
            shuffle = True
            train_sampler = None
            
            self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch, num_workers=args.cpus, shuffle=shuffle, drop_last=True, pin_memory=True, sampler=train_sampler, collate_fn=collate)
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=1, num_workers=max(args.cpus // 4, 1), shuffle=False, drop_last=False, pin_memory=True)
            self.train_loader_for_eval = DataLoader(self.train_dataset_for_eval, batch_size=1, num_workers=max(args.cpus // 4, 1), shuffle=False, drop_last=False, pin_memory=True)

            if is_print:
                log_fn('The size of training set: {}'.format(len(self.train_dataset)))
                log_fn('The size of training set for evaluation: {}'.format(len(self.train_loader_for_eval)))
                log_fn('The size of validation set: {}'.format(len(self.valid_dataset)))

        def reload_loader(self):
            del self.train_loader
            del self.valid_loader
            del self.train_loader_for_eval

            self.prepare_loader(is_print=False)
            
            self.train_iterations = len(self.train_loader)
            self.valid_iterations = len(self.valid_loader)
            self.max_iterations = len(self.train_loader) * self.param.max_epochs
        
        def configure_optimizers(self):
            self.optimizer = optimizers.SGD(
                params=[
                    {'params': self.param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
                    {'params': self.param_groups[1], 'lr': args.lr, 'weight_decay': args.wd},
                    {'params': self.param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
                    {'params': self.param_groups[3], 'lr': 10*args.lr, 'weight_decay': args.wd},
                ],
                lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov,
                scheduler_option={
                    'scheduler': args.scheduler,
                    'power': 0.9,
                    'max_iterations': self.train_iterations * self.param.max_epochs
                }
            )
        
        def forward(self, data, training: bool=True):
            if training:
                images = data['images'].to(self.device)
                masks = data['masks'].to(self.device).long()
                edges = data['edges'].to(self.device).long()

                logits, logits_edge = self.model(images)

                lr_masks = torch_utils.resize(masks.float(), logits.shape[2:], mode='nearest').long()
                seg_ce_loss = ce_loss_fn(logits, lr_masks)
                seg_dice_loss = dice_loss_fn(logits, lr_masks)

                edge_ce_loss = ce_loss_fn(logits_edge, edges)
                edge_dice_loss = dice_loss_fn(logits_edge, edges)

                loss = seg_ce_loss + seg_dice_loss + edge_ce_loss + edge_dice_loss
                
                return loss, {
                    'LR': self.get_learning_rate(), 'Loss': loss.item(), 
                    'CE_s': seg_ce_loss.item(), 'Dice_s': seg_dice_loss.item(), 
                    'CE_e': edge_ce_loss.item(), 'Dice_e': edge_dice_loss.item(), 
                }
            else:
                images, masks = data

                if self.ema is not None: model = self.ema.get()
                else: model = torch_utils.de_parallel(self.model)

                images = images.to(self.device)
                masks = masks.to(self.device)

                pred_mask, _ = model.apply_ms(images)

                return {
                    'pred_mask': np.argmax(torch_utils.get_numpy(pred_mask), axis=0), 
                    'gt_mask': torch_utils.get_numpy(masks[0])
                }

        def evaluation_step(self, debug=False):
            if self.ema is None: self.model.eval()
            
            self.evaluator = evaluators.SemanticSegmentation(['background', 'foreground'])

            valid_time = super().evaluation_step(debug, self.valid_loader)
            mIoU_val, mFPR_val, mFNR_val, _ = self.evaluator.get()

            train_time = super().evaluation_step(debug, self.train_loader_for_eval)
            mIoU_train, mFPR_train, mFNR_train, _ = self.evaluator.get()
            
            if self.ema is None: self.model.train()

            if self.best_mIoU_val < mIoU_val:
                self.best_mIoU_val = mIoU_val
                self.save_model(model_dir + 'best_val.pth')

            if self.best_mIoU_train < mIoU_train:
                self.best_mIoU_train = mIoU_train
                self.save_model(model_dir + 'best_train.pth')
            
            tb_dict = self.update_tensorboard(
                {
                    'best_mIoU_val': self.best_mIoU_val, 
                    'mIoU_val': mIoU_val, 
                    'mFPR_val': mFPR_val, 
                    'mFNR_val': mFNR_val, 

                    'best_mIoU_train': self.best_mIoU_train, 
                    'mIoU_train': mIoU_train, 
                    'mFPR_train': mFPR_train, 
                    'mFNR_train': mFNR_train,
                }
            )
            tb_dict['epoch'] = self.epoch - 1
            tb_dict['time'] = valid_time + train_time
            return tb_dict
    
    trainer = Trainer()
    for epoch in range(trainer.epoch, args.epochs+1):
        train_dict = trainer.training_step()
        log_fn('Epoch: {epoch:,}, LR: {LR:.4f}, Loss: {Loss:.3f}, CE_s: {CE_s:.3f}, Dice_s: {Dice_s:.3f}, CE_e: {CE_e:.3f}, Dice_e: {Dice_e:.3f},  {time:.0f}s'.format(**train_dict))
        trainer.save_model(model_dir + 'last.pth')

        if epoch % args.eval == 0:
            valid_dict = trainer.evaluation_step()
            log_fn('Epoch: {epoch:,}, mIoU_val: {mIoU_val:.1f}% ({best_mIoU_val:.1f}%), mIoU_train: {mIoU_train:.1f}% ({best_mIoU_train:.1f}%), mFPR_train: {mFPR_train:.3f}, mFNR_train: {mFNR_train:.3f}, {time:.0f}s'.format(**valid_dict))

        if epoch % 10 == 0:    
            trainer.save_model(model_dir + f'{epoch:03d}.pth')

if __name__ == '__main__':
    main(
        shjo.Parser(
            {
                'local_rank': -1, 'gpus': '0', 'cpus': 16, 'seed': 1,
                'root': './data/', 'data': 'MoNuSeg', 'train': 'train', 'valid': 'test', 'train_eval': 'train', 
                'backbone': 'resnet101v2', 'decoder': 'deeplabv3+', 'mask': '', 'tag': 'ResNet-101', 'scales': 100,
                'image': 512, 'batch': 16, 'epochs': 100, 'eval': 5, 'lambda_dice': 1.0, 
                'lr': 1e-3, 'wd': 4e-5, 'optimizer': 'SGD', 'momentum': 0.9, 'nesterov': False, 'skip_train': False, 'scheduler': 'PolyLR', 'ema': 0.999, 
                'min_scale': 0.5, 'max_scale': 2.0, 'b_factor': 0.5, 'c_factor': 0.5, 's_factor': 0.5, 'h_factor': 0.3,
                'train_transform': 'RandomRescale,RandomVFlip,RandomHFlip,ColorJitter,Normalize,RandomCrop', 
                'test_transform': 'Normalize',
            }
        )
    )