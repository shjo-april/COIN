import torch
from torch import nn
from torch.nn import functional as F

# class DiceLoss(nn.Module):
#     def __init__(self, ignore_index=255):
#         super().__init__()
#         self.ignore_index = ignore_index

#     def forward(self, inputs, targets, smooth=1):
#         B, H, W = targets.shape

#         inputs = F.softmax(inputs, dim=1)

#         inputs = inputs[:, 1, :, :].reshape(B, H*W)
#         targets = targets.reshape(B, H*W)
        
#         dice_losses = []

#         for b in range(B):
#             trainable_mask = targets[b] != self.ignore_index
#             inputs_per_batch = inputs[b, trainable_mask]
#             targets_per_batch = targets[b, trainable_mask]

#             intersection = (inputs_per_batch * targets_per_batch).sum()                            
#             dice_loss = 1 - (2.*intersection + smooth)/(inputs_per_batch.sum() + targets_per_batch.sum() + smooth)
            
#             dice_losses.append(dice_loss)
        
#         return torch.stack(dice_losses).mean()

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, smooth=1):
        B, C, H, W = inputs.shape

        inputs = F.softmax(inputs, dim=1)

        inputs = inputs.reshape(B, C, H*W)
        targets = targets.reshape(B, H*W)

        dice_losses = []

        for b in range(B):
            trainable_mask = targets[b] != self.ignore_index

            inputs_per_batch = inputs[b, :, trainable_mask].contiguous().view(-1)
            targets_per_batch = F.one_hot(targets[b, trainable_mask].long(), C).permute(1, 0).contiguous().view(-1)

            intersection = (inputs_per_batch * targets_per_batch).sum()                            
            dice_loss = 1 - (2.*intersection + smooth)/(inputs_per_batch.sum() + targets_per_batch.sum() + smooth)
            
            dice_losses.append(dice_loss)
        
        return torch.stack(dice_losses).mean()

class ForegroundDiceLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, smooth=1):
        B, C, H, W = inputs.shape

        inputs = F.softmax(inputs, dim=1)

        inputs = inputs.reshape(B, C, H*W)
        targets = targets.reshape(B, H*W)

        dice_losses = []

        for b in range(B):
            trainable_mask = targets[b] != self.ignore_index

            inputs_per_batch = inputs[b, 1, trainable_mask].contiguous().view(-1)
            targets_per_batch = targets[b, trainable_mask].float().contiguous().view(-1)

            intersection = (inputs_per_batch * targets_per_batch).sum()                            
            dice_loss = 1 - (2.*intersection + smooth)/(inputs_per_batch.sum() + targets_per_batch.sum() + smooth)
            
            dice_losses.append(dice_loss)
        
        return torch.stack(dice_losses).mean()

class FocalTverskyLoss(nn.Module):
    def __init__(self, ignore_index=255, alpha=0.7, beta=0.3, gamma=4/3):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):
        B, C, H, W = inputs.shape

        inputs = F.softmax(inputs, dim=1)

        inputs = inputs.reshape(B, C, H*W)
        targets = targets.reshape(B, H*W)

        focaltversky_losses = []

        for b in range(B):
            trainable_mask = targets[b] != self.ignore_index

            inputs_per_batch = inputs[b, :, trainable_mask].contiguous().view(-1)
            targets_per_batch = F.one_hot(targets[b, trainable_mask].long(), C).permute(1, 0).contiguous().view(-1)

            TP = (inputs_per_batch * targets_per_batch).sum()
            FP = ((1 - targets_per_batch) * inputs_per_batch).sum()
            FN = (targets_per_batch * (1 - inputs_per_batch)).sum()

            Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
            FocalTversky = (1 - Tversky) ** self.gamma
            
            focaltversky_losses.append(FocalTversky)
        
        return torch.stack(focaltversky_losses).mean()

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.tag = 'L1'

    def forward(self, logits, labels, masks):
        B, _, _, _ = logits.shape
        
        rec_losses = []
        for b in range(B):
            fg_mask = masks[b] == 1
            bg_mask = masks[b] == 0

            if fg_mask.sum() > 0: fg_loss = F.l1_loss(logits[b, :, fg_mask], labels[b, :, fg_mask]).mean() 
            else: fg_loss = torch.zeros(1).to(logits.device)
            
            if bg_mask.sum() > 0: bg_loss = F.l1_loss(logits[b, :, bg_mask], labels[b, :, bg_mask]).mean() 
            else: bg_loss = torch.zeros(1).to(logits.device)
            
            rec_losses.append((fg_loss + bg_loss) / 2.)
        return torch.stack(rec_losses).mean()
    
class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.tag = 'MSE'

    def forward(self, logits, labels, masks):
        B, _, _, _ = logits.shape
        
        rec_losses = []
        for b in range(B):
            fg_mask = masks[b] == 1
            bg_mask = masks[b] == 0

            if fg_mask.sum() > 0: fg_loss = F.mse_loss(logits[b, :, fg_mask], labels[b, :, fg_mask]).mean() 
            else: fg_loss = 0.

            if bg_mask.sum() > 0: bg_loss = F.mse_loss(logits[b, :, bg_mask], labels[b, :, bg_mask]).mean() 
            else: bg_loss = 0.
            
            rec_losses.append((fg_loss + bg_loss) / 2.)

            # fgbg_mask = masks[b] != 255
            # rec_losses.append(F.mse_loss(logits[b, :, fgbg_mask], labels[b, :, fgbg_mask]).mean())

        return torch.stack(rec_losses).mean()