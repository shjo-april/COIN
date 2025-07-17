import torch
from torch.utils import model_zoo

from .resnest import resnest50, resnest101, resnest200, resnest269

def build_resnest(model_name, pretrained=True, output_stride=16):
    if output_stride == 16:
        dilation, dilated = 2, False
    elif output_stride == 8:
        dilation, dilated = 4, True
    
    # resnest269()
    model = eval(model_name)(pretrained=pretrained, dilated=dilated, dilation=dilation)
    
    del model.avgpool
    del model.fc

    return model
