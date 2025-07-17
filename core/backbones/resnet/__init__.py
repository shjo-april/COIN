import torch
from torch.utils import model_zoo

from .model import ResNet
from .model import Bottleneck, BasicBlock
from .model import urls_dic, urls_dic_v2, layers_dic

def build_resnet(model_name, norm_fn, activation_fn, last_stride=2, pretrained=True, output_stride=16, v2=False):
    if output_stride == 16:
        strides=(2, 2, 2, last_stride)
        dilations=(1, 1, 1, 1)
    elif output_stride == 8:
        strides=(2, 2, 1, last_stride)
        dilations=(1, 1, 2, 4)
    
    model = ResNet(
        BasicBlock if model_name in ['resnet18', 'resnet34'] else Bottleneck, 
        layers_dic[model_name], 
        strides=strides, dilations=dilations,
        batch_norm_fn=norm_fn, activation_fn=activation_fn
    )
    
    if pretrained:
        url = urls_dic_v2[model_name] if v2 else urls_dic[model_name]
        model.load_state_dict(model_zoo.load_url(url), strict=False)
    
    return model
