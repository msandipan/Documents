#from pysot.models.backbone.alexnet import alexnetlegacy, alexnet

from Networks.Resnet_atrous import resnet18, resnet34, resnet50
#from pysot.models.head.mask import MaskCorr, Refine
from Networks.NetworkSiamRPN import UPChannelRPN, DepthwiseRPN, MultiRPN

BACKBONES = {

              'resnet18': resnet18,
              'resnet34': resnet34,
              'resnet50': resnet50,

            }




def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)

RPNS = {
        'UPChannelRPN': UPChannelRPN,
        'DepthwiseRPN': DepthwiseRPN,
        'MultiRPN': MultiRPN
       }


def get_rpn_head(name, **kwargs):
    return RPNS[name](**kwargs)




