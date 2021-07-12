import torch.nn as nn
import torch.nn.functional as F

#from pysot.core.config import cfg
#from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from Networks.get_networks import get_backbone
from Networks.get_networks import get_rpn_head
#from pysot.models.neck import get_neck
import info


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        #read details form the code in git for kwargs
        self.backbone = get_backbone(info.BACKBONE_TYPE,
                                     **cfg.BACKBONE.KWARGS)




        # build rpn head
        #read details form the code in git for kwargs
        self.rpn_head = get_rpn_head(info.RPN_TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head

    def template(self, z):
        zf = self.backbone(z)

        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)

        cls, loc = self.rpn_head(self.zf, xf)

        return {
                'cls': cls,
                'loc': loc,
               }

