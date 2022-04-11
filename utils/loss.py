import torch
import math
from torch import nn
from utils.general import*
import numpy as np

#implementation https://github.com/ultralytics/yolov3/blob/master/utils/metrics.py
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class ComputerLoss(nn.Module):
    def __init__(self, Dloss, Sloss):
        super(ComputerLoss, self).__init__()
        self.Dloss = Dloss
        self.Sloss = Sloss
        self.d = 1.0
        self.s = 1.0
    def forward(self, Dp, Dtarget, anchor, stride, Sp, Starget, mask_maps, target_boxes):
        dloss, lbox, lobj = self.Dloss(Dp, Dtarget, anchor, stride)
        sloss = self.Sloss(Sp, Starget, mask_maps, target_boxes)

        loss = dloss * self.d + sloss * self.s

        return loss, torch.cat((lbox, lobj.unsqueeze(dim=0), dloss, sloss.unsqueeze(dim=0), loss)).detach()

class SegLoss(nn.Module):
    def __init__(self, posw=1, mode=None, device='cuda'):
        super(SegLoss, self).__init__()
        BCE = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([posw], device=device))
        #focal loss?
        self.mode = mode
        
        if mode=='focal':
            self.BCE = FocalLoss(BCE)
        else:
            self.BCE = BCE
        

    def forward(self, pred, targets, mask_maps, target_boxes):
        pred = pred.squeeze(dim=-1)
        #pred = pred.sigmoid()
        b, _, = pred.shape

        targets = targets.squeeze(dim=1)
        loss = 0
        for i in range(b):
            mask_map = mask_maps[i]
            target = targets[i]
            p = pred[i]

            target = target[~mask_map]
            p = p[:len(target)]
        
            loss += self.BCE(p, target)
        
        loss = loss / b
        #print(f"{pred.max()}")
        #print(f'{loss}')
        return loss

#implementation https://github.com/ultralytics/yolov3/blob/master/utils/loss.py
class DetectLoss(nn.Module):
    def __init__(self, obj_pw, device):
        super(DetectLoss, self).__init__()
        self.device = device
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw], device=self.device))
        self.box = 0.5
        self.obj = 1.0
    
    def forward(self, p, target, anchor, stride):
        lbox = torch.zeros(1, device=self.device)
        indices, tbox = [], []
        fh, fw = p.shape[1:3]#feature map size
        gain = torch.tensor([fw, fh], device=self.device)

        b = target[:,0].long().T #image
        gxy = target[:, 1:3] * gain#grid xy
        gwh = target[:, 3:5] * gain#grid wh
        gij = gxy.long()
        gi, gj = gij.T #grid xy indices

        gj = gj.clamp_(0, fh-1)
        gi = gi.clamp_(0, fw-1)
        tbox = (torch.cat((gxy - gij, gwh), 1))

        tobj = torch.zeros_like(p[..., 0], device=self.device)
        n = b.shape[0] #number of targets

        if n:
            ps = p[b, gj, gi] #prediction subset corresponding to targets

            #regression
            pxy = ps[:, :2].sigmoid() * 2 - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchor.to(self.device) / stride
            pbox = torch.cat((pxy, pwh), 1)

            iou = bbox_iou(pbox.T, tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            lbox += (1.0 - iou).mean() 

            #objectness
            tobj[b, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)
        lobj = self.BCEobj(p[..., 4], tobj)

        lbox *= self.box
        lobj *= self.obj

        loss = lbox + lobj
        return loss, lbox, lobj
        #torch.cat((lbox, lobj.unsqueeze(dim=0), loss)).detach()







