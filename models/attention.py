import torch
from torch import nn
from torch._C import dtype
import torch.nn.functional as F
import numpy as np
import random
import os

from utils.general import*
from .embedding import*
from .backbone import semantic_estab

#filter out extremely tiny bounding boxes
def tiny_filter(boxes):
    """
    Input:
        boxes: x1, y1, x2, y2, conf
    """
    delta_w = boxes[:, 2] - boxes[:, 0]
    delta_h = boxes[:, 3] - boxes[:, 1]

    keep = (delta_w >=1) & (delta_h >=1)

    return boxes[keep]

def RandomSelectNegRegion(x, num, smin=5, smax=8):
    C, H, W = x.shape
    #random select number of negative boxes
    #num = random.randint(1, num)
    num = num
    neg_boxes = []
    for n in range(num):
        cx = random.randint(smax, W-smax)
        cy = random.randint(smax, H-smax)
        rw = random.randint(smin, smax)
        rh = random.randint(smin, smax)

        neg_boxes.append(torch.tensor([cx-rw, cy-rh, cx+rw, cy+rh, 0.5], dtype=torch.float))
    if num == 0:
        neg_boxes = None
    else:
        neg_boxes = torch.stack(neg_boxes, dim=0)
    return neg_boxes


class attention(nn.Module):
    def __init__(self, transformer, region_module, pos='cosin', d_model=256):
        super().__init__()
        self.transformer = transformer
        self.region_module = region_module
        self.seman_module = semantic_estab(d_model=d_model)
        self.pos = pos
        
        if pos == 'cosin':
            self.posembedding = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)
        elif pos == 'learned':
            self.posembedding = PositionEmbeddingLearned(num_pos_feats=d_model // 2, num_embedding=700)

        self.pro_embed = MLP(d_model, d_model, 1, 3)
        self.d_model = d_model

    def forward(self, x, max_det_num=10, conf_thres=0.2, iou_thres=0.4, expand=10, topk=5, fast=False):
        b, c, h, w = x.shape
        #get semantic feature map of the whole input images
        if (self.training) | (not fast):
            seman_feat = self.seman_module(x)
        #position embedding the whole image
        if self.pos is None:
            image_poses = torch.zeros((b, self.d_model, h, w), device=x.device)
        else:
            image_poses = self.posembedding(x)
        
        
        detect_output, region_boxes = self.region_module(x)
        region_boxes = region_boxes.detach()
        
        #For self-attn
        region_words = []
        region_poses = []
        mask_maps = torch.ones((b, h, w), device=x.device, dtype=torch.bool)#mask area out of proposed boxes

        target_boxes = []
        max_words_num = 0

        region_boxes_exists = True#whether detector find targets (Inference)
        max_words_num = 0
        for i in range(b):
            r_boxes = region_boxes[i]
            image_pos = image_poses[i]
            #NMS
            boxes = non_max_suppression(r_boxes, conf_thres=conf_thres, iou_thres=iou_thres)
            
            if (r_boxes[:,4] > conf_thres).sum() > 0:
                boxes = non_max_suppression(r_boxes, conf_thres=conf_thres, iou_thres=iou_thres)
                #filter out boxes that are too small
                boxes = tiny_filter(boxes)

            else:
                boxes = non_max_suppression(r_boxes, conf_thres=0.0, iou_thres=iou_thres)
                #filter out boxes that are too small
                boxes = tiny_filter(boxes)
                boxes = boxes[:topk]
            
            #If detector proposed no region boxes, record and jump out (Inference).
            if (self.training==False) & (len(boxes)==0):
                region_boxes_exists = False
                break

            if len(boxes) < max_det_num:
                #Select negative region
                neg_boxes = RandomSelectNegRegion(x[i], max_det_num-len(boxes), len(boxes))
                #combine
                if neg_boxes is not None:
                    neg_boxes = neg_boxes.to(boxes.device)
                    boxes = torch.cat([boxes, neg_boxes], dim=0)
                #Keep no overlap
                boxes = non_max_suppression(boxes, conf_thres=0.0, iou_thres=iou_thres)

            elif self.training:
                boxes = boxes[:max_det_num]

            boxes = boxes[:,:4]

            #region_pos = []
            target_box = []

            region_word = torch.ones((h, w, self.d_model), device=x.device)
            region_pos = torch.ones((h, w, self.d_model), device=x.device)
            mask_map = mask_maps[i]
            for box in boxes:
                x1, y1, x2, y2 = (box[0]+0.5).to(torch.int), (box[1]+0.5).to(torch.int), (box[2]+0.5).to(torch.int), (box[3]+0.5).to(torch.int)
                target_box.append([x1.item(), y1.item(), x2.item(), y2.item()])

                if (self.training) | (not fast):
                    #get region's semantic feature
                    r_seman_feat = seman_feat[i, :, y1:y2, x1:x2]
                else:
                    cw, ch = (x2-x1), (y2-y1)        
                    #get region's semantic feature
                    #get local semantic of input images
                    seman_feat = self.seman_module(x[i,:,max(0,y1-expand):min(h,y2+expand), max(0,x1-expand):min(w,x2+expand)].unsqueeze(dim=0))

                    ly1 = y1 if y1-expand < 0 else expand
                    ly2 = ly1 + ch
    
                    lx1 = x1 if x1-expand < 0 else expand
                    lx2 = lx1 + cw
                    
                    seman_feat = seman_feat.squeeze(dim=0)
                    r_seman_feat = seman_feat[:, ly1:ly2, lx1:lx2]

                r_seman_feat = r_seman_feat.permute(1,2,0).contiguous()
                word = r_seman_feat
                
                region_word[y1:y2, x1:x2, :] = word
                region_pos[y1:y2, x1:x2, :] = image_pos[:, y1:y2, x1:x2].permute(1, 2, 0)
                mask_map[y1:y2, x1:x2] = False


            target_boxes.append(target_box)

            region_word = region_word[~mask_map]
            region_pos = region_pos[~mask_map]

            region_words.append(region_word)
            region_poses.append(region_pos)

            if len(region_word) > max_words_num:
                max_words_num = len(region_word)

        if (self.training==False) & (region_boxes_exists==False):
            seg_output = None
            target_box = None
            region_boxes = None
        else:
            #pad words/poses length of each batch to be equal
            region_mask = torch.ones((b, max_words_num), device=region_word.device, dtype=torch.bool)
            region_words_pad = torch.zeros((b, max_words_num, self.d_model), device=region_word.device, dtype=region_word.dtype)
            region_poses_pad = torch.zeros((b, max_words_num, self.d_model), device=region_word.device, dtype=region_word.dtype)
            for i, w in enumerate(region_words):
                l, _ = w.shape
                region_words_pad[i,:l,:] = w
                region_poses_pad[i,:l,:] = region_poses[i]
                region_mask[i,:l] = False

            output = self.transformer(region_words_pad, region_poses_pad, region_mask)
            output = output.permute(1, 0, 2)

            seg_output = self.pro_embed(output)
 

        return (detect_output, seg_output, mask_maps, target_boxes) if self.training else (seg_output, mask_maps, region_boxes, target_boxes)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
