import numpy as np
import torch
import torchvision

def xywh2xyxy(boxes):
    '''
    parameters:
    [Tensor]
    boxes:(n, 4), cx, cy, w, h

    function:
    trans cx cy w h to x1 y1 x2 y2
    '''
    boxes_pre = boxes.clone()
    boxes[..., 0] = boxes_pre[..., 0] - boxes_pre[...,2] / 2
    boxes[..., 1] = boxes_pre[..., 1] - boxes_pre[...,3] / 2
    boxes[..., 2] = boxes_pre[..., 0] + boxes_pre[...,2] / 2
    boxes[..., 3] = boxes_pre[..., 1] + boxes_pre[...,3] / 2

    return boxes

def non_max_suppression(prediction, conf_thres=0.4, iou_thres=0.4):
    '''
    parameters:
    [Tensor]
    prediction:(n, 5), x1, y1, x2, y2, conf
    '''
    prediction = prediction[prediction[..., 4] > conf_thres]

    boxes, scores = prediction[..., :4], prediction[..., 4]
    i = torchvision.ops.nms(boxes, scores, iou_thres)

    return prediction[i]
