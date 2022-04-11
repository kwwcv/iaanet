from numpy.lib.function_base import average
import torch
import os
import numpy as np
import cv2
import argparse

#Metric F1: https://github.com/wanghuanphd/MDvsFA_cGAN/blob/master/demo_MDvsFA_pytorch.py
def calculateF1Measure(output_image,gt_image,thre):
    output_image = np.squeeze(output_image)
    gt_image = np.squeeze(gt_image)
    out_bin = output_image>thre
    gt_bin = gt_image>thre
    recall = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(gt_bin))
    prec   = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(out_bin))
    F1 = 2*recall*prec/np.maximum(0.001,recall+prec)
    return prec, recall, F1

def test(image_path, mask_path, model, device, conf_thres, iou_thres, expand, topk, fast):
    '''
    only support batch size = 1
    '''  
    average_F1 = 0
    average_prec = 0
    average_recall = 0

    with torch.no_grad():
        num = len(os.listdir(image_path))
        for i, img_name in enumerate(os.listdir(image_path)):
            print(f'{i+1}/{num}', end='\r', flush=True)
            input_img = cv2.imread(os.path.join(image_path, img_name), 0) / 255.0
            target = cv2.imread(os.path.join(mask_path, img_name), 0) / 255.0
            input = torch.from_numpy(input_img).to(torch.float)
            input = input[None, None, :]
            _, _, h, w = input.shape
            
            output, mask_maps, _, _= model(input.to(device), max_det_num=0, conf_thres=conf_thres, iou_thres=iou_thres, expand=expand, topk=topk, fast=fast)

            probability_map = torch.zeros((h, w), dtype=torch.float, device=device)
            if output is not None:
                output = output.squeeze()
                output = output.sigmoid()
                mask_maps = mask_maps.squeeze()

                probability_map[~mask_maps] = output

            probability_map = probability_map.cpu().numpy()
            prec, recall, F1 = calculateF1Measure(probability_map, target, 0.5)
            average_F1 = (average_F1 * i + F1) / (i + 1)
            average_prec = (average_prec * i + prec) / (i + 1)
            average_recall = (average_recall * i + recall) / (i + 1)
    print(f'prec:{average_prec}  recall:{average_recall}  F1: {average_F1} ')

    return average_F1

if __name__ == '__main__':
    #######################################
    #set up
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_image", type=str, default="./cGAN_data/test_org/", help="path to load testing image")
    parser.add_argument("--test_mask", type=str, default="./cGAN_data/test_gt/", help="path to load testing masks")
    parser.add_argument('--weights', type=str, default="./pretrained/iaanet.pt", help="path of the weights")
    parser.add_argument("--iou_thres", type=float, default=0.6, help="iou threshold for detection stage")
    parser.add_argument("--conf_thres", type=float, default=0.2, help="confidence threshold for detection stage")
    parser.add_argument("--topk", type=int, default=5, help="if predict no boxes, select out k region boxes with top confidence")
    parser.add_argument("--expand", type=int, default=8, help="The additonal side length of expanded local region for semantic generator")
    parser.add_argument('--fast', action='store_true', help='fast inference')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #####################################
    model = torch.load(args.weights)
    model.to(device)
    model.eval()
    
    test(args.test_image, args.test_mask, model, device, args.conf_thres, args.iou_thres, args.expand, args.topk, args.fast)
    