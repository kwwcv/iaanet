import torch
import os
import cv2
import numpy as np
import argparse


if __name__ == '__main__':
    #######################################
    #set up
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="./img.png", help="path of the image folder/file")
    parser.add_argument("--save_path", type=str, default="./inference/", help="path to save results")
    parser.add_argument('--folder', action='store_true', help='detect images in folder (default:image file)')
    parser.add_argument('--weights', type=str, default="./pretrained/iaanet.pt", help="path of the weights")
    parser.add_argument("--iou_thres", type=float, default=0.6, help="iou threshold for detection stage")
    parser.add_argument("--conf_thres", type=float, default=0.2, help="confidence threshold for detection stage")
    parser.add_argument("--topk", type=int, default=5, help="if predict no boxes, select out k region boxes with top confidence")
    parser.add_argument("--expand", type=int, default=8, help="The additonal length of expanded local region for semantic generator")
    parser.add_argument('--fast', action='store_true', help='fast inference')
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #####################################
    #dataset 
    if args.folder:
        datalist = os.listdir(args.image_path)
    else:
        datalist = [(args.image_path).split('/')[-1]]
    #model
    Model = torch.load(args.weights)
    Model.to(device)
    Model.eval()

    with torch.no_grad():
        for img_path in datalist:
            if args.folder:
                input = cv2.imread(os.path.join(args.image_path, img_path), 0)
            else:
                input = cv2.imread(args.image_path, 0)
                
            h, w = input.shape
            img = input[None,None,:]
            img = np.float32(img) / 255.0

            input = torch.from_numpy(img)
            #max_det_num=0 for inference
            output, mask_maps, region_boxes, _ = Model(input.to(device), max_det_num=0, conf_thres=args.conf_thres, iou_thres=args.iou_thres, expand=args.expand, topk=args.topk, fast=args.fast)
        
            #segmentation results
            probability_map = torch.zeros((h, w), dtype=torch.float, device=device)
            if output is not None:
                output = output.squeeze()
                output = output.sigmoid()
                mask_maps = mask_maps.squeeze()

                probability_map[~mask_maps] = output
                
            probability_map = probability_map.cpu().numpy()
            probability_map = np.uint8(probability_map * 255)

            cv2.imwrite(os.path.join(args.save_path, img_path.replace('jpg','png')), probability_map)
            print(f'record: {img_path}')
