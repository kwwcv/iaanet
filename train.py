import torch
import os
import matplotlib.pyplot as plt
import matplotlib
import yaml
import argparse
import torch.backends.cudnn as cudnn
from PIL import Image

matplotlib.use('Agg')
import numpy as np

from tqdm import tqdm

from models.attention import *
from models.backbone import *
from models.transformer import Transformer
from utils.datasets import *
from utils.loss import DetectLoss, SegLoss, ComputerLoss
from test import test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default='./cGAN_data/training/', help="path to load training image")
    parser.add_argument("--mask_path", type=str, default='./cGAN_data/training/', help="path to load training masks")
    parser.add_argument("--val_image", type=str, default="./cGAN_data/val_org/", help="path to load validation image")
    parser.add_argument("--val_mask", type=str, default="./cGAN_data/val_gt/", help="path to load validation masks")
    parser.add_argument("--bboxfile", type=str, default="./cGAN_data/trainning_box_gt.csv", help="path to bounding boxes ground truth")
    parser.add_argument("--save_path", type=str, default='./outputs/demo', help="path to save model")
    parser.add_argument("--pos_mode", type=str, default='cosin', help="position embedding type, ['cosin'] & [''learned] are available")
    parser.add_argument("--iou_thres", type=float, default=0.6, help="iou threshold for detection stage")
    parser.add_argument("--conf_thres", type=float, default=0.2, help="confidence threshold for detection stage")
    parser.add_argument("--topk", type=int, default=5, help="if predict no boxes, select out k region boxes with top confidence")
    parser.add_argument("--nel", type=int, default=4, help="number of encoder layer")
    parser.add_argument("--loss_mode", default=None, help="['focal'], None")
    parser.add_argument("--seg_posw", type=int, default=3, help="Positive weights of BCEwithLogitLoss for segmentation")
    parser.add_argument("--obj_posw", type=int, default=10, help="Positive weights of BCEwithLogitLoss for object detection")
    parser.add_argument("--max_det_num", type=int, default=5, help="Maximum number of region boxes proposed by detect head")
    parser.add_argument("--hidden_dim", type=int, default=512, help="hidden_dim of transformer")
    parser.add_argument("--d_lr", type=float, default=0.01, help="Learning Rate for detection")
    parser.add_argument("--s_lr", type=float, default=0.001, help="Learning Rate for segmentation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Size")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs")
    parser.add_argument('--rpn_pretrained', type=str, default=None, help='load pretrained rpn weights')
    parser.add_argument("--expand", type=int, default=8, help="The additonal length of expanded local region for semantic generator")
    parser.add_argument('--fast', action='store_true', help='fast inference')
    args = parser.parse_args()
    #################set up######################
    os.makedirs(args.save_path, exist_ok=True)
    #save args
    with open(os.path.join(args.save_path, 'args.yaml'), mode='w') as f:
        yaml.dump(vars(args), f)
    #####################################
    #device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark, cudnn.deterministic = True, False
    #  Model #
    #Region Proposal Networks
    backbone = backbone()
    region_module = region_propose(backbone)
    if args.rpn_pretrained:
        region_module = torch.load(args.rpn_pretrained)
        
    #Attention Encoder
    attention_module = Transformer(num_encoder_layers=args.nel, d_model=args.hidden_dim)
    #IAANet
    Model = attention(attention_module, region_module, pos=args.pos_mode, d_model=args.hidden_dim)
    Model.to(device)
    #####################################
    #training dataset 
    dataset_train = cGANDataset(args.image_path, args.mask_path, bboxfile=args.bboxfile, stride=backbone.stride)
   
    train_dataloader = torch.utils.data.DataLoader(dataset_train,
                                                    shuffle = True,
                                                    batch_size = args.batch_size,
                                                    num_workers = 8,
                                                    collate_fn=dataset_train.collate_fn)
    nb = len(train_dataloader)

    ####################################
    #optimizer
    param_dicts = [
        {"params":[v for k, v in Model.named_parameters() if "region_module" in k and v.requires_grad],
         "lr": args.d_lr},
        {"params":[v for k, v in Model.named_parameters() if "region_module" not in k and v.requires_grad],
         "lr":args.s_lr}
        ]
    
    optimizer = torch.optim.SGD(param_dicts)
    
    fun_1 = lambda epoch: 1
    if args.rpn_pretrained:
        fun_2 = lambda epoch: 1
    else:
        fun_2 = lambda epoch: 0 if epoch<1 else 1

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[fun_1, fun_2], verbose=True)

    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, verbose=True)
    ######################################
    #Loss function
    Dcriterion = DetectLoss(obj_pw=args.obj_posw, device=device)
    Scriterion = SegLoss(posw=args.seg_posw, mode=args.loss_mode, device=device)
    criterion = ComputerLoss(Dcriterion, Scriterion)

    Loss_list = []
    Loss_box = []
    Loss_obj = []
    Loss_d = []
    Loss_s = []
    F1_list = []
    x = range(args.epochs)
    #Best metric F1 measurement for val
    Best_F1 = 0
    for epoch in range(args.epochs):
        #set model to training model
        Model.train()
        print(f'Epoch: {epoch} / {args.epochs}')
        
        pbar = enumerate(train_dataloader)
        pbar = tqdm(pbar, total=nb)
        
        mloss = np.zeros(5, dtype=np.float)
        for i, (input, seg_targets, bbox_targets) in pbar:

            detect_output, reg_output, mask_maps, target_boxes = Model(input.to(device), max_det_num=args.max_det_num, conf_thres=args.conf_thres, iou_thres=args.iou_thres, expand=args.expand, topk=args.topk)
            loss, loss_items = criterion(Dp=detect_output, Dtarget=bbox_targets.to(device), 
                                            anchor=backbone.anchor, stride=backbone.stride,
                                            Sp=reg_output, Starget=seg_targets.to(device), mask_maps=mask_maps,
                                            target_boxes=target_boxes)
            #print(loss_items)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            mloss = (mloss * i + loss_items.cpu().numpy()) / (i + 1)

        Loss_box.append(mloss[0])
        Loss_obj.append(mloss[1])
        Loss_d.append(mloss[2])
        Loss_s.append(mloss[3])
        Loss_list.append(mloss[4])
        print(f'{reg_output.sigmoid().max()}')

        print(('%10s' * 5) % ('box', 'obj', 'Dloss', 'Sloss', 'total'))
        print(('%10.4g' * 5) % (mloss[0], mloss[1], mloss[2], mloss[3], mloss[4]))
        
        #print(f'{outputs[..., -1].sigmoid().max()}')
        lr_scheduler.step()

        #save last model
        torch.save(Model, os.path.join(args.save_path, 'last.pt'))
        #val 
        Model.eval()
        F1 = test(args.val_image, args.val_mask, Model, device=device, conf_thres=args.conf_thres, iou_thres=args.iou_thres, expand=args.expand, topk=args.topk, fast=args.fast)
        F1_list.append(F1)
        if F1 > Best_F1:
            Best_F1 = F1
            torch.save(Model, os.path.join(args.save_path, 'best.pt'))
        #save
        if (epoch + 1) % 3 == 0:
            torch.save(Model, os.path.join(args.save_path, f'model{epoch}.pt'))
    print(f'Best F1: {Best_F1}')
    #save the last epoch
    torch.save(Model, os.path.join(args.save_path, 'model.pt'))
    #draw
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(x, Loss_list, 'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')

    ax2 = plt.subplot(2, 3, 2)
    plt.plot(x, F1_list, 'o-')
    plt.xlabel('Epoch')
    plt.ylabel('F1')

    ax4 = plt.subplot(2, 3, 4)
    plt.plot(x, Loss_d, 'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Detect Loss')

    ax3 = plt.subplot(2, 3, 5)
    plt.plot(x, Loss_s, 'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Segment Loss')

    plt.savefig(os.path.join(args.save_path, 'result.jpg'))