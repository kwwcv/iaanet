import torch
import cv2
import os
import random
import numpy as np
import pandas as pd
import math

#   MDvsFA_cGAN Dataset #        
class cGANDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, mask_path, bboxfile=None, stride=None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.stride = stride
        
        self.image = []#image names
        self.gt_bbox = []#bounding boxes 
        self.gt_mask = []#mask names

        gt_df = pd.read_csv(bboxfile)
        for _, row in gt_df.iterrows():
            image_name, BoxesString = row['image_name'], row['BoxesString']

            self.image.append(image_name)
            self.gt_mask.append(image_name.replace('_1', '_2'))

            BoxString = BoxesString.split(';')
            self.gt_bbox.append([[int(i) for i in item.split()] for item in BoxString])
            
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):  
        #input 
        img_path = os.path.join(self.image_path, self.image[index])
        img = cv2.imread(img_path, -1)
        img = img[..., 2]
        h, w = img.shape
        #segmentation label
        mask_path = os.path.join(self.mask_path, self.gt_mask[index])
        mask_label = cv2.imread(mask_path, -1)
        #detection label
        bbox_label = np.array(self.gt_bbox[index], dtype=np.float)
        
        pre_label = bbox_label.copy()#x1 y1 x2 y2
        #cx
        bbox_label[:,0] = (pre_label[:,0] + pre_label[:,2]) / 2.0 / w
        #cy
        bbox_label[:,1] = (pre_label[:,1] + pre_label[:,3]) / 2 / h
        #w
        bbox_label[:,2] = (pre_label[:,2] - pre_label[:,0]) / w
        #h
        bbox_label[:,3] = (pre_label[:,3] - pre_label[:,1]) / h
        #for batch index
        b_num = np.zeros((bbox_label.shape[0],1))
        bbox_label = np.hstack((b_num, bbox_label))
        
        img = np.expand_dims(img, axis=0)
        img = np.float32(img) / 255.0

        mask_label = np.expand_dims(mask_label, axis=0)
        mask_label = np.float32(mask_label) / 255.0

        return torch.from_numpy(img), torch.from_numpy(mask_label), torch.from_numpy(bbox_label)

    @staticmethod
    def collate_fn(batch):
        img, mask_label, bbox_label = zip(*batch)
        for i, l in enumerate(bbox_label):
            l[:, 0] = i

        return torch.stack(img, 0), torch.stack(mask_label, 0), torch.cat(bbox_label, 0)

if __name__ == '__main__':
    dataset_path = './training/'
    batch_size = 8
    epochs = 1
    
    dataset = cGANDataset(dataset_path)
    train_dataloader = torch.utils.data.DataLoader(dataset,
                                                    batch_size = batch_size,
                                                    num_workers = 8)

    for epoch in range(epochs):
        for i, (img, label) in enumerate(train_dataloader):
            print(i)
            print(img)
            print(label)