import os
import pandas as pd
import cv2
import numpy as np
import argparse

def non_max_supression(box):
    '''
    parameters:
    box(numpy[N, 4])
    '''
    def box_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    
    area = box_area(box)
    #from big to small
    index = np.argsort(-area, axis=0)
    box = box[index]

    box_new = None
    for b in box:
        if box_new is None:
            box_new = np.expand_dims(b, axis=0)
        else:
            np.vstack((box_new, b))
        x_start = np.maximum(b[0], box[:, 0])
        y_start = np.maximum(b[1], box[:, 1])
        x_end = np.minimum(b[2], box[:, 2])
        y_end = np.minimum(b[3], box[:, 3])

        inter = (y_end - y_start) * (x_end - x_start)
        box = box[inter==0]
    
    return box_new

        

class Point():
    def __init__(self, y, x, adj):
        self.y = y
        self.x = x
        #Eights adjacent point
        self.adj = adj
    def show(self):
        print(f'x: {self.x}')
        print(f'y: {self.y}')
        print(f'adj: {self.adj}')

def chess_board_distance(point1, point2):
    distance = max(abs(point1.x - point2.x), abs(point2.y - point1.y))
    return distance

def generate_adj(points, point):
    if len(points) == 0:
        points.append([point])
    else:              
        #if current point and points groups are adjacent
        adjacent = False
        for i, pointgroup in enumerate(points):
            for inpoint in pointgroup:
                if inpoint.adj == 8:
                    continue
                dis = chess_board_distance(inpoint, point)
                #adjacent
                if dis == 1:
                    adjacent = True
                    inpoint.adj += 1

            if adjacent:
                point.adj = 1
                pointgroup.append(point)
                break
        if not adjacent:
            points.append([point])
    return points

def seg2box(img, bord):
    H,W = img.shape
    points = []
    for h in range(H):
        for w in range(W):
            #background
            if img[h, w] == 0:
                continue
            #foreground
            point = Point(h, w, 0)

            points = generate_adj(points, point)
            
    #merge point group
    new_points = []
    for pointgroup in points:
        new_points += pointgroup

    points = []
    for point in new_points:
        points = generate_adj(points, point)
                
    boxes = []
    for pointgroup in points:
        for i, point in enumerate(pointgroup):
            if i == 0:
                x1 = point.x
                x2 = point.x
                y1 = point.y
                y2 = point.y
            else:
                if point.x < x1:
                    x1 = point.x
                if point.x > x2:
                    x2 = point.x
                if point.y < y1:
                    y1 = point.y
                if point.y > y2:
                    y2 = point.y
        #x1/y1/x2/y2:the left/top/right/bottom point of the point group
        boxes.append([max(x1-bord,0), max(y1-bord,0), min(x2+bord,W), min(y2+bord,H)])
    
    boxes = np.array(boxes)
    #boxes = non_max_supression(boxes)
    
    return boxes
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='./cGAN_data/training/', help="path of training ground truth")
    parser.add_argument("--save_path", type=str, default='./cGAN_data/training_box_gt.csv/', help="path to save box ground truth")
    parser.add_argument("--bord", type=int, default=4, help="expand mask's edge to generate box")
    args = parser.parse_args()

    bord = args.bord
    path = args.path
    save_path = args.save_path

    gtpaths = os.listdir(path)
    message = []
    for gtpath in gtpaths:
        if gtpath.endswith('_1.png'):
            continue
        gt_img = cv2.imread(os.path.join(path, gtpath), 0)
        h, w = gt_img.shape

        Boxes = seg2box(gt_img, bord=bord)#x1, y1, x2, y2
        Boxes = [' '.join(str(int(i)) for i in item) for item in Boxes]
        BoxesString = ";".join(Boxes)

        message.append([gtpath.replace('_2','_1'), BoxesString])


        print(f'{gtpath} record')

    message = pd.DataFrame(message, columns=['image_name', 'BoxesString'])
    #save bounding boxes message into a csv
    message.to_csv(save_path, index=False)
        
    


