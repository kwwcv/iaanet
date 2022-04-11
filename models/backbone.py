import torch
import torch.nn.functional as F
from torch import nn
import torchvision

from utils.general import*

class backbone(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet.resnet18(pretrained=True)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

        return_layer = {"layer3": "0"}
        self.stride = 16
        self.num_channel = 256
        self.anchor = torch.tensor([10, 10])

        self.body = torchvision.models._utils.IntermediateLayerGetter(backbone, return_layers=return_layer)
        #x y w h obj 
        self.detect = nn.Conv2d(self.num_channel, 5, kernel_size=(1,1), stride=(1,1))

    def forward(self, x, device='cuda'):
        src = self.body(x)
        src = src['0']
        x = self.detect(src)
        bs, _, ny, nx = x.shape
        x = x.permute(0, 2, 3, 1)
        
        #if not self.training:
        #get bounding box 
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        grid = torch.stack((xv, yv), 2).view((1, ny, nx, 2)).float().to(device)

        y = x.sigmoid()
        xy = (y[..., 0:2] * 2 - 0.5 + grid) * self.stride
        wh = (y[..., 2:4] * 2) ** 2 * self.anchor.view(1, 1, 1, 2).to(device)
        y = torch.cat((xy, wh, y[..., 4:]), -1)

        return x, y.view(bs, -1, 5)

class region_propose(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    
    def forward(self, x):
        _, _, h, w = x.shape
        detect_output, boxes = self.backbone(x)
        boxes[..., :4] = xywh2xyxy(boxes[..., :4])
        #clamp
        boxes[...,[0,2]]= boxes[...,[0,2]].clamp(0, w)
        boxes[...,[1,3]]= boxes[...,[1,3]].clamp(0, h)

        
        return detect_output, boxes

class semantic_estab(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(512, d_model, kernel_size=(1,1), stride=(1,1), bias=False),
        )

    def forward(self, x):
        x = self.block1(x)
        layer1 = self.layer1(x)

        x = self.block2(layer1)
        
        return x 

if __name__ == '__main__':

    input = torch.rand(1,1,146,256)
    backbone = backbone(mode='bbox')
    model = region_propose(backbone)
    model = model.to('cuda')
    output = model(input.to('cuda'))

    print(output.shape)
