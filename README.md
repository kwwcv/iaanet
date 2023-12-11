# Interior Attention-Aware Network for Infrared Small Target Detection
Official implementation for IEEE Transactions on Geoscience and Remote Sensing (TGRS) paper: "Interior Attention-Aware Network for Infrared Small Target Detection".
[**[Paper]**](https://ieeexplore.ieee.org/document/9745054)

## News ##
[2023/12/11] I plan to reorganize the code and provide a more friendly model version.

## Requirement
**Packages:**
* Python 3.8
* Pytorch 1.7
* opencv-python
* numpy
* tqdm
* pandas
* yaml

## File Structure
```
iaanet
├─ box_generate.py
├─ cGAN_data
│  └─ training_box_gt.csv
├─ detect.py
├─ models
│  ├─ attention.py
│  ├─ backbone.py
│  ├─ embedding.py
│  └─ transformer.py
├─ pretrained
│  ├─ iaanet.pt
│  └─ rpn.pt
├─ test.py
├─ train.py
└─ utils
   ├─ datasets.py
   ├─ general.py
   └─ loss.py

```
## Dataset Preparation 
* MDvsFA-cGAN Dataset [**[Dataset]**](https://github.com/wanghuanphd/MDvsFA_cGAN)
[**[Paper]**](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Miss_Detection_vs._False_Alarm_Adversarial_Learning_for_Small_Object_ICCV_2019_paper.pdf)
* Dirs should be organized as:
```
cGAN_data
├─ training
│  ├─ 000000_1.png
│  ├─ 000000_2.png
│  ├─ ...
│  ├─ 009999_1.png
│  └─ 009999_2.png
├─ test_org
│  ├─ 00000.png
│  ├─ ...
│  └─ 00099.png 
└─  test_gt
   ├─ 00000.png
   ├─ ...
   └─ 00099.png 
```
* Following test dirs to organize validation set:
```
cGAN_data
├─ ...
├─ val_org
└─ val_gt
```
* Use prepared bounding boxes ground truth directly 
```
cGAN_data
├─ ...
└─ training_box_gt.csv
```
* Or run following command to generate bounding box ground truth from ground truth masks:
```
python box_generate.py --path ./cGAN_data/training/ --save_path ./cGAN_data/training_box_gt.csv --bord 4
```

## Training
Experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with a single GeForce RTX 3090 GPU of 24 GB Memory.

Train from scratch
```
python train.py --batch_size 8 --epochs 10 --save_path ./outputs/demo/
```
Start from pretraind RPN
```
python train.py --rpn_pretrained ./pretrained/rpn.pt --save_path ./outputs/demo/
```
Run `python train.py --help` for more configurations

## Testing
Use pretrained model for testing
```
python test.py --weights ./pretrained/iaanet.pt
```
Fast version (SG convs the proposed regions only. ):
```
python test.py --weights ./pretrained/iaanet.pt --fast
```
Run `python test.py --help` for more configurations

## Results
We follow MDvsFA-cGAN to calculate F-measure [[Code](https://github.com/wanghuanphd/MDvsFA_cGAN/blob/master/demo_MDvsFA_pytorch.py)]
| **Dataset** | **F-measure** | **Precision** | **Recall** |
| :---: | :---: | :---: | :---: |
| MDvsFA | 0.639 | 0.606 | 0.818 |
## Inference
Infer a single image
```
python detect.py --image_path img.png --save_path ./inference/ --weights ./pretrained/iaanet.pt
```
Infer images in a folder
```
python detect.py --image_path ./folder/ --save_path ./inference/ --weights ./pretrained/iaanet.pt --folder
```
Fast version
```
python detect.py --fast
```
Run `python detect.py --help` for more configurations

## Citation
If you find our work useful in your research, please cite our paper

```
@ARTICLE{9745054,
  author={Wang, Kewei and Du, Shuaiyuan and Liu, Chengxin and Cao, Zhiguo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Interior Attention-Aware Network for Infrared Small Target Detection}, 
  year={2022},
  doi={10.1109/TGRS.2022.3163410}
  }
```

## License
MIT License
