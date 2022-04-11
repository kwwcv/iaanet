# Interior Attention-Aware Network for Infrared Small Target Detection
Official implementation for TGRS article "Interior Attention-Aware Network for Infrared Small Target Detection".
[**[Link]**](https://ieeexplore.ieee.org/document/9745054)
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

## Requirement
**Packages:**
* Python 3.8
* Pytorch 1.7
* opencv-python
* numpy
* tqdm
* pandas

## Dataset Preparation 
* MDvsFA-cGAN Dataset[**[Link]**](https://github.com/wanghuanphd/MDvsFA_cGAN)
* After unzip, dirs should be organized as:
    * cGAN_data
        * training
            * 000000_1.png
            * 000000_2.png
            * ...
            * 009999_1.png
            * 009999_2.png
        * test_org
            * 00000.png
            * ...
            * 00099.png 
        * test_gt
            * 00000.png
            * ...
            * 00099.png 
* Following test dirs to organize validation set as:
    * cGAN_data
        * val_org
        * val_gt
* Use our prepared bounding boxes ground truth directly 
    * cGAN_data
        * training_box_gt.csv
* Or run following command to generate bounding box ground truth from ground truth masks:
```
python box_generate.py --path ./cGAN_data/training/ --save_path ./cGAN_data/training_box_gt.csv --bord 4
```


## Training
Run following command to train model from scratch
```
python train.py --batch_size 8 --epochs 10 --save_path ./outputs/demo/
```
Start from pretraind RPN
```
python train.py --rpn_pretrained ./pretrained/rpn.pt --save_path ./outputs/demo/
```

## Testing
Run our pretrained model for testing
```
python test.py --weights ./pretrained/iaanet.pt
```
Run in fast version:
```
python test.py --weights ./pretrained/iaanet.pt --fast
```
## Inference
Run following command to infer a single image
```
python detect.py --image_path img.png --save_path ./inference/ --weights ./pretrained/iaanet.pt
```
Run following command to infer images in a folder
```
python detect.py --image_path ./folder/ --save_path ./inference/ --weights ./pretrained/iaanet.pt --folder
```