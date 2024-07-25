# 训练相关基本操作
## 环境
+ python
+ torch
+ 其他
```
pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple

```
+ 注意panopticapi和pycocotools可能会安装报错
## 数据集
+ 数据格式为coco格式,和detr要求一样


```
├── annotations
│    ├── instances_train2017.json
│    └── instances_val2017.json
│ 
├── train2017
│   
└── val2017
   

```
+ 数据集地址和类别数放在redetr_pytorch/configs/dataset/coco_detection.yml中
+ 替换rtdetr_pytorch/src/data/coco/coco_dataset.py中的mscoco_category2name类，定义为自己的类别名称
## 命令行执行
### 训练redetr_pytorch/tools/train.py
+ --config 训练有关参数主要放在redetr_pytorch/configs/redetr下的yml中指定
+ --resume 预训练权重checkpoint
+ --test-only 相当于验证
### 导出redetr_pytorch/tools/export_onnx.py
+ --config 导出有关参数主要放在redetr_pytorch/configs/redetr下的yml中指定
+ --resume 预训练权重checkpoint
+ --simplify 调用onnx-simplify

# RT-DETR比DETR更适合实时目标检测，已经集成到ultralytics库里面了
+ 按照yolo格式准备数据集和yaml配置文件
+ 直接调用RTDETR即可，代码如下：
```
import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='dataset/moon.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=4,
                workers=4,
                device='0',
                # resume='', # last.pt path
                project='runs/train',
                name='exp',
                )

```