# GarbageDetection
“华为云杯”2020深圳开放数据应用创新大赛·生活垃圾图片分类.排名:50/4388;方案:SSD-efficientd3-DiouLoss

# “华为云杯”2020深圳开放数据应用创新大赛·生活垃圾图片分类.

- 排名:50/4388;
- 方案:SSD-efficientd3-DiouLoss
  本次比赛由于需要在modelart上面线上部署，而且有推理时间限制，因此我采取的是单阶段模型SSD.主干网络efficientNetd3/ResNet50.
  loss 由smoothL1更改为了CiouLoss.

## 对仓库代码的说明

所用的SSD是pytoch版本，来源于该仓库：[SSD](https://github.com/lufficc/SSD).

针对本次垃圾检测分类我所做的更改如下：

- BackBone的替换:增加了ResNet系列(torchvision的官方实现)，提取了resnet四层feature
  - 关键在于提取resnet的四层feature,得到每个特征图的channel大小,对应输入尺度的相应尺寸，写好相关配置文件
- loss的替换:由SmoothL1替换为IOU/Diou/Giou/Ciou-loss系列.分类loss有采取过focalLoss,但表现差强人意
  - 在SSD中，网络的输出是偏置，原来仓库的实现计算的偏置的SmoothL1
  - 算iouloss需要四个点的坐标.因此需要将网络的输出(偏置)利用先验框解码为x1,y1,x2,y2
### ssd/modeling/backbone/resnet.py

这个是新增加的以resnet作为banckbone的脚本(提取出4层featureMaps)

### ssd/modeling/box_head/loss_iou.py

iou/giou/diou/ciou的实现

### make_imagesets.py

分割voc的训练集/验证集/测试集合

### customize_service.py

华为云modelart需要的推理脚本

###  get_ratio.py

计算类别比例
