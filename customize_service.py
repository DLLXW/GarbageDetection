# -*- encoding: utf-8 -*-
from PIL import Image
import logging as log
from model_service.pytorch_model_service import PTServingBaseService
from metric.metrics_manager import MetricsManager
import torch.nn.functional as F

import torch.nn as nn
import torch
import json
import numpy as np
import torchvision
import time
import os
import copy

import sys
import cv2
from ssd.config import cfg
from ssd.modeling.detector import build_detection_model
import torch
from torch import nn

import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from save_json import label_list, create_class_dict, get_classes_name, save_result_as_json

print('CUDA available: {}'.format(torch.cuda.is_available()))


logger = log.getLogger(__name__)

IMAGES_KEY = 'images'
MODEL_INPUT_KEY = 'images'

def Net(model_path):
    cfg.merge_from_file('configs/efficient_net_b3_ssd300_voc0712.yaml')
    model = build_detection_model(cfg)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['model']
    model.load_state_dict(state_dict)
    #model.eval()
    return model
class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                   self.size))
        return image, boxes, labels
class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels
class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels
def remove_empty_boxes(boxes, labels):
    """Removes bounding boxes of W or H equal to 0 and its labels

    Args:
        boxes   (ndarray): NP Array with bounding boxes as lines
                           * BBOX[x1, y1, x2, y2]
        labels  (labels): Corresponding labels with boxes

    Returns:
        ndarray: Valid bounding boxes
        ndarray: Corresponding labels
    """
    del_boxes = []
    for idx, box in enumerate(boxes):
        if box[0] == box[2] or box[1] == box[3]:
            del_boxes.append(idx)

    return np.delete(boxes, del_boxes, 0), np.delete(labels, del_boxes)
class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
            if boxes is not None:
                boxes, labels = remove_empty_boxes(boxes, labels)
        return img, boxes, labels
def decode_image(file_content):
    """
    Decode bytes to a single image
    :param file_content: bytes
    :return: ndarray with rank=3
    """
    image = Image.open(file_content)
    image = image.convert('RGB')
    # print(image.shape)
    image = np.array(image)
    return image

def build_transforms():
    transform = [
        Resize(300),
        SubtractMeans([123, 117, 104]),
        ToTensor()
    ]
    transform = Compose(transform)
    return transform



class PTVisionService(PTServingBaseService):

    def __init__(self, model_path):
        super(PTVisionService, self).__init__(model_path)

        self.dir_path = os.path.dirname(os.path.realpath(model_path))
        # load label name
        self.label = ('__background__','一次性快餐盒','书籍纸张','充电宝',
                  '剩饭剩菜' ,'包','垃圾桶','塑料器皿','塑料玩具','塑料衣架',
                   '大骨头','干电池','快递纸袋','插头电线','旧衣服','易拉罐',
                   '枕头','果皮果肉','毛绒玩具','污损塑料','污损用纸','洗护用品',
                   '烟蒂','牙签','玻璃器皿','砧板','筷子','纸盒纸箱','花盆',
                   '茶叶渣','菜帮菜叶','蛋壳','调料瓶','软膏','过期药物',
                   '酒瓶','金属厨具','金属器皿','金属食品罐','锅','陶瓷器皿',
                   '鞋','食用油桶','饮料瓶','鱼骨')
        self.classify={'充电宝': '可回收物', '包': '可回收物', '洗护用品': '可回收物', '塑料玩具': '可回收物',
                    '塑料器皿': '可回收物', '塑料衣架': '可回收物', '玻璃器皿': '可回收物', '金属器皿': '可回收物',
                    '快递纸袋': '可回收物', '插头电线': '可回收物', '旧衣服': '可回收物', '易拉罐': '可回收物',
                    '枕头': '可回收物', '毛绒玩具': '可回收物', '鞋': '可回收物', '砧板': '可回收物', '纸盒纸箱': '可回收物', '调料瓶': '可回收物',
                    '酒瓶': '可回收物', '金属食品罐': '可回收物', '金属厨具': '可回收物', '锅': '可回收物',
                    '食用油桶': '可回收物', '饮料瓶': '可回收物', '书籍纸张': '可回收物', '垃圾桶': '可回收物',
                    '剩饭剩菜': '厨余垃圾', '大骨头': '厨余垃圾', '果皮果肉': '厨余垃圾', '茶叶渣': '厨余垃圾',
                    '菜帮菜叶': '厨余垃圾', '蛋壳': '厨余垃圾', '鱼骨': '厨余垃圾',
                    '干电池': '有害垃圾', '软膏': '有害垃圾', '过期药物': '有害垃圾',
                    '一次性快餐盒': '其他垃圾', '污损塑料': '其他垃圾', '烟蒂': '其他垃圾', '牙签': '其他垃圾',
                    '花盆': '其他垃圾', '陶瓷器皿': '其他垃圾', '筷子': '其他垃圾', '污损用纸': '其他垃圾'
                     }
        self.num_class = len(self.label)
        self.transform = build_transforms()
        self.score_threshold=0.5

        # Load your model
        self.model = Net(model_path)

    def _preprocess(self, data):

        preprocessed_data = {}

        pre_st = time.time()

        for k, v in data.items():
            for file_name, file_content in v.items():
                # print('\tAppending image: %s' % file_name)
                image = decode_image(file_content)
                height, width = image.shape[:2]
                image = self.transform(image)[0].unsqueeze(0)
                sample = {'img': image, 'img_name': file_name,'scale':[height,width]}
                preprocessed_data[k] = sample

        pre_et = time.time()
        self.pre_time = pre_st-pre_et
        return preprocessed_data

    def _inference(self, data):
        sample = data[IMAGES_KEY]  # img, img_name, scale
        images=sample['img']
        height, width=sample['scale']
        st = time.time()
        if torch.cuda.is_available():
            device = torch.device('cuda')
            cpu_device = torch.device('cpu')
        else:
            device = torch.device('cpu')
        self.model = self.model.to(device)
        self.model = self.model.eval()
        with torch.no_grad():
            result = self.model(images.to(device))[0]
            result = result.resize((width, height)).to(cpu_device).numpy()
            boxes, labels, scores = result['boxes'], result['labels'], result['scores']
            indices = scores > self.score_threshold
            boxes = boxes[indices]
            labels = labels[indices]
            scores = scores[indices]
            et = time.time()
            # print('Elapsed time: {} or {}(et-st)'.format(time.time() - st, et - st))
            result_i = {'img_name': sample['img_name'], 'scale': sample['scale'],
                        'class': labels, 'pred_boxes': boxes,
                        'score': scores, 'time': et - st}

        return result_i

    def _postprocess(self, data):
        class_name = self.label
        labels = label_list(os.path.join(self.dir_path, 'data/class_name.csv'))  # ('data/class_name.csv')#

        post_st = time.time()
        result = data
        img_name = result['img_name']
        scale = result['scale']
        classification = result['class']
        predict_bboxes = result['pred_boxes']
        scores = result['score']
        lantecy_time = result['time']
        detection_bboxes = []
        detection_classes = []
        for c in range(len(scores)):
            text1 = self.label[classification[c]]
            text2=self.classify[text1]
            text=text1+'/'+text2
            detection_classes.append(text)
            detection_bboxes.append(predict_bboxes[c])

        post_et = time.time()
        self.post_time = post_st - post_et
        all_run_time = lantecy_time + self.pre_time + self.post_time
        all_run_time *=1000     # ms
        json_file = save_result_as_json(img_name, detection_classes, np.array(scores),
                                        np.array(detection_bboxes), all_run_time)

        return json_file

    def inference(self, data):
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')

        if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)

        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000

        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)

        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
        if self.model_name + '_LatencyInference' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)

        # Update overall latency metric
        if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)

        logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        data['latency_time'] = str(round(pre_time_in_ms + infer_in_ms + post_time_in_ms, 1)) + ' ms'
        return data


