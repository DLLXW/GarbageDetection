# -*- coding: utf-8 -*-
import glob
import os
import time
import cv2

import random
import torch
from PIL import Image, ImageFont, ImageDraw
from vizer.draw import draw_boxes

from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset
import argparse
import numpy as np

from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer
import pdb

@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type):
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == 'coco':
        class_names = COCODataset.class_names
    else:
        raise NotImplementedError('Not implemented now.')
    device = torch.device(cfg.MODEL.DEVICE)
    smoke_name_dic = ('__background__','一次性快餐盒','书籍纸张','充电宝',
                  '剩饭剩菜' ,'包','垃圾桶','塑料器皿','塑料玩具','塑料衣架',
                   '大骨头','干电池','快递纸袋','插头电线','旧衣服','易拉罐',
                   '枕头','果皮果肉','毛绒玩具','污损塑料','污损用纸','洗护用品',
                   '烟蒂','牙签','玻璃器皿','砧板','筷子','纸盒纸箱','花盆',
                   '茶叶渣','菜帮菜叶','蛋壳','调料瓶','软膏','过期药物',
                   '酒瓶','金属厨具','金属器皿','金属食品罐','锅','陶瓷器皿',
                   '鞋','食用油桶','饮料瓶','鱼骨')

    model = build_detection_model(cfg)
    cpu_device = torch.device("cpu")
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    mkdir(output_dir)

    transforms = build_transforms(cfg, is_train=False)
    model.eval()
    miss = 0

    for i, image_path in enumerate(image_paths):
        start = time.time()
        image_name = os.path.basename(image_path)
        cv_image = cv2.imread(image_path)
        PIL_image = Image.open(image_path)

        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        result = model(images.to(device))[0]
        inference_time = time.time() - start

        result = result.resize((width, height)).to(cpu_device).numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]

        miss = miss + (1 - len(boxes))
        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ]
        )
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))

        draw_ = ImageDraw.Draw(PIL_image)
        for c in range(len(scores)):
            text = smoke_name_dic[labels[c]]
            font = ImageFont.truetype('/usr/share/fonts/truetype/arphic/uming.ttc', 40)
            draw_.text((int(boxes[c][0]) + 2, int(boxes[c][1]) - 2), text, (255, 0, 0), font=font)

        cv_image = cv2.cvtColor(np.asarray(PIL_image), cv2.COLOR_RGB2BGR)
        for c in range(len(scores)):
            cv2.rectangle(cv_image, (int(boxes[c][0]), int(boxes[c][1])), (int(boxes[c][2]), int(boxes[c][3])),
                          (0, 0, 255), 4)
        cv2.imwrite(os.path.join(output_dir, image_name), cv_image)
    smoke_count = len(image_paths)
    print("出现：%d 漏掉： %d 漏检率：%.2f" % (smoke_count, miss, miss / smoke_count))
    # print(len(label_list))

    # print("漏检率: %.3f"%(miss/smoke_count))


def main():
    parser = argparse.ArgumentParser(description='SSD demo.')
    parser.add_argument(
        "--config-file",
        default="configs/resnet50_224.yaml",  # "configs/vgg_ssd300_voc0712.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--images_dir", default='demo/test_trash/', type=str,
                        help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='demo/result/5.18/', type=str,
                        help='Specify a image dir to save predicted images.')
    parser.add_argument("--dataset_type", default="voc", type=str,
                        help='Specify dataset type. Currently support voc and coco.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir,
             dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()
