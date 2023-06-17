import pandas as pd
import numpy as np
import path
import json
import os
import pathlib
import os.path as osp
import sys
import random
import time
import glob
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
import glob
import os
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from torch.utils.data import Dataset, DataLoader
import gc
from timm.data import ImageDataset, create_loader
from functools import lru_cache
import torch
from glob import glob
from PIL import Image
import ffmpeg
from typing import Dict, List, Optional, Tuple, Union
from torch import nn, Tensor
from torchvision import ops
from torchvision.transforms import functional as F, InterpolationMode, transforms as T
import math
import sys
import pickle

from typing import Dict, List, Optional, Tuple, Union
from torch import nn, Tensor
from torchvision import ops
from torchvision.transforms import functional as F, InterpolationMode, transforms as T
from PIL import Image
#----------------------------------------------------
train_dir = './train'
video_path = os.path.join(train_dir, 'scene_00.mp4')
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print('width:', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print('height:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('fps:', fps)
print('frame_count:', frame_count)
images = []
elapsed_times = []
frame_id = 0
elapsed_sec = 0

num_frames = 0
appear_frame = {'要補修-1.区画線': 0,
                '要補修-2.道路標識': 0,
                '要補修-3.照明': 0,
                '補修不要-1.区画線': 0,
                '補修不要-2.道路標識': 0,
                '補修不要-3.照明': 0}

colors = {
    '要補修-1.区画線': (1, 0, 0, 1),    # 赤
    '要補修-2.道路標識': (0, 1, 0, 1),  # 緑
    '要補修-3.照明': (0, 0, 1, 1),      # 青
    '補修不要-1.区画線': (1, 1, 0, 1),   # 黄
    '補修不要-2.道路標識': (1, 0, 1, 1), # 紫
    '補修不要-3.照明': (0, 1, 1, 1)     # 水
}

texts = {
    '要補修-1.区画線': 'need to repair: line',    # 赤
    '要補修-2.道路標識': 'need to repair: sign',  # 緑
    '要補修-3.照明': 'need to repair: light',      # 青
    '補修不要-1.区画線': 'no need to repair: line',   # 黄
    '補修不要-2.道路標識': 'no need to repair: sign', # 紫
    '補修不要-3.照明': 'no need to repair: light'     # 水
}
#----------------------------------------------------

def vis_annotation(image, elapsed_time, ann, colors, texts):
    fig, ax = plt.subplots(figsize=(15,15))
    ax.imshow(image)
    ax.set_title(elapsed_time, fontsize=15)
    for l, bboxes in ann['labels'].items():
        print('{}: {}'.format(l, texts[l]))
        for bbox in bboxes:
            left = min(bbox[0][0], bbox[1][0])
            top = min(bbox[0][1], bbox[1][1])
            width = abs(bbox[1][0]-bbox[0][0])
            height = abs(bbox[1][1]-bbox[0][1])
            rect = patches.Rectangle((left, top), width, height,
                                     linewidth=1,
                                     edgecolor=colors[l],
                                     facecolor='none')
            ax.add_patch(rect)
            bbox_props = dict(boxstyle='square,pad=0', linewidth=1, facecolor=colors[l], edgecolor=colors[l])
            ax.text(left, top, texts[l], ha='left', va='bottom', rotation=0,size=15, bbox=bbox_props)
    plt.show()

def cut_video(video_path, out_img_dir):
    print('\n'+video_path)
    frame_id = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('  width:', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print('  height:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('  fps:', fps)
    print('  frame_count:', frame_count)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_filename = '{}_{:03d}.jpg'.format(video_name, frame_id)
        cv2.imwrite(os.path.join(out_img_dir, out_filename), frame)
        count += 1
        frame_id += 1

    cap.release()
    print('total frames:', count)

class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, root, labels, transforms):
        self.root = root
        self.labels = labels
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = sorted(glob(os.path.join(root, '*.jpg')))
        self.annotations = sorted(glob(os.path.join(root, '*.json')))

    def __getitem__(self, idx):
        # load images and annotations
        img_path = self.imgs[idx]
        annotation_path = self.annotations[idx]
        img = Image.open(img_path).convert('RGB')
        with open(annotation_path, encoding='utf-8') as f:
            annotation = json.load(f)

        # get bounding box coordinates and labels
        boxes = []
        labels = []
        for k, v in annotation['labels'].items():
            labels += [self.labels[k]]*len(v)
            for leftright in v:
                boxes.append(leftright[0]+leftright[1])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = image.shape
                target['boxes'][:, [0, 2]] = width - target['boxes'][:, [2, 0]]
        return image, target


class PILToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target


class ConvertImageDtype(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.convert_image_dtype(image, self.dtype)
        return image, target



def get_transform(train):
    transforms = []
    transforms.append(PILToTensor())
    transforms.append(ConvertImageDtype(torch.float))
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

labels = {'要補修-1.区画線':1, '要補修-2.道路標識':2, '要補修-3.照明':3, '補修不要-1.区画線':4, '補修不要-2.道路標識':5, '補修不要-3.照明':6}
frame_dataset = FrameDataset('./train_data', labels, get_transform(train=True))

# load a pre-trained model for classification and return
# only the features
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# FasterRCNN needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 1280
# so we need to add it here
backbone.out_channels = 1280

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=7,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    size = len(data_loader)
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        if (i+1) %print_freq==0:
            loss_each = {k: round(float(v), 3) for k, v in loss_dict.items()}
            print('[{}/{}] Loss: {}, {}'.format(i+1, size, round(float(losses),3), loss_each))
        if not math.isfinite(losses):
            print(f"Loss is {losses}, stopping training")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()


def collate_fn(batch):
    return tuple(zip(*batch))

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    frame_dataset, batch_size=2, shuffle=True,
    collate_fn=collate_fn)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# let's train it for 3 epochs
num_epochs = 3

for epoch in range(num_epochs):
    print('Epoch {}:'.format(epoch))
    # train for one epoch, printing every 100 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
    # update the learning rate
    lr_scheduler.step()

img, _ = get_transform(False)(Image.fromarray(image), None)
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

pred_cpu = {k: v.to('cpu').detach().numpy() for k, v in prediction[0].items()}
pred = {k: v[np.where(pred_cpu['scores']>=0.5)] for k, v in pred_cpu.items()}

label_list = ['要補修-1.区画線', '要補修-2.道路標識', '要補修-3.照明', '補修不要-1.区画線', '補修不要-2.道路標識', '補修不要-3.照明']
ann_pred = {'labels':{}}
for i, label in enumerate(pred['labels']):
    if label_list[label-1] not in ann_pred['labels']:
        ann_pred['labels'][label_list[label-1]] = []
    bboxes = pred['boxes'][i]
    ann_pred['labels'][label_list[label-1]].append([[float(bboxes[0]), float(bboxes[1])],[float(bboxes[2]), float(bboxes[3])]])

with open('detection_model.pkl', 'wb') as f:
    pickle.dump(model, f)


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.

        Returns:
            bool: The return value. True for success.
        """
        cls.label_list = ['要補修-1.区画線', '要補修-2.道路標識', '要補修-3.照明', '補修不要-1.区画線', '補修不要-2.道路標識', '補修不要-3.照明']
        cls.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        cls.model = load_model(model_path)
        cls.model.eval()

        return True

    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input: Data of the sample you want to make inference from (str)

        Returns:
            list: Inference for the given input.

        """
        prediction = []
        cap = cv2.VideoCapture(input)
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if ret:
                preprocessed = preprocess_frame(frame)
                with torch.no_grad():
                    pred = cls.model([preprocessed.to(cls.device)])
                    postprocessed = postprocess_pred(pred, frame_id, cls.label_list)
                prediction.append(postprocessed)
                frame_id += 1
            else:
                break

        return prediction


def load_model(model_path):
    with open(os.path.join(model_path, 'detection_model.pkl'), 'rb') as f:
        model = pickle.load(f)

    return model


# PILのImageに変換後torch.tensorにする
def preprocess_frame(frame):
    preprocessed, _ = get_transform(False)(Image.fromarray(frame), None)

    return preprocessed


# 確信度が0.5以上の矩形に絞り, 要補修物体が存在する場合は1, それ以外は0とする
def postprocess_pred(pred, frame_id, label_list):
    """
    returns: dict
    """
    pred_cpu = {k: v.to('cpu').detach().numpy() for k, v in pred[0].items()}
    pred_filtered = {k: v[np.where(pred_cpu['scores'] >= 0.5)] for k, v in pred_cpu.items()}
    pred_labels = [label_list[i - 1] for i in pred_filtered['labels']]
    postprocessed = {'frame_id': frame_id, 'line': 0, 'sign': 0, 'light': 0}
    for pred_label in pred_labels:
        if pred_label == '要補修-1.区画線':
            postprocessed['line'] = 1
        elif pred_label == '要補修-2.道路標識':
            postprocessed['sign'] = 1
        elif pred_label == '要補修-3.照明':
            postprocessed['light'] = 1

    return postprocessed


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
            self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = image.shape
                target['boxes'][:, [0, 2]] = width - target['boxes'][:, [2, 0]]
        return image, target


class PILToTensor(nn.Module):
    def forward(
            self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target


class ConvertImageDtype(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
            self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


def get_transform(train):
    transforms = []
    transforms.append(PILToTensor())
    transforms.append(ConvertImageDtype(torch.float))
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)