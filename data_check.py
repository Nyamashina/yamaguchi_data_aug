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
from sklearn.model_selection import train_test_split
import random
import math
import gc
from tqdm import tqdm
import time
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import matthews_corrcoef
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data

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
train_dir = './train'
train_path = 'train/'
test_path = 'test/'
json_file = glob.glob(train_path + '*.json')
json_file = [os.path.basename(json_f) for json_f in json_file]
mp4_file = glob.glob(train_path + '*.mp4')
mp4_file = [os.path.basename(mp4) for mp4 in mp4_file]
test_mp4_file = glob.glob(test_path + '*.mp4')
test_mp4_file = [os.path.basename(mp4) for mp4 in test_mp4_file]

video_path = os.path.join(train_dir, 'scene_00.mp4')
print('video_path:', video_path)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print('width:', width)
print('height:', height)
print('fps:', fps)
print('frame_count:', frame_count)
images = []
elapsed_times = []
frame_id = 0
elapsed_sec = 0

while True:
    ret, frame = cap.read()
    hours, minutes = elapsed_sec // 3600, elapsed_sec % 3600
    minutes, seconds = minutes // 60, minutes % 60
    h, m, s, ms = '{:02d}'.format(int(hours)), '{:02d}'.format(int(minutes)), '{:02d}'.format(
         int(seconds)), '{:05d}'.format(int((seconds - int(seconds)) * (10 ** 5)))
    frame_id += 1
    elapsed_sec = frame_id / fps
    if not ret:
        break
    else:
        images.append(frame[:, :, ::-1])
        elapsed_times.append('{}:{}:{}:{}'.format(h, m, s, ms))
cap.release()

annotation_path = os.path.join(train_dir, 'scene_00.json')


def vis_annotation(image, elapsed_time, ann, colors, texts):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image)
    ax.set_title(elapsed_time, fontsize=15)
    for l, bboxes in ann['labels'].items():
        print('{}: {}'.format(l, texts[l]))
        for bbox in bboxes:
            left = min(bbox[0][0], bbox[1][0])
            top = min(bbox[0][1], bbox[1][1])
            width = abs(bbox[1][0] - bbox[0][0])
            height = abs(bbox[1][1] - bbox[0][1])
            rect = patches.Rectangle((left, top), width, height,linewidth=1,edgecolor=colors[l],facecolor='none')
            ax.add_patch(rect)
            bbox_props = dict(boxstyle='square,pad=0', linewidth=1, facecolor=colors[l], edgecolor=colors[l])
            ax.text(left, top, texts[l], ha='left', va='bottom', rotation=0, size=15, bbox=bbox_props)
    plt.show()


frame_id = 10
image = images[frame_id]
elapsed_time = elapsed_times[frame_id]
fig, ax = plt.subplots(figsize=(15, 15))
ax.imshow(image)
ax.set_title(elapsed_time, fontsize=15)

with open(annotation_path, encoding='utf-8') as f:
    annotation = json.load(f)
    ann = sorted(annotation, key=lambda x: x['frame_id'])[frame_id]
    vis_annotation(image, elapsed_time, ann, colors, texts)

total_bboxes = {'要補修-1.区画線': 0,
                '要補修-2.道路標識': 0,
                '要補修-3.照明': 0,
                '補修不要-1.区画線': 0,
                '補修不要-2.道路標識': 0,
                '補修不要-3.照明': 0}
for file_path in os.listdir(train_dir):
    file_name, ext = os.path.splitext(file_path)
    if ext == '.json':
        num_bboxes = {'要補修-1.区画線': 0,
                      '要補修-2.道路標識': 0,
                      '要補修-3.照明': 0,
                      '補修不要-1.区画線': 0,
                      '補修不要-2.道路標識': 0,
                      '補修不要-3.照明': 0}
        with open(os.path.join(train_dir, file_path), encoding='utf-8') as f:
            ann = json.load(f)
        for frames in ann:
            for l, bboxes in frames['labels'].items():
                total_bboxes[l] += len(bboxes)
                num_bboxes[l] += len(bboxes)
        print(file_name)
        for k, v in num_bboxes.items():
            print('  {}: {}'.format(k, v))
print('\ntotal')
for k, v in total_bboxes.items():
    print('  {}: {}'.format(k, v))

num_frames = 0
appear_frame = {'要補修-1.区画線': 0,
                '要補修-2.道路標識': 0,
                '要補修-3.照明': 0,
                '補修不要-1.区画線': 0,
                '補修不要-2.道路標識': 0,
                '補修不要-3.照明': 0}
for file_path in os.listdir(train_dir):
    file_name, ext = os.path.splitext(file_path)
    if ext == '.json':
        appear = {'要補修-1.区画線': 0,
                      '要補修-2.道路標識': 0,
                      '要補修-3.照明': 0,
                      '補修不要-1.区画線': 0,
                      '補修不要-2.道路標識': 0,
                      '補修不要-3.照明': 0}
        with open(os.path.join(train_dir, file_path), encoding='utf-8') as f:
            ann = json.load(f)
        for frames in ann:
            for l, bboxes in frames['labels'].items():
                appear[l] += 1
                appear_frame[l] += 1
            print(file_name)
    for k, v in appear.items():
        print('  {}: {}/{}'.format(k, v, len(ann)))
    num_frames += len(ann)
print('\ntotal')
for k, v in appear_frame.items():
    print('  {}: {}/{}  ({}%)'.format(k, v, num_frames, round(100 * v / num_frames, 2)))

plt.show()